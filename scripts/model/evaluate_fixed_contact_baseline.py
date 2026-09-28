#!/usr/bin/env python3
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.data import build_interaction_graph_dataset as graph_builder  # noqa: E402
from scripts.data import build_interaction_pair_features as pair_features  # noqa: E402
from scripts.model import train_interaction_scorer as train_visnet  # noqa: E402


DEFAULT_ACTIVE_INTERACTIONS = [
    "hydrophobic_contact",
    "hydrogen_bond",
    "salt_bridge",
    "pi_stacking",
    "pi_cation",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PLIP active-interaction residue localization in loose "
            "contact windows built from PLINDER systems."
        )
    )
    parser.add_argument(
        "--split-manifest",
        default=str(
            ROOT
            / "data/processed/structure_supervision/structure_supervision_split_manifest.csv.gz"
        ),
    )
    parser.add_argument(
        "--interaction-labels",
        default=str(ROOT / "data/processed/plinder/plinder_plip_interaction_labels.csv.gz"),
    )
    parser.add_argument("--model", choices=["distance", "visnet"], default="distance")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["external_val", "external_test"],
        choices=["external_val", "external_test"],
    )
    parser.add_argument(
        "--candidate-radii",
        nargs="+",
        type=float,
        default=[4.0, 5.0, 6.0, 8.0, 10.0, 12.0],
        help="Residue heavy-atom to ligand heavy-atom distance cutoffs.",
    )
    parser.add_argument(
        "--positive-interactions",
        nargs="+",
        default=DEFAULT_ACTIVE_INTERACTIONS,
        choices=train_visnet.INTERACTION_TYPES,
    )
    parser.add_argument("--max-graphs-per-split", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--finite-representation-guard", action="store_true")
    parser.add_argument("--representation-clip", type=float, default=0.0)
    return parser.parse_args()


def safe_metric(fn, labels, scores):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or labels.min() == labels.max():
        return math.nan
    return float(fn(labels, scores))


def describe(values):
    clean = np.asarray([value for value in values if not math.isnan(value)], dtype=np.float64)
    if clean.size == 0:
        return {"n": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "n": int(clean.size),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def active_indices(interaction_names):
    return [train_visnet.INTERACTION_TYPES.index(name) for name in interaction_names]


def load_model(checkpoint_path, args, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    model = train_visnet.PairViSNet(
        hidden=int(ckpt_args.get("hidden", 64)),
        layers=int(ckpt_args.get("layers", 3)),
        heads=int(ckpt_args.get("heads", 8)),
        num_rbf=int(ckpt_args.get("num_rbf", 16)),
        cutoff=float(ckpt_args.get("cutoff", 5.0)),
        max_neighbors=int(ckpt_args.get("max_neighbors", 64)),
        dropout=float(ckpt_args.get("dropout", 0.10)),
        finite_representation_guard=args.finite_representation_guard,
        representation_clip=args.representation_clip,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint.get("history", [])


def build_label_context(interaction_labels, positive_interactions):
    interactions = pd.read_csv(interaction_labels)
    positive_interactions = set(positive_interactions)
    positive_residues = defaultdict(set)
    ligand_serials = defaultdict(set)
    for row in interactions.to_dict("records"):
        sample_key = (row["system_id"], row["ligand_id"])
        try:
            ligand_serials[sample_key].add(int(row["ligand_atom_serial"]))
        except Exception:
            pass
        if row["interaction_type"] not in positive_interactions:
            continue
        residue_key = (
            pair_features.clean_str(row["protein_chain"]),
            pair_features.clean_resnr(row["protein_residue_number"]),
            pair_features.clean_str(row["protein_residue_type"]),
        )
        positive_residues[sample_key].add(residue_key)
    return positive_residues, ligand_serials


def unique_standard_residues(residue_atoms):
    residues = []
    seen = set()
    for key in residue_atoms:
        chain, resnr, resname = key
        if not resname or resname not in pair_features.AA3:
            continue
        if key in seen:
            continue
        seen.add(key)
        residues.append(key)
    return sorted(residues)


def min_ligand_distance(residue_atoms, ligand_atoms):
    protein_xyz = np.stack([atom["xyz"] for atom in residue_atoms if atom["element"] != "H"], axis=0)
    ligand_xyz = np.stack([atom["xyz"] for atom in ligand_atoms], axis=0)
    distances = np.sqrt(np.square(protein_xyz[:, None, :] - ligand_xyz[None, :, :]).sum(axis=2))
    return float(distances.min())


def selected_ligand_atoms(row, ligand_serials_by_sample):
    sample_key = (row["system_id"], row["ligand_id"])
    positive_serials = ligand_serials_by_sample.get(sample_key, set())
    _, ligand_serials = pair_features.choose_binding_site(row, positive_serials)
    pdb_atoms, residue_atoms = pair_features.parse_pdb_atoms(row["plip_pdb"])
    ligand_atoms = []
    for serial in ligand_serials:
        atom = pdb_atoms.get(int(serial))
        if atom is not None and atom["element"] != "H":
            ligand_atoms.append(atom)
    return pdb_atoms, residue_atoms, ligand_atoms


def build_candidate_record(row, positive_residues, ligand_serials_by_sample, max_radius):
    sample_key = (row["system_id"], row["ligand_id"])
    try:
        _, residue_atoms, ligand_atoms = selected_ligand_atoms(row, ligand_serials_by_sample)
    except Exception as exc:
        return None, f"parse_or_ligand_error:{exc}"
    if not ligand_atoms:
        return None, "no_ligand_atoms"

    positives = positive_residues.get(sample_key, set())
    candidates = []
    for residue_key in unique_standard_residues(residue_atoms):
        atoms = graph_builder.get_residue_atoms(residue_atoms, residue_key)
        if not atoms:
            continue
        distance = min_ligand_distance(atoms, ligand_atoms)
        if distance <= max_radius:
            candidates.append(
                {
                    "residue_key": residue_key,
                    "atoms": atoms,
                    "distance": distance,
                    "label": int(residue_key in positives),
                }
            )
    if not candidates:
        return None, "no_candidates"

    node_type = []
    positions = []
    occupied_positions = set()
    residue_node = {}
    residue_records = []
    kept = []

    for candidate in candidates:
        atoms = candidate["atoms"]
        coords = np.stack([atom["xyz"] for atom in atoms], axis=0)
        centroid = coords.mean(axis=0).astype(np.float32)
        position_key = tuple(centroid.tolist())
        if position_key in occupied_positions:
            continue
        residue_key = candidate["residue_key"]
        residue_node[residue_key] = len(node_type)
        node_type.append(1 + graph_builder.residue_id(residue_key[2]))
        positions.append(centroid)
        occupied_positions.add(position_key)
        residue_records.append((atoms, centroid))
        kept.append(candidate)

    for atoms, centroid in residue_records:
        for atom in atoms:
            xyz = np.asarray(atom["xyz"], dtype=np.float32)
            position_key = tuple(xyz.tolist())
            if np.linalg.norm(xyz - centroid) < 1e-4 or position_key in occupied_positions:
                continue
            node_type.append(graph_builder.PROTEIN_ELEMENT_OFFSET + graph_builder.element_id(atom["element"]))
            positions.append(xyz)
            occupied_positions.add(position_key)

    ligand_node = {}
    ligand_positions = set()
    for atom in ligand_atoms:
        xyz = np.asarray(atom["xyz"], dtype=np.float32)
        position_key = tuple(xyz.tolist())
        if position_key in occupied_positions or position_key in ligand_positions:
            continue
        ligand_node[atom["serial"]] = len(node_type)
        node_type.append(graph_builder.LIGAND_ELEMENT_OFFSET + graph_builder.element_id(atom["element"]))
        positions.append(xyz)
        ligand_positions.add(position_key)
    if not ligand_node or not kept:
        return None, "empty_after_duplicate_filter"

    pairs = []
    residue_for_pair = []
    for residue_index, candidate in enumerate(kept):
        protein_index = residue_node[candidate["residue_key"]]
        for ligand_index in ligand_node.values():
            pairs.append((protein_index, ligand_index))
            residue_for_pair.append(residue_index)

    data = Data(
        z=torch.tensor(node_type, dtype=torch.long),
        pos=torch.tensor(np.stack(positions), dtype=torch.float32),
        pair_index=torch.tensor(pairs, dtype=torch.long).T.contiguous(),
        num_nodes=len(node_type),
    )
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    return {
        "sample_id": row["sample_id"],
        "split": row["supervision_split"],
        "data": data,
        "labels": np.asarray([candidate["label"] for candidate in kept], dtype=np.int32),
        "distances": np.asarray([candidate["distance"] for candidate in kept], dtype=np.float64),
        "residue_for_pair": np.asarray(residue_for_pair, dtype=np.int64),
    }, ""


def load_candidate_records(args, max_radius):
    manifest = pd.read_csv(args.split_manifest, low_memory=False)
    manifest = manifest[
        (manifest["source"] == "plinder") & (manifest["supervision_split"].isin(args.splits))
    ].copy()
    manifest = manifest.sort_values(["supervision_split", "sample_id"])
    if args.max_graphs_per_split > 0:
        manifest = (
            manifest.groupby("supervision_split", group_keys=False)
            .head(args.max_graphs_per_split)
            .reset_index(drop=True)
        )

    positive_residues, ligand_serials = build_label_context(
        args.interaction_labels,
        args.positive_interactions,
    )
    records = []
    skipped = defaultdict(int)
    for row in manifest.to_dict("records"):
        record, reason = build_candidate_record(row, positive_residues, ligand_serials, max_radius)
        if record is None:
            skipped[reason] += 1
            continue
        records.append(record)
    return records, dict(skipped)


@torch.no_grad()
def score_record(record, args, model, class_indices, device):
    if args.model == "distance":
        return -record["distances"]

    data = record["data"].to(device)
    logits = model(data)
    if not torch.isfinite(logits).all():
        raise FloatingPointError(f"Non-finite logits for {record['sample_id']}")
    probs = torch.sigmoid(logits[:, class_indices]).max(dim=1).values.detach().cpu().numpy()
    scores = np.full(record["labels"].shape, -math.inf, dtype=np.float64)
    np.maximum.at(scores, record["residue_for_pair"], probs.astype(np.float64))
    scores[~np.isfinite(scores)] = 0.0
    return scores


def summarize_split_radius(entries):
    labels = np.concatenate([entry["labels"] for entry in entries]) if entries else np.asarray([], dtype=np.int32)
    scores = np.concatenate([entry["scores"] for entry in entries]) if entries else np.asarray([], dtype=np.float64)
    graph_auprc = []
    graph_auroc = []
    graph_positive_rate = []
    for entry in entries:
        y = entry["labels"]
        s = entry["scores"]
        graph_positive_rate.append(float(y.mean()) if y.size else math.nan)
        if y.size == 0 or y.min() == y.max():
            continue
        graph_auprc.append(float(average_precision_score(y, s)))
        graph_auroc.append(float(roc_auc_score(y, s)))
    return {
        "graphs": int(len(entries)),
        "residues": int(labels.size),
        "positives": int(labels.sum()) if labels.size else 0,
        "prevalence": float(labels.mean()) if labels.size else math.nan,
        "auprc": safe_metric(average_precision_score, labels, scores),
        "auroc": safe_metric(roc_auc_score, labels, scores),
        "mean_graph_auprc": describe(graph_auprc),
        "mean_graph_auroc": describe(graph_auroc),
        "mean_graph_positive_rate": describe(graph_positive_rate),
    }


def evaluate_records(records, args, model, class_indices, device):
    scored_by_split_radius = {
        split: {f"{radius:g}": [] for radius in args.candidate_radii}
        for split in args.splits
    }
    for record in records:
        scores = score_record(record, args, model, class_indices, device)
        for radius in args.candidate_radii:
            mask = record["distances"] <= radius
            if not mask.any():
                continue
            scored_by_split_radius[record["split"]][f"{radius:g}"].append(
                {
                    "sample_id": record["sample_id"],
                    "labels": record["labels"][mask],
                    "scores": scores[mask],
                }
            )

    metrics = {}
    for split, by_radius in scored_by_split_radius.items():
        metrics[split] = {}
        for radius_key, entries in by_radius.items():
            metrics[split][radius_key] = summarize_split_radius(entries)
    return metrics


def fmt(value):
    return "nan" if math.isnan(value) else f"{value:.4f}"


def write_table(metrics, path):
    lines = [
        "# PLIP Contact-Residue Localization",
        "",
        f"- model: `{metrics['model']}`",
        f"- checkpoint: `{metrics.get('checkpoint') or 'none'}`",
        f"- positive interactions: `{', '.join(metrics['positive_interactions'])}`",
        f"- finite representation guard: `{str(metrics['finite_representation_guard']).lower()}`",
        "",
        "| split | radius | graphs | residues | positives | prevalence | residue AUPRC | residue AUROC | mean graph AUPRC | mean graph AUROC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in metrics["splits"]:
        for radius_key, row in metrics["splits"][split].items():
            lines.append(
                f"| {split} | {radius_key} | {row['graphs']} | {row['residues']} | "
                f"{row['positives']} | {fmt(row['prevalence'])} | {fmt(row['auprc'])} | "
                f"{fmt(row['auroc'])} | {fmt(row['mean_graph_auprc']['mean'])} | "
                f"{fmt(row['mean_graph_auroc']['mean'])} |"
            )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.candidate_radii = sorted(set(float(radius) for radius in args.candidate_radii))
    max_radius = max(args.candidate_radii)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    class_indices = active_indices(args.positive_interactions)

    if args.model == "visnet":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --model visnet")
        model, history = load_model(args.checkpoint, args, device)
    else:
        model = None
        history = []

    records, skipped = load_candidate_records(args, max_radius)
    split_metrics = evaluate_records(records, args, model, class_indices, device)
    metrics = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "split_manifest": args.split_manifest,
        "interaction_labels": args.interaction_labels,
        "splits_requested": args.splits,
        "candidate_radii": args.candidate_radii,
        "positive_interactions": args.positive_interactions,
        "device": str(device),
        "finite_representation_guard": bool(args.finite_representation_guard),
        "representation_clip": float(args.representation_clip),
        "records_loaded": int(len(records)),
        "records_skipped": skipped,
        "history": history,
        "splits": split_metrics,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_table(metrics, args.table)
    print(json.dumps({"metrics": str(metrics_path), "table": args.table}))


if __name__ == "__main__":
    main()
