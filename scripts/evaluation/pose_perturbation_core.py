#!/usr/bin/env python3
"""Pose robustness smoke for the LAIGN Stage-2 localization bridge."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.model import train_residue_localizer as stage2  # noqa: E402


OUT_JSON = ROOT / "outputs/pose_robustness_perturbation_smoke_summary.json"
OUT_TABLE = ROOT / "outputs/tables/table_pose_robustness_perturbation_smoke.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints/interaction_graph_visnet_hardneg_full_raw10_seed2401/best.pt"))
    parser.add_argument(
        "--ensemble-checkpoints",
        nargs="+",
        default=[
            str(ROOT / "checkpoints/interaction_graph_visnet_hardneg_full_raw10_seed2402/best.pt"),
            str(ROOT / "checkpoints/interaction_graph_visnet_hardneg_full_raw10_seed2403/best.pt"),
        ],
    )
    parser.add_argument("--max-train-graphs", type=int, default=1200)
    parser.add_argument("--max-internal-val-graphs", type=int, default=300)
    parser.add_argument("--max-external-val-graphs", type=int, default=120)
    parser.add_argument("--max-external-test-graphs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=3401)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experiment-id", default="Perturbation")
    parser.add_argument("--scope", default="smoke")
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--table", default=str(OUT_TABLE))
    return parser.parse_args()


def stage2_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        split_manifest=str(ROOT / "data/processed/structure_supervision/structure_supervision_split_manifest.csv.gz"),
        biolip_interactions=str(ROOT / "data/processed/biolip2/biolip2_nr_plip_interaction_labels.csv.gz"),
        plinder_interactions=str(ROOT / "data/processed/plinder/plinder_plip_interaction_labels.csv.gz"),
        biolip_site_labels=str(ROOT / "data/processed/biolip2/biolip2_nr_plip_site_labels.csv.gz"),
        plinder_site_labels=str(ROOT / "data/processed/plinder/plinder_plip_site_labels.csv.gz"),
        label_mode="active_interaction",
        checkpoint=args.checkpoint,
        ensemble_checkpoints=args.ensemble_checkpoints,
        ensemble_uncertainty_features=False,
        interaction_backbone="visnet",
        seed=args.seed,
        max_train_graphs=args.max_train_graphs,
        max_internal_val_graphs=args.max_internal_val_graphs,
        max_external_val_graphs=args.max_external_val_graphs,
        max_external_test_graphs=args.max_external_test_graphs,
        candidate_radius=999.0,
        positive_interactions=stage2.DEFAULT_ACTIVE_INTERACTIONS,
        score_interactions=stage2.DEFAULT_ACTIVE_INTERACTIONS,
        stage2_feature_mode="typed_multiscale",
        c_grid=[0.01, 0.1, 1.0, 10.0],
        stage2_classifier="logreg",
        hgbt_l2_grid=[0.0],
        hgbt_leaf_grid=[15],
        hgbt_max_iter=120,
        hgbt_learning_rate=0.06,
        max_iter=1000,
        visnet_control="normal",
        control_seed=args.seed,
        device=args.device,
        finite_representation_guard=False,
        representation_clip=0.0,
    )


def rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    axis = rng.normal(size=3)
    axis = axis / max(np.linalg.norm(axis), 1e-8)
    angle = float(rng.uniform(-math.pi, math.pi))
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float32,
    )


def transformed_record(record: dict[str, object], mode: str, seed: int) -> dict[str, object]:
    if mode == "native":
        return record
    out = dict(record)
    out["data"] = record["data"].clone()
    out["labels"] = np.asarray(record["labels"]).copy()
    out["residue_for_pair"] = np.asarray(record["residue_for_pair"]).copy()
    out["pair_residue_type"] = np.asarray(record["pair_residue_type"]).copy()
    out["pair_ligand_element"] = np.asarray(record["pair_ligand_element"]).copy()
    out["residue_centroids"] = np.asarray(record["residue_centroids"]).copy()
    rng = np.random.default_rng(seed)

    data = out["data"]
    pos = data.pos.detach().cpu().numpy().astype(np.float32)
    ligand_nodes = torch.unique(data.pair_index[1]).detach().cpu().numpy().astype(np.int64)
    protein_nodes = torch.unique(data.pair_index[0]).detach().cpu().numpy().astype(np.int64)
    ligand_pos = pos[ligand_nodes]
    centroid = ligand_pos.mean(axis=0)

    if mode.startswith("rigid_"):
        magnitude = float(mode.split("_", 1)[1].replace("A", ""))
        direction = rng.normal(size=3).astype(np.float32)
        direction = direction / max(float(np.linalg.norm(direction)), 1e-8)
        rotated = (ligand_pos - centroid) @ rotation_matrix(rng).T + centroid
        pos[ligand_nodes] = rotated + direction * magnitude
    elif mode == "wrong_pocket_centroid":
        protein_pos = pos[protein_nodes]
        distances = np.sqrt(np.square(protein_pos - centroid[None, :]).sum(axis=1))
        far = protein_nodes[distances >= np.quantile(distances, 0.70)]
        target_node = int(rng.choice(far if far.size else protein_nodes))
        shift = pos[target_node] - centroid + rng.normal(scale=1.0, size=3).astype(np.float32)
        pos[ligand_nodes] = ligand_pos + shift
    else:
        raise ValueError(mode)

    data.pos = torch.tensor(pos, dtype=torch.float32)
    pair_index = data.pair_index.detach().cpu().numpy().astype(np.int64)
    pair_dist = np.sqrt(np.square(pos[pair_index[0]] - pos[pair_index[1]]).sum(axis=1)).astype(np.float32)
    out["pair_distances"] = np.stack([pair_dist, pair_dist], axis=1).astype(np.float32)
    n_residues = int(out["labels"].shape[0])
    residue_distance = np.full(n_residues, np.inf, dtype=np.float32)
    np.minimum.at(residue_distance, out["residue_for_pair"], pair_dist)
    residue_distance[~np.isfinite(residue_distance)] = 80.0
    out["distances"] = residue_distance.astype(np.float64)
    return out


def load_models(args2: SimpleNamespace, device: torch.device):
    score_args = SimpleNamespace(
        finite_representation_guard=args2.finite_representation_guard,
        representation_clip=args2.representation_clip,
    )
    class_indices = stage2.contact_eval.active_indices(args2.score_interactions)
    model, history = stage2.contact_eval.load_model(args2.checkpoint, score_args, device)
    models = [model]
    histories = [{"checkpoint": args2.checkpoint, "history": history}]
    for checkpoint in args2.ensemble_checkpoints or []:
        extra_model, extra_history = stage2.contact_eval.load_model(checkpoint, score_args, device)
        models.append(extra_model)
        histories.append({"checkpoint": checkpoint, "history": extra_history})
    return models, class_indices, histories


def score_split(split_data: dict[str, object], classifier, args2: SimpleNamespace) -> dict[str, object]:
    x, _ = stage2.feature_matrix(split_data, "stage2", args2.stage2_feature_mode)
    scores = classifier.predict_proba(x)[:, 1]
    raw_distance = stage2.raw_scores(split_data, "raw_distance")
    fixed_hybrid = stage2.raw_scores(split_data, "fixed_hybrid")
    return {
        "stage2": stage2.summarize_scores(split_data, scores),
        "raw_distance": stage2.summarize_scores(split_data, raw_distance),
        "fixed_hybrid": stage2.summarize_scores(split_data, fixed_hybrid),
    }


def main() -> int:
    args = parse_args()
    args2 = stage2_namespace(args)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    records, skipped, split_counts = stage2.load_candidate_records(args2)
    models, class_indices, histories = load_models(args2, device)

    native_arrays = stage2.assemble_split_arrays(records, models, class_indices, device, args2)
    classifier, best, trace, feature_names = stage2.fit_variant("stage2", native_arrays, args2)

    modes = ["native", "rigid_0.5A", "rigid_1A", "rigid_2A", "rigid_5A", "rigid_10A", "wrong_pocket_centroid"]
    results: dict[str, object] = {}
    for mode in modes:
        mode_records = []
        for idx, record in enumerate(records):
            if record["split"] not in {"external_val", "external_test"}:
                continue
            mode_records.append(transformed_record(record, mode, args.seed + idx * 1009 + len(mode)))
        mode_arrays = stage2.assemble_split_arrays(mode_records, models, class_indices, device, args2)
        results[mode] = {}
        for split in ["external_val", "external_test"]:
            results[mode][split] = score_split(mode_arrays[split], classifier, args2)

    payload = {
        "hypothesis": args.experiment_id,
        "scope": args.scope,
        "limits": {
            "max_train_graphs": args.max_train_graphs,
            "max_internal_val_graphs": args.max_internal_val_graphs,
            "max_external_val_graphs": args.max_external_val_graphs,
            "max_external_test_graphs": args.max_external_test_graphs,
        },
        "split_counts": split_counts,
        "skipped": skipped,
        "classifier_selection": best,
        "feature_count": len(feature_names),
        "device": str(device),
        "histories": histories,
        "results": results,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_table(payload, Path(args.table))
    print(json.dumps({"json": str(out_json), "table": args.table}))
    return 0


def write_table(payload: dict[str, object], path: Path) -> None:
    lines = [
        f"# {payload['hypothesis']} Pose Robustness {str(payload['scope']).title()}",
        "",
        f"- scope: {payload['scope']}",
        f"- train/internal/external_val/external_test limits: {payload['limits']}",
        f"- feature count: {payload['feature_count']}",
        "",
        "| pose mode | split | method | AUPRC | AUROC | graph AUPRC | recall@5 | precision@5 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    results = payload["results"]
    assert isinstance(results, dict)
    for mode, split_rows in results.items():
        assert isinstance(split_rows, dict)
        for split, method_rows in split_rows.items():
            assert isinstance(method_rows, dict)
            for method in ["raw_distance", "fixed_hybrid", "stage2"]:
                row = method_rows[method]
                public = row["public_rank_metrics"]
                lines.append(
                    f"| {mode} | {split} | {method} | {row['auprc']:.4f} | {row['auroc']:.4f} | "
                    f"{row['mean_graph_auprc']['mean']:.4f} | {public['recall_at_5']['mean']:.4f} | "
                    f"{public['precision_at_5']['mean']:.4f} |"
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
