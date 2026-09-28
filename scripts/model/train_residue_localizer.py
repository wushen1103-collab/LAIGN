#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.data import build_interaction_graph_dataset as graph_builder  # noqa: E402
from scripts.data import build_interaction_pair_features as pair_features  # noqa: E402
from scripts.model import evaluate_fixed_contact_baseline as contact_eval  # noqa: E402
from scripts.model import train_interaction_scorer as train_visnet  # noqa: E402
from scripts.model import train_pair_mlp as train_pair_mlp  # noqa: E402


DEFAULT_ACTIVE_INTERACTIONS = [
    "hydrophobic_contact",
    "hydrogen_bond",
    "salt_bridge",
    "pi_stacking",
    "pi_cation",
]

SPLIT_ORDER = ["train", "internal_val", "external_val", "external_test"]
SOURCE_FOR_SPLIT = {
    "train": "biolip2_nr",
    "internal_val": "biolip2_nr",
    "external_val": "plinder",
    "external_test": "plinder",
}
TAUS = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a BioLiP-supervised Stage-2 residue localizer from distance and frozen interaction scores."
    )
    parser.add_argument(
        "--split-manifest",
        default=str(ROOT / "data/processed/structure_supervision/structure_supervision_split_manifest.csv.gz"),
    )
    parser.add_argument(
        "--biolip-interactions",
        default=str(ROOT / "data/processed/biolip2/biolip2_nr_plip_interaction_labels.csv.gz"),
    )
    parser.add_argument(
        "--plinder-interactions",
        default=str(ROOT / "data/processed/plinder/plinder_plip_interaction_labels.csv.gz"),
    )
    parser.add_argument(
        "--biolip-site-labels",
        default=str(ROOT / "data/processed/biolip2/biolip2_nr_plip_site_labels.csv.gz"),
    )
    parser.add_argument(
        "--plinder-site-labels",
        default=str(ROOT / "data/processed/plinder/plinder_plip_site_labels.csv.gz"),
    )
    parser.add_argument(
        "--label-mode",
        choices=["active_interaction", "site_label"],
        default="active_interaction",
        help="Residue positives are PLIP active interactions or the conventional 5A/contact site label.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--ensemble-checkpoints",
        nargs="+",
        default=None,
        help="Optional extra ViSNet checkpoints; when set, pair probabilities are averaged across checkpoints.",
    )
    parser.add_argument(
        "--ensemble-uncertainty-features",
        action="store_true",
        help=(
            "When using multiple ViSNet checkpoints, expose per-residue agreement/disagreement features "
            "to the Stage-2 bridge. Defaults off for exact backward compatibility."
        ),
    )
    parser.add_argument(
        "--interaction-backbone",
        choices=["visnet", "pair_mlp"],
        default="visnet",
        help="Frozen interaction model used to produce residue evidence.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument(
        "--predictions-output",
        default=None,
        help="Optional compressed NPZ containing labels, graph groups, and per-method scores for statistical analysis.",
    )
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument("--max-train-graphs", type=int, default=5000)
    parser.add_argument("--max-internal-val-graphs", type=int, default=1000)
    parser.add_argument("--max-external-val-graphs", type=int, default=0)
    parser.add_argument("--max-external-test-graphs", type=int, default=0)
    parser.add_argument("--candidate-radius", type=float, default=999.0)
    parser.add_argument(
        "--positive-interactions",
        nargs="+",
        default=DEFAULT_ACTIVE_INTERACTIONS,
        choices=train_visnet.INTERACTION_TYPES,
    )
    parser.add_argument(
        "--score-interactions",
        nargs="+",
        default=None,
        choices=train_visnet.INTERACTION_TYPES,
        help="Interaction channels exposed as frozen evidence; defaults to positive-interactions.",
    )
    parser.add_argument(
        "--stage2-feature-mode",
        choices=["base", "typed_multiscale", "typed_context"],
        default="base",
        help=(
            "base keeps the Base scalar bridge; typed_multiscale adds type-resolved and multi-scale pooled evidence; "
            "typed_context also adds per-complex rank normalization and residue-neighborhood context."
        ),
    )
    parser.add_argument("--c-grid", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument(
        "--stage2-classifier",
        choices=["logreg", "hgbt"],
        default="logreg",
        help="Classifier used for trained distance/visnet/stage2 variants.",
    )
    parser.add_argument("--hgbt-l2-grid", nargs="+", type=float, default=[0.0, 0.01, 0.1])
    parser.add_argument("--hgbt-leaf-grid", nargs="+", type=int, default=[15, 31])
    parser.add_argument("--hgbt-max-iter", type=int, default=120)
    parser.add_argument("--hgbt-learning-rate", type=float, default=0.06)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--visnet-control",
        choices=["normal", "within_graph_shuffle"],
        default="normal",
        help="Optional corruption control applied to frozen ViSNet residue scores.",
    )
    parser.add_argument(
        "--control-seed",
        type=int,
        default=0,
        help="Seed offset for deterministic ViSNet score corruption controls.",
    )
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


def expected_calibration_error(labels, probs, bins=15):
    labels = np.asarray(labels, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    if labels.size == 0:
        return math.nan
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probs[mask].mean()))
    return float(ece)


def risk_coverage_auc(labels, probs):
    labels = np.asarray(labels, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float64)
    if labels.size == 0:
        return math.nan
    confidence = np.maximum(probs, 1.0 - probs)
    predictions = (probs >= 0.5).astype(np.int32)
    errors = (predictions != labels).astype(np.float64)
    order = np.argsort(-confidence)
    coverage = np.arange(1, labels.size + 1, dtype=np.float64) / float(labels.size)
    risk = np.cumsum(errors[order]) / np.arange(1, labels.size + 1, dtype=np.float64)
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(risk, coverage))


def probability_calibration(labels, probs):
    labels = np.asarray(labels, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float64)
    finite = np.isfinite(probs)
    if labels.size == 0 or not finite.all():
        return None
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return {
        "brier": float(np.mean(np.square(probs - labels))),
        "nll": float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))),
        "ece_15": expected_calibration_error(labels, probs, bins=15),
        "risk_coverage_auc": risk_coverage_auc(labels, probs),
    }


def bounded_probability(scores):
    scores = np.asarray(scores, dtype=np.float64)
    return scores.size > 0 and np.isfinite(scores).all() and float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0


def fit_score_calibrator(labels, scores, seed):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or labels.min() == labels.max() or not np.isfinite(scores).all():
        return None
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs", random_state=seed),
    )
    model.fit(scores.reshape(-1, 1), labels)
    return model


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


def sample_key(row):
    if row["source"] == "biolip2_nr":
        return ("biolip2_nr", row["complex_id"])
    return ("plinder", row["system_id"], row["ligand_id"])


def clean_residue_key(row):
    return (
        pair_features.clean_str(row["protein_chain"]),
        pair_features.clean_resnr(row["protein_residue_number"]),
        pair_features.clean_str(row["protein_residue_type"]),
    )


def load_label_context(args):
    positive_interactions = set(args.positive_interactions)
    positive_residues = defaultdict(set)
    ligand_serials = defaultdict(set)

    biolip = pd.read_csv(args.biolip_interactions)
    for row in biolip.to_dict("records"):
        key = ("biolip2_nr", row["complex_id"])
        try:
            ligand_serials[key].add(int(row["ligand_atom_serial"]))
        except Exception:
            pass
        if args.label_mode == "active_interaction" and row["interaction_type"] in positive_interactions:
            positive_residues[key].add(clean_residue_key(row))

    plinder = pd.read_csv(args.plinder_interactions)
    for row in plinder.to_dict("records"):
        key = ("plinder", row["system_id"], row["ligand_id"])
        try:
            ligand_serials[key].add(int(row["ligand_atom_serial"]))
        except Exception:
            pass
        if args.label_mode == "active_interaction" and row["interaction_type"] in positive_interactions:
            positive_residues[key].add(clean_residue_key(row))

    if args.label_mode == "site_label":
        biolip_sites = pd.read_csv(args.biolip_site_labels)
        for row in biolip_sites.to_dict("records"):
            if int(row.get("site_label", 0)) != 1:
                continue
            positive_residues[("biolip2_nr", row["complex_id"])].add(clean_residue_key(row))

        plinder_sites = pd.read_csv(args.plinder_site_labels)
        for row in plinder_sites.to_dict("records"):
            if int(row.get("site_label", 0)) != 1:
                continue
            positive_residues[("plinder", row["system_id"], row["ligand_id"])].add(clean_residue_key(row))

    return positive_residues, ligand_serials


def select_manifest(args):
    manifest = pd.read_csv(args.split_manifest, low_memory=False)
    selected = []
    limits = {
        "train": args.max_train_graphs,
        "internal_val": args.max_internal_val_graphs,
        "external_val": args.max_external_val_graphs,
        "external_test": args.max_external_test_graphs,
    }
    for index, split in enumerate(SPLIT_ORDER):
        source = SOURCE_FOR_SPLIT[split]
        group = manifest[
            (manifest["source"] == source)
            & (manifest["supervision_split"] == split)
            & (manifest["structure_complete"] == 1)
        ].copy()
        limit = int(limits[split])
        if limit > 0 and len(group) > limit:
            group = group.sample(n=limit, random_state=args.seed + 17 * (index + 1))
        selected.append(group)
    return pd.concat(selected, ignore_index=True).sort_values(["supervision_split", "sample_id"])


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
    key = sample_key(row)
    positive_serials = ligand_serials_by_sample.get(key, set())
    _, ligand_serials = pair_features.choose_binding_site(row, positive_serials)
    if not ligand_serials and positive_serials:
        ligand_serials = sorted(positive_serials)
    pdb_atoms, residue_atoms = pair_features.parse_pdb_atoms(row["plip_pdb"])
    ligand_atoms = []
    for serial in ligand_serials:
        atom = pdb_atoms.get(int(serial))
        if atom is not None and atom["element"] != "H":
            ligand_atoms.append(atom)
    return residue_atoms, ligand_atoms


def build_candidate_record(row, positive_residues_by_sample, ligand_serials_by_sample, max_radius):
    key = sample_key(row)
    try:
        residue_atoms, ligand_atoms = selected_ligand_atoms(row, ligand_serials_by_sample)
    except Exception as exc:
        return None, f"parse_or_ligand_error:{exc}"
    if not ligand_atoms:
        return None, "no_ligand_atoms"

    positives = positive_residues_by_sample.get(key, set())
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
    pair_residue_type = []
    pair_ligand_element = []
    pair_distances = []
    aa_to_id = {aa: idx + 1 for idx, aa in enumerate(pair_features.AA3)}
    element_to_id = {element: idx + 1 for idx, element in enumerate(pair_features.ELEMENTS)}
    ligand_by_serial = {atom["serial"]: atom for atom in ligand_atoms}
    for residue_index, candidate in enumerate(kept):
        protein_index = residue_node[candidate["residue_key"]]
        residue_xyz = np.stack([atom["xyz"] for atom in candidate["atoms"]], axis=0)
        residue_centroid = residue_xyz.mean(axis=0)
        residue_type_id = aa_to_id.get(candidate["residue_key"][2], 0)
        for ligand_serial, ligand_index in ligand_node.items():
            ligand_atom = ligand_by_serial[ligand_serial]
            pairs.append((protein_index, ligand_index))
            residue_for_pair.append(residue_index)
            min_distance, centroid_distance = pair_features.distances_to_residue(
                ligand_atom["xyz"], residue_xyz, residue_centroid
            )
            pair_residue_type.append(residue_type_id)
            pair_ligand_element.append(element_to_id.get(str(ligand_atom["element"]).upper(), 0))
            pair_distances.append((min_distance, centroid_distance))

    data = Data(
        z=torch.tensor(node_type, dtype=torch.long),
        pos=torch.tensor(np.stack(positions), dtype=torch.float32),
        pair_index=torch.tensor(pairs, dtype=torch.long).T.contiguous(),
        num_nodes=len(node_type),
    )
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    return {
        "sample_id": row["sample_id"],
        "source": row["source"],
        "split": row["supervision_split"],
        "data": data,
        "labels": np.asarray([candidate["label"] for candidate in kept], dtype=np.int32),
        "distances": np.asarray([candidate["distance"] for candidate in kept], dtype=np.float64),
        "residue_centroids": np.stack([centroid for _, centroid in residue_records]).astype(np.float32),
        "residue_for_pair": np.asarray(residue_for_pair, dtype=np.int64),
        "pair_residue_type": np.asarray(pair_residue_type, dtype=np.int64),
        "pair_ligand_element": np.asarray(pair_ligand_element, dtype=np.int64),
        "pair_distances": np.asarray(pair_distances, dtype=np.float32),
    }, ""


def load_candidate_records(args):
    positive_residues, ligand_serials = load_label_context(args)
    manifest = select_manifest(args)
    records = []
    skipped = defaultdict(int)
    split_counts = defaultdict(int)
    for row in manifest.to_dict("records"):
        record, reason = build_candidate_record(
            row,
            positive_residues,
            ligand_serials,
            max_radius=float(args.candidate_radius),
        )
        if record is None:
            skipped[f"{row['supervision_split']}:{reason}"] += 1
            continue
        records.append(record)
        split_counts[record["split"]] += 1
    return records, dict(skipped), dict(split_counts)


def empty_extra(record):
    return np.zeros((record["labels"].shape[0], 0), dtype=np.float32), []


def topk_mean(values, k):
    if values.size == 0:
        return 0.0
    if values.size <= k:
        return float(values.mean())
    return float(np.partition(values, -k)[-k:].mean())


def typed_multiscale_features(record, pair_scores, interaction_names, include_spatial=False):
    n_residues = int(record["labels"].shape[0])
    if n_residues == 0:
        return np.zeros((0, 0), dtype=np.float32), []
    residue_for_pair = record["residue_for_pair"].astype(np.int64)
    min_dist = record["pair_distances"][:, 0].astype(np.float64)
    centroid_dist = record["pair_distances"][:, 1].astype(np.float64)
    pair_scores = np.asarray(pair_scores, dtype=np.float64)
    pair_max = pair_scores.max(axis=1) if pair_scores.size else np.zeros(min_dist.shape, dtype=np.float64)
    pair_count = np.bincount(residue_for_pair, minlength=n_residues).astype(np.float64)
    safe_count = np.maximum(pair_count, 1.0)

    columns = []
    names = []

    def add(name, values):
        columns.append(np.asarray(values, dtype=np.float64))
        names.append(name)

    add("ligand_atom_pair_count_log1p", np.log1p(pair_count))
    add(
        "centroid_distance_mean",
        np.bincount(residue_for_pair, weights=centroid_dist, minlength=n_residues) / safe_count,
    )
    for cutoff in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        count = np.bincount(residue_for_pair, weights=(min_dist <= cutoff).astype(float), minlength=n_residues)
        add(f"contact_count_le_{cutoff:g}A", count)
        add(f"contact_frac_le_{cutoff:g}A", count / safe_count)
    for tau in [1.0, 2.0, 4.0, 8.0]:
        weights = np.exp(-min_dist / tau)
        summed = np.bincount(residue_for_pair, weights=weights, minlength=n_residues)
        add(f"distance_soft_count_tau_{tau:g}", summed)
        add(f"distance_soft_mean_tau_{tau:g}", summed / safe_count)

    max_score = np.full(n_residues, -np.inf, dtype=np.float64)
    np.maximum.at(max_score, residue_for_pair, pair_max)
    max_score[~np.isfinite(max_score)] = 0.0
    mean_score = np.bincount(residue_for_pair, weights=pair_max, minlength=n_residues) / safe_count
    add("any_interaction_pair_max", max_score)
    add("any_interaction_pair_mean", mean_score)
    add(
        "any_interaction_weighted_tau_4",
        np.bincount(residue_for_pair, weights=pair_max * np.exp(-min_dist / 4.0), minlength=n_residues)
        / safe_count,
    )
    add(
        "any_interaction_weighted_tau_8",
        np.bincount(residue_for_pair, weights=pair_max * np.exp(-min_dist / 8.0), minlength=n_residues)
        / safe_count,
    )
    for threshold in [0.3, 0.5, 0.7]:
        count = np.bincount(residue_for_pair, weights=(pair_max >= threshold).astype(float), minlength=n_residues)
        add(f"any_interaction_count_ge_{threshold:g}", count)
        add(f"any_interaction_frac_ge_{threshold:g}", count / safe_count)

    top3 = np.zeros(n_residues, dtype=np.float64)
    for residue_index in range(n_residues):
        top3[residue_index] = topk_mean(pair_max[residue_for_pair == residue_index], 3)
    add("any_interaction_top3_mean", top3)

    for class_index, interaction_name in enumerate(interaction_names):
        values = pair_scores[:, class_index]
        typed_max = np.full(n_residues, -np.inf, dtype=np.float64)
        np.maximum.at(typed_max, residue_for_pair, values)
        typed_max[~np.isfinite(typed_max)] = 0.0
        typed_mean = np.bincount(residue_for_pair, weights=values, minlength=n_residues) / safe_count
        typed_weighted = (
            np.bincount(residue_for_pair, weights=values * np.exp(-min_dist / 4.0), minlength=n_residues)
            / safe_count
        )
        add(f"{interaction_name}_max", typed_max)
        add(f"{interaction_name}_mean", typed_mean)
        add(f"{interaction_name}_weighted_tau_4", typed_weighted)

    if include_spatial:
        centroids = np.asarray(record.get("residue_centroids", np.zeros((n_residues, 3))), dtype=np.float64)
        residue_distances = np.asarray(record["distances"], dtype=np.float64)
        if centroids.shape[0] == n_residues and n_residues > 1:
            deltas = centroids[:, None, :] - centroids[None, :, :]
            residue_graph_dist = np.sqrt(np.square(deltas).sum(axis=2))
            np.fill_diagonal(residue_graph_dist, np.inf)
            for radius in [6.0, 8.0, 10.0, 12.0]:
                neighbors = residue_graph_dist <= radius
                counts = neighbors.sum(axis=1).astype(np.float64)
                safe_counts = np.maximum(counts, 1.0)
                add(f"spatial_neighbor_count_le_{radius:g}A", counts)
                neighbor_score_sum = (neighbors * max_score[None, :]).sum(axis=1)
                add(f"spatial_neighbor_any_score_mean_le_{radius:g}A", neighbor_score_sum / safe_counts)
                neighbor_score_max = np.where(neighbors, max_score[None, :], -np.inf).max(axis=1)
                neighbor_score_max[~np.isfinite(neighbor_score_max)] = 0.0
                add(f"spatial_neighbor_any_score_max_le_{radius:g}A", neighbor_score_max)
                neighbor_distance_min = np.where(neighbors, residue_distances[None, :], np.inf).min(axis=1)
                neighbor_distance_min[~np.isfinite(neighbor_distance_min)] = 80.0
                add(f"spatial_neighbor_min_ligand_distance_le_{radius:g}A", neighbor_distance_min)
            for k in [3, 5, 8]:
                kk = min(k, n_residues - 1)
                order = np.argsort(residue_graph_dist, axis=1)[:, :kk]
                add(f"spatial_knn{k}_any_score_mean", max_score[order].mean(axis=1))
                add(f"spatial_knn{k}_any_score_max", max_score[order].max(axis=1))
                add(f"spatial_knn{k}_distance_mean", residue_distances[order].mean(axis=1))
        else:
            for radius in [6.0, 8.0, 10.0, 12.0]:
                add(f"spatial_neighbor_count_le_{radius:g}A", np.zeros(n_residues))
                add(f"spatial_neighbor_any_score_mean_le_{radius:g}A", np.zeros(n_residues))
                add(f"spatial_neighbor_any_score_max_le_{radius:g}A", np.zeros(n_residues))
                add(f"spatial_neighbor_min_ligand_distance_le_{radius:g}A", np.full(n_residues, 80.0))
            for k in [3, 5, 8]:
                add(f"spatial_knn{k}_any_score_mean", np.zeros(n_residues))
                add(f"spatial_knn{k}_any_score_max", np.zeros(n_residues))
                add(f"spatial_knn{k}_distance_mean", np.full(n_residues, 80.0))

    return np.stack(columns, axis=1).astype(np.float32), names


def ensemble_uncertainty_features(record, model_pair_probs):
    model_pair_probs = np.asarray(model_pair_probs, dtype=np.float64)
    n_residues = int(record["labels"].shape[0])
    if n_residues == 0 or model_pair_probs.ndim != 3 or model_pair_probs.shape[0] <= 1:
        return np.zeros((n_residues, 0), dtype=np.float32), []

    residue_for_pair = record["residue_for_pair"].astype(np.int64)
    pair_count = np.bincount(residue_for_pair, minlength=n_residues).astype(np.float64)
    safe_count = np.maximum(pair_count, 1.0)

    per_model_any = model_pair_probs.max(axis=2)
    any_mean = per_model_any.mean(axis=0)
    any_std = per_model_any.std(axis=0)
    any_min = per_model_any.min(axis=0)
    any_range = per_model_any.max(axis=0) - any_min
    channel_std = model_pair_probs.std(axis=0)
    channel_std_mean = channel_std.mean(axis=1)
    channel_std_max = channel_std.max(axis=1)
    channel_consensus = model_pair_probs.min(axis=0).max(axis=1)

    columns = []
    names = []

    def add(name, values):
        columns.append(np.asarray(values, dtype=np.float64))
        names.append(name)

    def add_pair_aggregates(prefix, pair_values):
        pair_values = np.asarray(pair_values, dtype=np.float64)
        max_values = np.full(n_residues, -np.inf, dtype=np.float64)
        np.maximum.at(max_values, residue_for_pair, pair_values)
        max_values[~np.isfinite(max_values)] = 0.0
        add(f"{prefix}_max", max_values)
        add(f"{prefix}_mean", np.bincount(residue_for_pair, weights=pair_values, minlength=n_residues) / safe_count)
        top3 = np.zeros(n_residues, dtype=np.float64)
        for residue_index in range(n_residues):
            top3[residue_index] = topk_mean(pair_values[residue_for_pair == residue_index], 3)
        add(f"{prefix}_top3_mean", top3)

    add_pair_aggregates("ensemble_any_prob_min", any_min)
    add_pair_aggregates("ensemble_any_prob_mean_minus_std", any_mean - any_std)
    add_pair_aggregates("ensemble_any_prob_std", any_std)
    add_pair_aggregates("ensemble_any_prob_range", any_range)
    add_pair_aggregates("ensemble_channel_prob_std_mean", channel_std_mean)
    add_pair_aggregates("ensemble_channel_prob_std_max", channel_std_max)
    add_pair_aggregates("ensemble_channel_consensus", channel_consensus)
    return np.stack(columns, axis=1).astype(np.float32), names


@torch.no_grad()
def score_record_visnet(record, model, class_indices, device, args, interaction_names):
    data = record["data"].clone().to(device)
    models = model if isinstance(model, (list, tuple)) else [model]
    chunks = []
    for item in models:
        logits = item(data)
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Non-finite logits for {record['sample_id']}")
        chunks.append(torch.sigmoid(logits[:, class_indices]).detach().cpu())
    model_pair_probs = torch.stack(chunks, dim=0).numpy()
    pair_probs = model_pair_probs.mean(axis=0)
    probs = pair_probs.max(axis=1)
    scores = np.full(record["labels"].shape, -math.inf, dtype=np.float64)
    np.maximum.at(scores, record["residue_for_pair"], probs.astype(np.float64))
    scores[~np.isfinite(scores)] = 0.0
    if args.stage2_feature_mode in {"typed_multiscale", "typed_context"}:
        extra, names = typed_multiscale_features(
            record,
            pair_probs,
            interaction_names,
            include_spatial=args.stage2_feature_mode == "typed_context",
        )
    else:
        extra, names = empty_extra(record)
    if args.ensemble_uncertainty_features:
        uncertainty_extra, uncertainty_names = ensemble_uncertainty_features(record, model_pair_probs)
        if uncertainty_names:
            extra = np.concatenate([extra, uncertainty_extra], axis=1)
            names = list(names) + uncertainty_names
    return scores, extra, names


def load_pair_mlp(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model"]
    ckpt_args = checkpoint.get("args", {})
    feature_mode = checkpoint.get("feature_mode", ckpt_args.get("feature_mode", "full"))
    identity_dim = 0
    if feature_mode in {"full", "identity_only"}:
        identity_dim = int(state["residue_emb.weight"].shape[1] + state["element_emb.weight"].shape[1])
    first_layer_dim = int(state["net.0.weight"].shape[1])
    dist_dim = first_layer_dim - identity_dim if feature_mode in {"full", "distance_only"} else 0
    model = train_pair_mlp.PairMLP(
        n_residue=int(state.get("residue_emb.weight", torch.empty(len(pair_features.AA3) + 1, 0)).shape[0]),
        n_element=int(state.get("element_emb.weight", torch.empty(len(pair_features.ELEMENTS) + 1, 0)).shape[0]),
        dist_dim=dist_dim,
        hidden=int(ckpt_args.get("hidden", 128)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
        out_dim=int(state["net.8.weight"].shape[0]),
        feature_mode=feature_mode,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint.get("history", []), checkpoint.get("trainable_classes", [])


@torch.no_grad()
def score_record_pair_mlp(record, model, class_indices, device, args, interaction_names, batch_size=65536):
    residue_type = torch.from_numpy(record["pair_residue_type"])
    ligand_element = torch.from_numpy(record["pair_ligand_element"])
    distance_features = torch.from_numpy(train_pair_mlp.distance_features(record["pair_distances"]))
    chunks = []
    for start in range(0, residue_type.shape[0], batch_size):
        stop = min(start + batch_size, residue_type.shape[0])
        logits = model(
            residue_type[start:stop].to(device),
            ligand_element[start:stop].to(device),
            distance_features[start:stop].to(device),
        )
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Non-finite pair-MLP logits for {record['sample_id']}")
        chunks.append(torch.sigmoid(logits[:, class_indices]).cpu())
    pair_probs = torch.cat(chunks).numpy()
    probs = pair_probs.max(axis=1)
    scores = np.full(record["labels"].shape, -math.inf, dtype=np.float64)
    np.maximum.at(scores, record["residue_for_pair"], probs.astype(np.float64))
    scores[~np.isfinite(scores)] = 0.0
    if args.stage2_feature_mode in {"typed_multiscale", "typed_context"}:
        extra, names = typed_multiscale_features(
            record,
            pair_probs,
            interaction_names,
            include_spatial=args.stage2_feature_mode == "typed_context",
        )
    else:
        extra, names = empty_extra(record)
    return scores, extra, names


def apply_visnet_control(visnet_scores, record_index, args):
    if args.visnet_control == "normal":
        return visnet_scores
    if args.visnet_control == "within_graph_shuffle":
        seed = int(args.control_seed or args.seed) + 1000003 * int(record_index)
        rng = np.random.default_rng(seed)
        return rng.permutation(visnet_scores)
    raise ValueError(args.visnet_control)


def assemble_split_arrays(records, model, class_indices, device, args):
    by_split = {
        split: {
            "labels": [],
            "distances": [],
            "visnet": [],
            "extra_features": [],
            "extra_feature_names": None,
            "sample_index": [],
            "sample_ids": [],
        }
        for split in SPLIT_ORDER
    }
    score_interactions = args.score_interactions or args.positive_interactions
    for record_index, record in enumerate(records):
        if args.interaction_backbone == "visnet":
            visnet_scores, extra_features, extra_names = score_record_visnet(
                record, model, class_indices, device, args, score_interactions
            )
        else:
            visnet_scores, extra_features, extra_names = score_record_pair_mlp(
                record, model, class_indices, device, args, score_interactions
            )
        visnet_scores = apply_visnet_control(visnet_scores, record_index, args)
        split = record["split"]
        if by_split[split]["extra_feature_names"] is None:
            by_split[split]["extra_feature_names"] = list(extra_names)
        elif by_split[split]["extra_feature_names"] != list(extra_names):
            raise RuntimeError("Inconsistent typed/multiscale feature names across records")
        sample_index = len(by_split[split]["sample_ids"])
        by_split[split]["sample_ids"].append(record["sample_id"])
        by_split[split]["labels"].append(record["labels"])
        by_split[split]["distances"].append(record["distances"])
        by_split[split]["visnet"].append(visnet_scores)
        by_split[split]["extra_features"].append(extra_features)
        by_split[split]["sample_index"].append(np.full(record["labels"].shape, sample_index, dtype=np.int64))

    out = {}
    for split, values in by_split.items():
        if values["labels"]:
            out[split] = {
                "labels": np.concatenate(values["labels"]).astype(np.int32),
                "distances": np.concatenate(values["distances"]).astype(np.float64),
                "visnet": np.concatenate(values["visnet"]).astype(np.float64),
                "extra_features": np.concatenate(values["extra_features"], axis=0).astype(np.float32),
                "extra_feature_names": values["extra_feature_names"] or [],
                "sample_index": np.concatenate(values["sample_index"]).astype(np.int64),
                "sample_ids": values["sample_ids"],
            }
        else:
            out[split] = {
                "labels": np.asarray([], dtype=np.int32),
                "distances": np.asarray([], dtype=np.float64),
                "visnet": np.asarray([], dtype=np.float64),
                "extra_features": np.zeros((0, 0), dtype=np.float32),
                "extra_feature_names": values["extra_feature_names"] or [],
                "sample_index": np.asarray([], dtype=np.int64),
                "sample_ids": [],
            }
    return out


def graph_rank(values, sample_index):
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros(values.shape, dtype=np.float64)
    for idx in np.unique(sample_index):
        mask = sample_index == idx
        v = values[mask]
        if v.size <= 1:
            out[mask] = 0.5
            continue
        order = np.argsort(v)
        ranks = np.empty(v.size, dtype=np.float64)
        ranks[order] = np.arange(v.size, dtype=np.float64) / float(v.size - 1)
        out[mask] = ranks
    return out


def graph_zscore(values, sample_index):
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros(values.shape, dtype=np.float64)
    for idx in np.unique(sample_index):
        mask = sample_index == idx
        v = values[mask]
        std = float(v.std())
        if std <= 1e-12:
            continue
        out[mask] = (v - float(v.mean())) / std
    return out


def feature_matrix(split_data, variant, stage2_feature_mode="base"):
    distances = split_data["distances"].astype(np.float64)
    visnet = np.clip(split_data["visnet"].astype(np.float64), 0.0, 1.0)
    priors = [np.exp(-distances / tau) for tau in TAUS]
    columns = []
    names = []

    if variant in {"distance", "stage2"}:
        clipped = np.clip(distances, 0.0, 80.0)
        columns.extend([clipped, np.log1p(clipped)])
        names.extend(["distance", "log1p_distance"])
        for tau, prior in zip(TAUS, priors):
            columns.append(prior)
            names.append(f"range_tau_{tau:g}")

    if variant in {"visnet", "stage2"}:
        eps = 1e-5
        logit = np.log(np.clip(visnet, eps, 1 - eps) / np.clip(1 - visnet, eps, 1 - eps))
        columns.extend([visnet, logit])
        names.extend(["visnet", "visnet_logit"])

    if variant == "stage2":
        for tau, prior in zip(TAUS, priors):
            columns.append(visnet * prior)
            names.append(f"visnet_x_range_tau_{tau:g}")
        columns.append(visnet / (1.0 + distances))
        names.append("visnet_over_1_plus_distance")
        if stage2_feature_mode in {"typed_multiscale", "typed_context"}:
            extra = split_data.get("extra_features")
            extra_names = split_data.get("extra_feature_names", [])
            if extra is not None and extra.shape[1] != len(extra_names):
                raise RuntimeError("Typed/multiscale feature matrix does not match feature names")
            if extra is not None:
                for index, name in enumerate(extra_names):
                    columns.append(extra[:, index].astype(np.float64))
                    names.append(f"typed_multiscale__{name}")
        if stage2_feature_mode == "typed_context":
            sample_index = split_data["sample_index"]
            context_sources = {
                "near_distance": -distances,
                "visnet": visnet,
            }
            extra = split_data.get("extra_features")
            extra_names = split_data.get("extra_feature_names", [])
            keep_extra = {
                "any_interaction_pair_max",
                "any_interaction_pair_mean",
                "any_interaction_top3_mean",
                "contact_count_le_4A",
                "contact_count_le_5A",
                "contact_frac_le_5A",
                "distance_soft_count_tau_4",
                "distance_soft_mean_tau_4",
                "spatial_neighbor_any_score_max_le_8A",
                "spatial_neighbor_any_score_mean_le_8A",
                "spatial_knn5_any_score_mean",
                "spatial_knn5_distance_mean",
            }
            if extra is not None:
                for index, extra_name in enumerate(extra_names):
                    if extra_name in keep_extra:
                        context_sources[extra_name] = extra[:, index]
            for source_name, values in context_sources.items():
                values = np.asarray(values, dtype=np.float64)
                columns.append(graph_rank(values, sample_index))
                names.append(f"context_rank__{source_name}")
                columns.append(graph_zscore(values, sample_index))
                names.append(f"context_zscore__{source_name}")

    if not columns:
        raise ValueError(variant)
    return np.stack(columns, axis=1).astype(np.float32), names


def summarize_scores(split_data, scores, calibrated_scores=None):
    labels = split_data["labels"]
    sample_index = split_data["sample_index"]
    graph_auprc = []
    graph_auroc = []
    graph_positive_rate = []
    hit_at_1 = []
    precision_at_5 = []
    recall_at_5 = []
    precision_at_positives = []
    recall_at_positives = []
    enrichment_at_positives = []
    for idx in np.unique(sample_index):
        mask = sample_index == idx
        y = labels[mask]
        s = scores[mask]
        graph_positive_rate.append(float(y.mean()) if y.size else math.nan)
        positives = int(y.sum())
        if y.size and positives > 0:
            order = np.argsort(-s)
            hit_at_1.append(float(y[order[0]]))
            top5 = order[: min(5, y.size)]
            precision_at_5.append(float(y[top5].mean()))
            recall_at_5.append(float(y[top5].sum() / positives))
            top_l = order[:positives]
            precision_l = float(y[top_l].mean())
            recall_l = float(y[top_l].sum() / positives)
            precision_at_positives.append(precision_l)
            recall_at_positives.append(recall_l)
            baseline = float(y.mean())
            enrichment_at_positives.append(precision_l / baseline if baseline > 0 else math.nan)
        if y.size == 0 or y.min() == y.max():
            continue
        graph_auprc.append(float(average_precision_score(y, s)))
        graph_auroc.append(float(roc_auc_score(y, s)))
    summary = {
        "graphs": int(len(split_data["sample_ids"])),
        "residues": int(labels.size),
        "positives": int(labels.sum()) if labels.size else 0,
        "prevalence": float(labels.mean()) if labels.size else math.nan,
        "auprc": safe_metric(average_precision_score, labels, scores),
        "auroc": safe_metric(roc_auc_score, labels, scores),
        "mean_graph_auprc": describe(graph_auprc),
        "mean_graph_auroc": describe(graph_auroc),
        "mean_graph_positive_rate": describe(graph_positive_rate),
        "public_rank_metrics": {
            "hit_at_1": describe(hit_at_1),
            "precision_at_5": describe(precision_at_5),
            "recall_at_5": describe(recall_at_5),
            "precision_at_num_positives": describe(precision_at_positives),
            "recall_at_num_positives": describe(recall_at_positives),
            "enrichment_at_num_positives": describe(enrichment_at_positives),
        },
    }
    calibration = {}
    if bounded_probability(scores):
        calibration["raw_probability"] = probability_calibration(labels, scores)
    if calibrated_scores is not None:
        calibration["platt_internal_val"] = probability_calibration(labels, calibrated_scores)
    if calibration:
        summary["calibration"] = calibration
    return summary


def raw_scores(split_data, method):
    if method == "raw_distance":
        return -split_data["distances"]
    if method == "raw_visnet":
        return split_data["visnet"]
    if method == "fixed_hybrid":
        return 0.55 * split_data["visnet"] + 0.45 * np.exp(-split_data["distances"] / 12.0)
    raise ValueError(method)


def fit_variant(variant, data_by_split, args):
    x_train, feature_names = feature_matrix(data_by_split["train"], variant, args.stage2_feature_mode)
    y_train = data_by_split["train"]["labels"]
    x_val, _ = feature_matrix(data_by_split["internal_val"], variant, args.stage2_feature_mode)
    y_val = data_by_split["internal_val"]["labels"]
    if y_train.size == 0 or y_train.min() == y_train.max():
        raise RuntimeError(f"Training split for {variant} does not contain both classes")
    best = None
    trace = []
    if args.stage2_classifier == "logreg":
        candidates = [{"classifier": "logreg", "C": c_value} for c_value in sorted(set(float(value) for value in args.c_grid))]
    else:
        candidates = [
            {"classifier": "hgbt", "l2_regularization": l2_value, "max_leaf_nodes": leaf_nodes}
            for l2_value in sorted(set(float(value) for value in args.hgbt_l2_grid))
            for leaf_nodes in sorted(set(int(value) for value in args.hgbt_leaf_grid))
        ]
    for candidate in candidates:
        if candidate["classifier"] == "logreg":
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=candidate["C"],
                    class_weight="balanced",
                    max_iter=args.max_iter,
                    solver="lbfgs",
                    random_state=args.seed,
                ),
            )
        else:
            model = HistGradientBoostingClassifier(
                loss="log_loss",
                learning_rate=float(args.hgbt_learning_rate),
                max_iter=int(args.hgbt_max_iter),
                max_leaf_nodes=int(candidate["max_leaf_nodes"]),
                l2_regularization=float(candidate["l2_regularization"]),
                class_weight="balanced",
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=args.seed,
            )
        model.fit(x_train, y_train)
        val_scores = model.predict_proba(x_val)[:, 1]
        row = {
            **candidate,
            "internal_val_auprc": safe_metric(average_precision_score, y_val, val_scores),
            "internal_val_auroc": safe_metric(roc_auc_score, y_val, val_scores),
        }
        trace.append(row)
        score = row["internal_val_auprc"]
        if math.isnan(score):
            continue
        if best is None or score > best["internal_val_auprc"] + 1e-12:
            best = {"model": model, **row}
        elif best is not None and abs(score - best["internal_val_auprc"]) <= 1e-12:
            if json.dumps(candidate, sort_keys=True) < json.dumps({k: best[k] for k in candidate}, sort_keys=True):
                best = {"model": model, **row}
    if best is None:
        raise RuntimeError(f"No valid validation metric for {variant}")
    return best["model"], {k: v for k, v in best.items() if k != "model"}, trace, feature_names


def evaluate_all(data_by_split, fitted, args):
    score_cache = {}
    calibrated_cache = {}
    methods = {}
    for method in ["raw_distance", "raw_visnet", "fixed_hybrid"]:
        score_cache[method] = {split: raw_scores(split_data, method) for split, split_data in data_by_split.items()}
        calibrator = fit_score_calibrator(
            data_by_split["internal_val"]["labels"],
            score_cache[method]["internal_val"],
            args.seed,
        )
        calibrated_cache[method] = {}
        if calibrator is not None:
            for split, scores in score_cache[method].items():
                calibrated_cache[method][split] = (
                    calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
                    if scores.size
                    else np.empty(0, dtype=np.float64)
                )
        methods[method] = {
            split: summarize_scores(split_data, score_cache[method][split], calibrated_cache[method].get(split))
            for split, split_data in data_by_split.items()
        }
    for variant, item in fitted.items():
        model = item["model"]
        method_name = f"trained_{variant}"
        score_cache[method_name] = {}
        for split, split_data in data_by_split.items():
            x, _ = feature_matrix(split_data, variant, args.stage2_feature_mode)
            score_cache[method_name][split] = (
                model.predict_proba(x)[:, 1]
                if x.shape[0]
                else np.empty(0, dtype=np.float64)
            )
        calibrator = fit_score_calibrator(
            data_by_split["internal_val"]["labels"],
            score_cache[method_name]["internal_val"],
            args.seed,
        )
        calibrated_cache[method_name] = {}
        if calibrator is not None:
            for split, scores in score_cache[method_name].items():
                calibrated_cache[method_name][split] = (
                    calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
                    if scores.size
                    else np.empty(0, dtype=np.float64)
                )
        methods[method_name] = {}
        for split, split_data in data_by_split.items():
            scores = score_cache[method_name][split]
            methods[method_name][split] = summarize_scores(split_data, scores, calibrated_cache[method_name].get(split))
    return methods, score_cache, calibrated_cache


def write_prediction_archive(path, data_by_split, score_cache, calibrated_cache):
    arrays = {}
    for split in SPLIT_ORDER:
        split_data = data_by_split[split]
        prefix = f"{split}__"
        arrays[f"{prefix}labels"] = np.asarray(split_data["labels"], dtype=np.int8)
        arrays[f"{prefix}sample_index"] = np.asarray(split_data["sample_index"], dtype=np.int32)
        arrays[f"{prefix}sample_ids"] = np.asarray(split_data["sample_ids"], dtype=str)
        for method, split_scores in score_cache.items():
            arrays[f"{prefix}score__{method}"] = np.asarray(split_scores[split], dtype=np.float64)
        for method, split_scores in calibrated_cache.items():
            if split in split_scores:
                arrays[f"{prefix}calibrated__{method}"] = np.asarray(split_scores[split], dtype=np.float64)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return output


def fmt(value):
    return "nan" if math.isnan(float(value)) else f"{float(value):.4f}"


def fmt_optional(value):
    if value is None:
        return "NA"
    try:
        return fmt(value)
    except (TypeError, ValueError):
        return "NA"


def write_table(metrics, path):
    lines = [
        "# Base BioLiP-Trained Stage-2 Residue Localizer",
        "",
        f"- checkpoint: `{metrics['checkpoint']}`",
        f"- interaction backbone: `{metrics['interaction_backbone']}`",
        f"- label mode: `{metrics['label_mode']}`",
        f"- score interactions: `{', '.join(metrics['score_interactions'])}`",
        f"- Stage-2 feature mode: `{metrics['stage2_feature_mode']}`",
        f"- seed: `{metrics['seed']}`",
        f"- ViSNet control: `{metrics['visnet_control']}`",
        f"- train graphs: `{metrics['split_counts'].get('train', 0)}`",
        f"- internal_val graphs: `{metrics['split_counts'].get('internal_val', 0)}`",
        "",
        "| method | split | graphs | residues | positives | prevalence | residue AUPRC | residue AUROC | mean graph AUPRC | mean graph AUROC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    method_order = [
        "raw_distance",
        "raw_visnet",
        "fixed_hybrid",
        "trained_distance",
        "trained_visnet",
        "trained_stage2",
    ]
    for method in method_order:
        for split in SPLIT_ORDER:
            row = metrics["methods"][method][split]
            lines.append(
                f"| {method} | {split} | {row['graphs']} | {row['residues']} | {row['positives']} | "
                f"{fmt(row['prevalence'])} | {fmt(row['auprc'])} | {fmt(row['auroc'])} | "
                f"{fmt(row['mean_graph_auprc']['mean'])} | {fmt(row['mean_graph_auroc']['mean'])} |"
            )
    lines.extend(["", "## Selected Regularization", "", "| variant | C | internal_val AUPRC | internal_val AUROC | features |", "| --- | ---: | ---: | ---: | ---: |"])
    for variant in ["distance", "visnet", "stage2"]:
        selected = metrics["trained_variants"][variant]["selected"]
        feature_count = len(metrics["trained_variants"][variant]["feature_names"])
        selector = selected.get("C", f"leaf={selected.get('max_leaf_nodes')},l2={selected.get('l2_regularization')}")
        lines.append(
            f"| {variant} | {selector} | {fmt(selected['internal_val_auprc'])} | "
            f"{fmt(selected['internal_val_auroc'])} | {feature_count} |"
        )
    lines.extend(
        [
            "",
            "## Probability Calibration",
            "",
            "| method | split | raw Brier | raw ECE-15 | raw NLL | Platt Brier | Platt ECE-15 | Platt NLL | Platt risk-coverage AUC |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in method_order:
        for split in SPLIT_ORDER:
            row = metrics["methods"][method][split]
            calibration = row.get("calibration", {})
            raw = calibration.get("raw_probability") or {}
            platt = calibration.get("platt_internal_val") or {}
            lines.append(
                f"| {method} | {split} | "
                f"{fmt_optional(raw.get('brier'))} | {fmt_optional(raw.get('ece_15'))} | {fmt_optional(raw.get('nll'))} | "
                f"{fmt_optional(platt.get('brier'))} | {fmt_optional(platt.get('ece_15'))} | "
                f"{fmt_optional(platt.get('nll'))} | {fmt_optional(platt.get('risk_coverage_auc'))} |"
            )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.ensemble_uncertainty_features:
        if args.interaction_backbone != "visnet" or not args.ensemble_checkpoints:
            raise RuntimeError("--ensemble-uncertainty-features requires ViSNet with --ensemble-checkpoints")
        if args.stage2_feature_mode == "base":
            raise RuntimeError("--ensemble-uncertainty-features requires typed_multiscale or typed_context features")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    score_interactions = args.score_interactions or args.positive_interactions
    class_indices = contact_eval.active_indices(score_interactions)
    if args.interaction_backbone == "visnet":
        score_args = SimpleNamespace(
            finite_representation_guard=args.finite_representation_guard,
            representation_clip=args.representation_clip,
        )
        checkpoints = [args.checkpoint] + list(args.ensemble_checkpoints or [])
        loaded = [contact_eval.load_model(path, score_args, device) for path in checkpoints]
        model = [item[0] for item in loaded] if len(loaded) > 1 else loaded[0][0]
        history = [{"checkpoint": path, "history": item[1]} for path, item in zip(checkpoints, loaded)]
    else:
        if args.ensemble_checkpoints:
            raise RuntimeError("--ensemble-checkpoints is only supported for the ViSNet interaction backbone")
        model, history, checkpoint_classes = load_pair_mlp(args.checkpoint, device)
        if checkpoint_classes and not set(class_indices).issubset(set(checkpoint_classes)):
            raise RuntimeError("Pair-MLP checkpoint does not contain all requested interaction classes")
    records, skipped, split_counts = load_candidate_records(args)
    data_by_split = assemble_split_arrays(records, model, class_indices, device, args)

    fitted = {}
    trained_variants = {}
    for variant in ["distance", "visnet", "stage2"]:
        clf, selected, trace, feature_names = fit_variant(variant, data_by_split, args)
        fitted[variant] = {"model": clf}
        trained_variants[variant] = {
            "selected": selected,
            "trace": trace,
            "feature_names": feature_names,
        }

    method_metrics, score_cache, calibrated_cache = evaluate_all(data_by_split, fitted, args)
    prediction_archive = None
    if args.predictions_output:
        prediction_archive = write_prediction_archive(
            args.predictions_output,
            data_by_split,
            score_cache,
            calibrated_cache,
        )
    metrics = {
        "checkpoint": args.checkpoint,
        "ensemble_checkpoints": args.ensemble_checkpoints or [],
        "ensemble_uncertainty_features": bool(args.ensemble_uncertainty_features),
        "interaction_backbone": args.interaction_backbone,
        "label_mode": args.label_mode,
        "seed": args.seed,
        "candidate_radius": args.candidate_radius,
        "positive_interactions": args.positive_interactions,
        "score_interactions": score_interactions,
        "stage2_feature_mode": args.stage2_feature_mode,
        "stage2_classifier": args.stage2_classifier,
        "visnet_control": args.visnet_control,
        "control_seed": int(args.control_seed or args.seed),
        "limits": {
            "max_train_graphs": args.max_train_graphs,
            "max_internal_val_graphs": args.max_internal_val_graphs,
            "max_external_val_graphs": args.max_external_val_graphs,
            "max_external_test_graphs": args.max_external_test_graphs,
        },
        "split_counts": split_counts,
        "skipped": skipped,
        "device": str(device),
        "finite_representation_guard": bool(args.finite_representation_guard),
        "representation_clip": float(args.representation_clip),
        "history": history,
        "trained_variants": trained_variants,
        "methods": method_metrics,
        "artifacts": {
            "prediction_archive": str(prediction_archive) if prediction_archive is not None else None,
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_table(metrics, args.table)
    print(
        json.dumps(
            {
                "metrics": str(metrics_path),
                "table": args.table,
                "predictions": str(prediction_archive) if prediction_archive is not None else None,
            }
        )
    )


if __name__ == "__main__":
    main()
