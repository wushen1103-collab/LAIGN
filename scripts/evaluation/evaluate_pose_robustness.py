#!/usr/bin/env python3
"""Evaluate Post-freeze pose robustness over repeated rigid-pose perturbations."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation import pose_perturbation_core as robustness  # noqa: E402
from scripts.model import train_residue_localizer as stage2  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "outputs/postfreeze_holdout"
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "checkpoints/interaction_graph_visnet_hardneg_full_raw10_seed2401/best.pt"),
    )
    parser.add_argument(
        "--ensemble-checkpoints",
        nargs=2,
        default=[
            str(ROOT / "checkpoints/interaction_graph_visnet_hardneg_full_raw10_seed2402/best.pt"),
            str(ROOT / "checkpoints/interaction_graph_visnet_hardneg_full_raw10_seed2403/best.pt"),
        ],
    )
    parser.add_argument("--max-train-graphs", type=int, default=5000)
    parser.add_argument("--max-internal-val-graphs", type=int, default=1000)
    parser.add_argument("--max-external-test-graphs", type=int, default=70)
    parser.add_argument("--localizer-seed", type=int, default=3401)
    parser.add_argument(
        "--c-grid",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0],
        help="Stage-2 regularization values; pass the frozen replicate value to avoid reselection.",
    )
    parser.add_argument("--perturb-seeds", type=int, nargs="+", default=[8101, 8102, 8103, 8104, 8105])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scope", default="full")
    parser.add_argument("--out-json", default=str(base / "pose_robustness_summary.json"))
    parser.add_argument("--rows", default=str(base / "pose_robustness_rows.csv"))
    parser.add_argument(
        "--table", default=str(ROOT / "outputs/tables/table_postfreeze_pose_robustness.md")
    )
    return parser.parse_args()


def stage2_namespace(args: argparse.Namespace) -> SimpleNamespace:
    base = ROOT / "outputs/postfreeze_holdout"
    return SimpleNamespace(
        split_manifest=str(base / "postfreeze_split_manifest.csv.gz"),
        biolip_interactions=str(
            ROOT / "data/processed/biolip2/biolip2_nr_plip_interaction_labels.csv.gz"
        ),
        plinder_interactions=str(base / "postfreeze_plip_interactions.csv.gz"),
        biolip_site_labels=str(
            ROOT / "data/processed/biolip2/biolip2_nr_plip_site_labels.csv.gz"
        ),
        plinder_site_labels=str(base / "postfreeze_site_labels.csv.gz"),
        label_mode="active_interaction",
        checkpoint=args.checkpoint,
        ensemble_checkpoints=args.ensemble_checkpoints,
        ensemble_uncertainty_features=False,
        interaction_backbone="visnet",
        seed=args.localizer_seed,
        max_train_graphs=args.max_train_graphs,
        max_internal_val_graphs=args.max_internal_val_graphs,
        max_external_val_graphs=0,
        max_external_test_graphs=args.max_external_test_graphs,
        candidate_radius=999.0,
        positive_interactions=stage2.DEFAULT_ACTIVE_INTERACTIONS,
        score_interactions=stage2.DEFAULT_ACTIVE_INTERACTIONS,
        stage2_feature_mode="typed_multiscale",
        c_grid=args.c_grid,
        stage2_classifier="logreg",
        hgbt_l2_grid=[0.0],
        hgbt_leaf_grid=[15],
        hgbt_max_iter=120,
        hgbt_learning_rate=0.06,
        max_iter=1000,
        visnet_control="normal",
        control_seed=args.localizer_seed,
        device=args.device,
        finite_representation_guard=True,
        representation_clip=0.0,
    )


def axis_angle_matrix(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    x, y, z = axis / max(float(np.linalg.norm(axis)), 1e-8)
    angle = math.radians(angle_degrees)
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


def controlled_transformed_record(
    record: dict[str, object],
    translation_angstrom: float,
    rotation_degrees: float,
    seed: int,
) -> dict[str, object]:
    out = dict(record)
    out["data"] = record["data"].clone()
    out["labels"] = np.asarray(record["labels"]).copy()
    out["residue_for_pair"] = np.asarray(record["residue_for_pair"]).copy()
    out["pair_residue_type"] = np.asarray(record["pair_residue_type"]).copy()
    out["pair_ligand_element"] = np.asarray(record["pair_ligand_element"]).copy()
    out["residue_centroids"] = np.asarray(record["residue_centroids"]).copy()

    data = out["data"]
    pos = data.pos.detach().cpu().numpy().astype(np.float32)
    ligand_nodes = torch.unique(data.pair_index[1]).detach().cpu().numpy().astype(np.int64)
    ligand_pos = pos[ligand_nodes]
    centroid = ligand_pos.mean(axis=0)
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3).astype(np.float32)
    axis /= max(float(np.linalg.norm(axis)), 1e-8)
    direction = rng.normal(size=3).astype(np.float32)
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    rotated = (ligand_pos - centroid) @ axis_angle_matrix(axis, rotation_degrees).T + centroid
    pos[ligand_nodes] = rotated + direction * translation_angstrom

    data.pos = torch.tensor(pos, dtype=torch.float32)
    pair_index = data.pair_index.detach().cpu().numpy().astype(np.int64)
    pair_dist = np.sqrt(
        np.square(pos[pair_index[0]] - pos[pair_index[1]]).sum(axis=1)
    ).astype(np.float32)
    out["pair_distances"] = np.stack([pair_dist, pair_dist], axis=1).astype(np.float32)
    n_residues = int(out["labels"].shape[0])
    residue_distance = np.full(n_residues, np.inf, dtype=np.float32)
    np.minimum.at(residue_distance, out["residue_for_pair"], pair_dist)
    residue_distance[~np.isfinite(residue_distance)] = 80.0
    out["distances"] = residue_distance.astype(np.float64)
    return out


def mean_recall_at_k(
    split_data: dict[str, object], scores: np.ndarray, k: int
) -> float:
    labels = np.asarray(split_data["labels"], dtype=np.int32)
    sample_index = np.asarray(split_data["sample_index"], dtype=np.int32)
    values = []
    for index in np.unique(sample_index):
        mask = sample_index == index
        y = labels[mask]
        positives = int(y.sum())
        if positives <= 0:
            continue
        order = np.argsort(-scores[mask], kind="stable")
        values.append(float(y[order[: min(k, y.size)]].sum() / positives))
    return float(np.mean(values)) if values else float("nan")


def compact_metrics(
    summary: dict[str, object],
    split_data: dict[str, object],
    scores: np.ndarray,
) -> dict[str, float]:
    public = summary["public_rank_metrics"]
    return {
        "pooled_auprc": float(summary["auprc"]),
        "graph_ap": float(summary["mean_graph_auprc"]["mean"]),
        "recall_at_5": float(public["recall_at_5"]["mean"]),
        "recall_at_10": mean_recall_at_k(split_data, scores, 10),
        "precision_at_p": float(public["precision_at_num_positives"]["mean"]),
    }


def summarize_mode(split_data: dict[str, object], classifier, args2: SimpleNamespace) -> dict[str, dict[str, float]]:
    x, _ = stage2.feature_matrix(split_data, "stage2", args2.stage2_feature_mode)
    method_scores = {
        "stage2": classifier.predict_proba(x)[:, 1],
        "raw_distance": stage2.raw_scores(split_data, "raw_distance"),
        "fixed_hybrid": stage2.raw_scores(split_data, "fixed_hybrid"),
    }
    return {
        method: compact_metrics(
            stage2.summarize_scores(split_data, scores), split_data, scores
        )
        for method, scores in method_scores.items()
    }


def main() -> int:
    args = parse_args()
    args2 = stage2_namespace(args)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    started = time.time()
    records, skipped, split_counts = stage2.load_candidate_records(args2)
    models, class_indices, histories = robustness.load_models(args2, device)
    native_arrays = stage2.assemble_split_arrays(records, models, class_indices, device, args2)
    classifier, best, _, feature_names = stage2.fit_variant("stage2", native_arrays, args2)

    rows = []
    native = summarize_mode(native_arrays["external_test"], classifier, args2)
    for method, metrics in native.items():
        rows.append({"perturb_seed": -1, "mode": "native", "method": method, **metrics})

    external_records = [record for record in records if record["split"] == "external_test"]
    mode_specs = [
        ("translation_0.5A", 0.5, 0.0),
        ("translation_1A", 1.0, 0.0),
        ("translation_2A", 2.0, 0.0),
        ("rotation_5deg", 0.0, 5.0),
        ("rotation_10deg", 0.0, 10.0),
        ("rotation_20deg", 0.0, 20.0),
        ("combined_0.5A_5deg", 0.5, 5.0),
        ("combined_1A_10deg", 1.0, 10.0),
        ("combined_2A_20deg", 2.0, 20.0),
        ("wrong_pocket_centroid", -1.0, -1.0),
    ]
    for perturb_seed in args.perturb_seeds:
        for mode, translation, rotation in mode_specs:
            transformed = []
            for record_index, record in enumerate(external_records):
                record_seed = perturb_seed + record_index * 1009
                if mode == "wrong_pocket_centroid":
                    transformed.append(
                        robustness.transformed_record(
                            record, mode, record_seed + 9000001
                        )
                    )
                else:
                    transformed.append(
                        controlled_transformed_record(
                            record, translation, rotation, record_seed
                        )
                    )
            arrays = stage2.assemble_split_arrays(
                transformed, models, class_indices, device, args2
            )
            result = summarize_mode(arrays["external_test"], classifier, args2)
            for method, metrics in result.items():
                rows.append(
                    {
                        "perturb_seed": perturb_seed,
                        "mode": mode,
                        "method": method,
                        **metrics,
                    }
                )

    frame = pd.DataFrame(rows)
    rows_path = Path(args.rows)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(rows_path, index=False)
    aggregate = []
    for (mode, method), group in frame.groupby(["mode", "method"], sort=False):
        entry: dict[str, object] = {
            "mode": mode,
            "method": method,
            "runs": int(len(group)),
        }
        for metric in ("pooled_auprc", "graph_ap", "recall_at_5", "recall_at_10", "precision_at_p"):
            values = group[metric].to_numpy(dtype=np.float64)
            entry[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            }
        aggregate.append(entry)

    payload = {
        "hypothesis": "Pose robustness",
        "scope": args.scope,
        "protocol": (
            "frozen Post-freeze primary ensemble A; paired translation directions and rotation axes "
            "within each complex and perturbation seed"
        ),
        "limits": {
            "train": args.max_train_graphs,
            "internal_val": args.max_internal_val_graphs,
            "external_test": args.max_external_test_graphs,
        },
        "perturb_seeds": args.perturb_seeds,
        "c_grid": args.c_grid,
        "pose_modes": [mode for mode, _, _ in mode_specs],
        "split_counts": split_counts,
        "skipped": skipped,
        "classifier_selection": best,
        "feature_count": len(feature_names),
        "device": str(device),
        "histories": histories,
        "aggregate": aggregate,
        "elapsed_seconds": time.time() - started,
        "rows": str(rows_path.resolve().relative_to(ROOT.resolve())),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        f"# Pose robustness Post-freeze Pose Robustness ({args.scope})",
        "",
        "| pose | method | runs | graph AP | recall@5 | recall@10 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['mode']} | {row['method']} | {row['runs']} | "
            f"{row['graph_ap']['mean']:.4f} +/- {row['graph_ap']['std']:.4f} | "
            f"{row['recall_at_5']['mean']:.4f} +/- {row['recall_at_5']['std']:.4f} | "
            f"{row['recall_at_10']['mean']:.4f} +/- {row['recall_at_10']['std']:.4f} |"
        )
    table = Path(args.table)
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(out_json), "table": str(table), "rows": len(frame)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
