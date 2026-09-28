#!/usr/bin/env python3
"""Benchmark frozen Post-freeze inference and report resource use transparently."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation import pose_perturbation_core as robustness  # noqa: E402
from scripts.model import train_residue_localizer as stage2  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "outputs/postfreeze_holdout"
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-json", default=str(base / "efficiency_summary.json"))
    parser.add_argument(
        "--table", default=str(ROOT / "outputs/tables/table_postfreeze_efficiency.md")
    )
    return parser.parse_args()


def namespace(device: str) -> SimpleNamespace:
    base = ROOT / "outputs/postfreeze_holdout"
    checkpoint_root = ROOT / "checkpoints/interaction_graph_visnet_hardneg_full_raw10_seed"
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
        checkpoint=str(Path(f"{checkpoint_root}2401") / "best.pt"),
        ensemble_checkpoints=[
            str(Path(f"{checkpoint_root}2402") / "best.pt"),
            str(Path(f"{checkpoint_root}2403") / "best.pt"),
        ],
        ensemble_uncertainty_features=False,
        interaction_backbone="visnet",
        seed=3401,
        max_train_graphs=1,
        max_internal_val_graphs=1,
        max_external_val_graphs=0,
        max_external_test_graphs=70,
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
        control_seed=3401,
        device=device,
        finite_representation_guard=True,
        representation_clip=0.0,
    )


def baseline_runtime(path: Path, method: str) -> dict[str, float | int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    runtime = payload["diagnostics"][method]["runtime_seconds"]
    return {
        "graphs": int(payload["diagnostics"][method]["successful_graphs"]),
        "total_seconds": float(runtime["sum"]),
        "mean_seconds_per_graph": float(runtime["mean"]),
        "max_seconds_per_graph": float(runtime["max"]),
    }


def main() -> int:
    args = parse_args()
    args2 = namespace(args.device)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    records, skipped, split_counts = stage2.load_candidate_records(args2)
    models, class_indices, _ = robustness.load_models(args2, device)
    external = [record for record in records if record["split"] == "external_test"]
    if len(external) != 70:
        raise ValueError(f"Expected 70 Post-freeze graphs, found {len(external)}")

    parameter_counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
    stage2_features = 0
    timings = []
    peak_allocated = []
    peak_incremental = []
    for repeat in range(args.repeats + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            baseline_memory = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            baseline_memory = 0
        started = time.perf_counter()
        arrays = stage2.assemble_split_arrays(external, models, class_indices, device, args2)
        features, names = stage2.feature_matrix(
            arrays["external_test"], "stage2", args2.stage2_feature_mode
        )
        _ = np.asarray(features, dtype=np.float64).sum(axis=1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        stage2_features = len(names)
        if repeat == 0:
            continue
        timings.append(elapsed)
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device)
            peak_allocated.append(peak / (1024**2))
            peak_incremental.append(max(0, peak - baseline_memory) / (1024**2))

    base = ROOT / "outputs/postfreeze_holdout"
    payload = {
        "hypothesis": "Validation",
        "scope": "frozen Post-freeze model inference on cached parsed complex graphs",
        "graphs": len(external),
        "repeats_after_warmup": args.repeats,
        "device": str(device),
        "split_counts": split_counts,
        "skipped": skipped,
        "LAIGN": {
            "total_seconds": {
                "mean": float(np.mean(timings)),
                "std": float(np.std(timings, ddof=1)) if len(timings) > 1 else 0.0,
            },
            "seconds_per_graph": {
                "mean": float(np.mean(timings) / len(external)),
                "std": float(np.std(timings, ddof=1) / len(external)) if len(timings) > 1 else 0.0,
            },
            "single_scorer_parameters": parameter_counts[0],
            "ensemble_parameters": int(sum(parameter_counts)),
            "stage2_feature_count": stage2_features,
            "stage2_linear_parameters": stage2_features + 1,
            "peak_cuda_allocated_MiB": float(max(peak_allocated)) if peak_allocated else 0.0,
            "peak_incremental_cuda_MiB": float(max(peak_incremental)) if peak_incremental else 0.0,
        },
        "external_tools": {
            "fpocket": baseline_runtime(base / "baselines_geometry_full/summary.json", "fpocket"),
            "P2Rank": baseline_runtime(base / "baselines_geometry_full/summary.json", "p2rank"),
            "ProLIF": baseline_runtime(base / "baseline_prolif_ccd_full/summary.json", "prolif"),
        },
        "comparability_note": (
            "LAIGN timing starts from cached parsed graphs and includes the three frozen scorers and "
            "typed-multiscale feature assembly. External-tool timing starts from prepared structure files."
        ),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Validation Post-freeze Efficiency",
        "",
        "| method | timing scope | graphs | seconds/complex | parameters |",
        "|---|---|---:|---:|---:|",
        f"| LAIGN | cached-graph 3-scorer inference + feature assembly | 70 | "
        f"{payload['LAIGN']['seconds_per_graph']['mean']:.4f} +/- "
        f"{payload['LAIGN']['seconds_per_graph']['std']:.4f} | "
        f"{payload['LAIGN']['ensemble_parameters']:,} + {payload['LAIGN']['stage2_linear_parameters']} |",
    ]
    for method, row in payload["external_tools"].items():
        lines.append(
            f"| {method} | official tool from prepared structure files | {row['graphs']} | "
            f"{row['mean_seconds_per_graph']:.4f} | n/a |"
        )
    lines.extend(["", payload["comparability_note"]])
    table = Path(args.table)
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(out), "table": str(table)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
