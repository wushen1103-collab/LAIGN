#!/usr/bin/env python3
"""Compute same-pose controls on the frozen PLINDER-200 subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVES = [
    ROOT / f"outputs/stage2_statistics_statistics_laign_full_seed{seed}/predictions.npz"
    for seed in (2401, 2402, 2403)
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        default=str(ROOT / "outputs/direct_sota_benchmark_laign_probability_ensemble.csv.gz"),
    )
    parser.add_argument("--archives", nargs="+", default=[str(path) for path in DEFAULT_ARCHIVES])
    parser.add_argument(
        "--out-json", default=str(ROOT / "outputs/plinder200_same_pose_controls_matched_summary.json")
    )
    parser.add_argument(
        "--table", default=str(ROOT / "outputs/tables/table_plinder200_same_pose_controls_matched.md")
    )
    return parser.parse_args()


def graph_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan"), float("nan"), float("nan")
    order = np.argsort(-scores, kind="stable")
    predicted = set(order[:positives].tolist())
    truth = set(np.flatnonzero(labels).tolist())
    intersection = len(predicted & truth)
    union = len(predicted | truth)
    recall10 = float(labels[order[: min(10, labels.size)]].sum() / positives)
    return float(average_precision_score(labels, scores)), intersection / max(1, union), recall10


def evaluate_archive(path: Path, selected: dict[str, set[str]], method: str) -> dict[str, list[float]]:
    archive = np.load(path, allow_pickle=True)
    out: dict[str, list[float]] = {}
    for split in ("external_val", "external_test"):
        labels = archive[f"{split}__labels"].astype(np.int32)
        sample_index = archive[f"{split}__sample_index"].astype(np.int32)
        sample_ids = archive[f"{split}__sample_ids"].astype(str)
        scores = archive[f"{split}__score__{method}"].astype(np.float64)
        rows = []
        for index, sample_id in enumerate(sample_ids):
            if sample_id not in selected[split]:
                continue
            mask = sample_index == index
            rows.append(graph_metrics(labels[mask], scores[mask]))
        values = np.asarray(rows, dtype=np.float64)
        out[split] = [float(np.nanmean(values[:, column])) for column in range(3)]
    return out


def combined_rows(path: Path, selected: dict[str, set[str]], method: str) -> np.ndarray:
    archive = np.load(path, allow_pickle=True)
    rows = []
    for split in ("external_val", "external_test"):
        labels = archive[f"{split}__labels"].astype(np.int32)
        sample_index = archive[f"{split}__sample_index"].astype(np.int32)
        sample_ids = archive[f"{split}__sample_ids"].astype(str)
        scores = archive[f"{split}__score__{method}"].astype(np.float64)
        for index, sample_id in enumerate(sample_ids):
            if sample_id in selected[split]:
                mask = sample_index == index
                rows.append(graph_metrics(labels[mask], scores[mask]))
    return np.asarray(rows, dtype=np.float64)


def combined_ensemble_rows(
    paths: list[Path], selected: dict[str, set[str]]
) -> np.ndarray:
    archives = [np.load(path, allow_pickle=True) for path in paths]
    rows = []
    for split in ("external_val", "external_test"):
        labels = archives[0][f"{split}__labels"].astype(np.int32)
        sample_index = archives[0][f"{split}__sample_index"].astype(np.int32)
        sample_ids = archives[0][f"{split}__sample_ids"].astype(str)
        scores = np.mean(
            [archive[f"{split}__score__trained_stage2"].astype(np.float64) for archive in archives],
            axis=0,
        )
        for archive in archives[1:]:
            if not np.array_equal(labels, archive[f"{split}__labels"].astype(np.int32)):
                raise ValueError(f"Label order differs across archives for {split}.")
            if not np.array_equal(sample_index, archive[f"{split}__sample_index"].astype(np.int32)):
                raise ValueError(f"Sample index differs across archives for {split}.")
            if not np.array_equal(sample_ids, archive[f"{split}__sample_ids"].astype(str)):
                raise ValueError(f"Sample identifiers differ across archives for {split}.")
        for index, sample_id in enumerate(sample_ids):
            if sample_id in selected[split]:
                mask = sample_index == index
                rows.append(graph_metrics(labels[mask], scores[mask]))
    return np.asarray(rows, dtype=np.float64)


def paired_test(delta: np.ndarray, seed: int) -> dict[str, float | list[float]]:
    delta = np.asarray(delta, dtype=np.float64)
    delta = delta[np.isfinite(delta)]
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [delta[rng.integers(0, delta.size, delta.size)].mean() for _ in range(10000)]
    )
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(10000, delta.size))
    null = np.mean(signs * delta[None, :], axis=1)
    observed = float(delta.mean())
    p_value = float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / 10001)
    return {
        "graphs": int(delta.size),
        "mean_difference": observed,
        "ci_95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "sign_flip_p": p_value,
        "win_rate": float(np.mean(delta > 1e-12)),
        "tie_rate": float(np.mean(np.abs(delta) <= 1e-12)),
    }


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def main() -> int:
    args = parse_args()
    subset = pd.read_csv(args.subset, usecols=["sample_id", "split"])
    selected = {
        split: set(subset.loc[subset["split"] == split, "sample_id"].astype(str))
        for split in ("external_val", "external_test")
    }
    methods = ["raw_distance", "fixed_hybrid", "trained_stage2"]
    collected = {
        split: {method: {metric: [] for metric in ("graph_ap", "iou_at_p", "recall_at_10")} for method in methods}
        for split in selected
    }
    for archive_path in map(Path, args.archives):
        for method in methods:
            rows = evaluate_archive(archive_path, selected, method)
            for split, metrics in rows.items():
                for name, value in zip(("graph_ap", "iou_at_p", "recall_at_10"), metrics):
                    collected[split][method][name].append(value)

    results = {
        split: {
            method: {metric: summarize(values) for metric, values in metrics.items()}
            for method, metrics in methods_rows.items()
        }
        for split, methods_rows in collected.items()
    }
    rng = np.random.default_rng(7901)
    combined = {}
    first_archive = Path(args.archives[0])
    control_rows = {}
    for method in ("raw_distance", "fixed_hybrid"):
        rows = combined_rows(first_archive, selected, method)
        control_rows[method] = rows
        draws = np.asarray(
            [np.nanmean(rows[rng.integers(0, rows.shape[0], rows.shape[0])], axis=0) for _ in range(1000)]
        )
        combined[method] = {
            name: {"point": float(np.nanmean(rows[:, index])), "bootstrap_sd": float(draws[:, index].std(ddof=1))}
            for index, name in enumerate(("graph_ap", "iou_at_p", "recall_at_10"))
        }
    ensemble_rows = combined_ensemble_rows(list(map(Path, args.archives)), selected)
    paired = {
        method: {
            name: paired_test(ensemble_rows[:, index] - rows[:, index], 7910 + method_index * 10 + index)
            for index, name in enumerate(("graph_ap", "iou_at_p", "recall_at_10"))
        }
        for method_index, (method, rows) in enumerate(control_rows.items())
    }
    payload = {
        "hypothesis": "Matched-control",
        "selected_graphs": {split: len(ids) for split, ids in selected.items()},
        "archives": [str(Path(path).relative_to(ROOT)) for path in args.archives],
        "results": results,
        "combined_plinder200": combined,
        "paired_laign_ensemble_minus_control": paired,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Matched-control PLINDER-200 Same-Pose Controls",
        "",
        "| split | method | graph AP | IoU@P | recall@10 |",
        "|---|---|---:|---:|---:|",
    ]
    for split, method_rows in results.items():
        for method, row in method_rows.items():
            lines.append(
                f"| {split} | {method} | {row['graph_ap']['mean']:.4f} +/- {row['graph_ap']['std']:.4f} | "
                f"{row['iou_at_p']['mean']:.4f} +/- {row['iou_at_p']['std']:.4f} | "
                f"{row['recall_at_10']['mean']:.4f} +/- {row['recall_at_10']['std']:.4f} |"
            )
    lines.extend(["", "## Combined PLINDER-200 deterministic controls", ""])
    lines.append("| method | graph AP | IoU@P | recall@10 |")
    lines.append("|---|---:|---:|---:|")
    for method, row in combined.items():
        lines.append(
            f"| {method} | {row['graph_ap']['point']:.4f} +/- {row['graph_ap']['bootstrap_sd']:.4f} | "
            f"{row['iou_at_p']['point']:.4f} +/- {row['iou_at_p']['bootstrap_sd']:.4f} | "
            f"{row['recall_at_10']['point']:.4f} +/- {row['recall_at_10']['bootstrap_sd']:.4f} |"
        )
    lines.extend(["", "## Paired LAIGN ensemble comparison", ""])
    lines.append("| control | metric | graphs | mean difference | 95% CI | sign-flip p | win rate |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for method, metric_rows in paired.items():
        for metric, row in metric_rows.items():
            low, high = row["ci_95"]
            p_text = "<0.0001" if row["sign_flip_p"] < 0.0001 else f"{row['sign_flip_p']:.4f}"
            lines.append(
                f"| {method} | {metric} | {row['graphs']} | {row['mean_difference']:+.4f} | "
                f"[{low:+.4f}, {high:+.4f}] | {p_text} | {row['win_rate']:.3f} |"
            )
    table = Path(args.table)
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(out), "table": str(table)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
