#!/usr/bin/env python3
"""Cluster-aware paired statistics for typed model versus LAIGN Stage-2 predictions."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typed_model-pattern", required=True)
    parser.add_argument("--laign-pattern", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2401, 2402, 2403])
    parser.add_argument("--splits", nargs="+", default=["external_val", "external_test"])
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--permutation-replicates", type=int, default=100000)
    parser.add_argument("--random-seed", type=int, default=2608)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    return parser.parse_args()


def metric_pair(labels, left, right):
    return {
        "typed_model_auprc": float(average_precision_score(labels, left)),
        "laign_auprc": float(average_precision_score(labels, right)),
        "typed_model_auroc": float(roc_auc_score(labels, left)),
        "laign_auroc": float(roc_auc_score(labels, right)),
    }


def percentile_interval(values):
    low, high = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
    return [float(low), float(high)]


def grouped_indices(sample_index):
    order = np.argsort(sample_index, kind="stable")
    _, starts = np.unique(sample_index[order], return_index=True)
    ends = np.r_[starts[1:], order.size]
    return [order[start:end] for start, end in zip(starts, ends)]


def paired_permutation_pvalue(differences, replicates, rng):
    differences = np.asarray(differences, dtype=np.float64)
    observed = abs(float(differences.mean()))
    if differences.size == 0 or observed == 0.0:
        return 1.0
    exceed = 0
    done = 0
    batch_size = 2000
    while done < replicates:
        size = min(batch_size, replicates - done)
        signs = rng.integers(0, 2, size=(size, differences.size), dtype=np.int8) * 2 - 1
        permuted = np.abs((signs * differences).mean(axis=1))
        exceed += int(np.count_nonzero(permuted >= observed - 1e-15))
        done += size
    return float((exceed + 1) / (replicates + 1))


def analyze_task(task):
    seed, split, typed_model_path, laign_path, bootstrap_replicates, permutation_replicates, random_seed = task
    left = np.load(typed_model_path, allow_pickle=False)
    right = np.load(laign_path, allow_pickle=False)
    labels_key = f"{split}__labels"
    groups_key = f"{split}__sample_index"
    ids_key = f"{split}__sample_ids"
    score_key = f"{split}__score__trained_stage2"
    labels = left[labels_key].astype(np.int8, copy=False)
    sample_index = left[groups_key].astype(np.int32, copy=False)
    if not np.array_equal(labels, right[labels_key]):
        raise RuntimeError(f"Label mismatch for seed={seed} split={split}")
    if not np.array_equal(sample_index, right[groups_key]):
        raise RuntimeError(f"Graph-index mismatch for seed={seed} split={split}")
    if not np.array_equal(left[ids_key], right[ids_key]):
        raise RuntimeError(f"Sample-ID mismatch for seed={seed} split={split}")
    typed_model = left[score_key].astype(np.float64, copy=False)
    laign = right[score_key].astype(np.float64, copy=False)
    if not (np.isfinite(typed_model).all() and np.isfinite(laign).all()):
        raise RuntimeError(f"Non-finite scores for seed={seed} split={split}")

    point = metric_pair(labels, typed_model, laign)
    point["auprc_delta"] = point["laign_auprc"] - point["typed_model_auprc"]
    point["auroc_delta"] = point["laign_auroc"] - point["typed_model_auroc"]
    graph_rows = grouped_indices(sample_index)
    rng = np.random.default_rng(random_seed + seed * 17 + sum(map(ord, split)))
    bootstrap = {key: [] for key in ["typed_model_auprc", "laign_auprc", "auprc_delta", "typed_model_auroc", "laign_auroc", "auroc_delta"]}
    for _ in range(bootstrap_replicates):
        draw = rng.integers(0, len(graph_rows), size=len(graph_rows))
        indices = np.concatenate([graph_rows[index] for index in draw])
        values = metric_pair(labels[indices], typed_model[indices], laign[indices])
        values["auprc_delta"] = values["laign_auprc"] - values["typed_model_auprc"]
        values["auroc_delta"] = values["laign_auroc"] - values["typed_model_auroc"]
        for key, value in values.items():
            bootstrap[key].append(value)

    graph_auprc_deltas = []
    for indices in graph_rows:
        y = labels[indices]
        if y.min() == y.max():
            continue
        graph_auprc_deltas.append(
            float(average_precision_score(y, laign[indices]) - average_precision_score(y, typed_model[indices]))
        )
    pvalue = paired_permutation_pvalue(graph_auprc_deltas, permutation_replicates, rng)
    return {
        "seed": seed,
        "split": split,
        "graphs": len(graph_rows),
        "residues": int(labels.size),
        "positives": int(labels.sum()),
        "point": point,
        "bootstrap_95_ci": {key: percentile_interval(values) for key, values in bootstrap.items()},
        "paired_graph_auprc_permutation_p": pvalue,
        "paired_graphs_with_both_classes": len(graph_auprc_deltas),
        "bootstrap_values": bootstrap,
    }


def describe(values):
    values = [float(value) for value in values]
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def seed_signflip_pvalue(differences):
    differences = np.asarray(differences, dtype=np.float64)
    observed = abs(float(differences.mean()))
    values = []
    for mask in range(1 << differences.size):
        signs = np.array([1.0 if mask & (1 << index) else -1.0 for index in range(differences.size)])
        values.append(abs(float((signs * differences).mean())))
    return float(np.mean(np.asarray(values) >= observed - 1e-15))


def fmt(value):
    return f"{float(value):.4f}"


def fmt_ci(values):
    return f"[{values[0]:.4f}, {values[1]:.4f}]"


def main():
    args = parse_args()
    tasks = []
    for seed in args.seeds:
        for split in args.splits:
            tasks.append(
                (
                    seed,
                    split,
                    args.typed_model_pattern.format(seed=seed),
                    args.laign_pattern.format(seed=seed),
                    args.bootstrap_replicates,
                    args.permutation_replicates,
                    args.random_seed,
                )
            )
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        rows = list(pool.map(analyze_task, tasks))

    summary = []
    for split in args.splits:
        selected = [row for row in rows if row["split"] == split]
        item = {"split": split}
        for key in ["typed_model_auprc", "laign_auprc", "auprc_delta", "typed_model_auroc", "laign_auroc", "auroc_delta"]:
            item[key] = describe(row["point"][key] for row in selected)
            aggregate_bootstrap = np.mean(
                np.asarray([row["bootstrap_values"][key] for row in selected], dtype=np.float64), axis=0
            )
            item[f"{key}_cluster_bootstrap_95_ci"] = percentile_interval(aggregate_bootstrap)
        item["seed_signflip_auprc_p"] = seed_signflip_pvalue(
            [row["point"]["auprc_delta"] for row in selected]
        )
        summary.append(item)

    for row in rows:
        row.pop("bootstrap_values")
    output = {
        "title": "Statistics Cluster-Aware Paired Stage-2 Statistics",
        "comparison": "LAIGN three-checkpoint consensus versus typed model single-checkpoint typed multi-scale",
        "seeds": args.seeds,
        "splits": args.splits,
        "bootstrap_replicates": args.bootstrap_replicates,
        "permutation_replicates": args.permutation_replicates,
        "rows": rows,
        "summary": summary,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Statistics Cluster-Aware Paired Stage-2 Statistics",
        "",
        "- comparison: LAIGN consensus versus typed model typed multi-scale",
        f"- graph-cluster bootstrap replicates: `{args.bootstrap_replicates}`",
        f"- paired graph-level permutation replicates: `{args.permutation_replicates}`",
        "",
        "## Three-Seed Summary",
        "",
        "| split | typed model AUPRC | LAIGN AUPRC | delta | delta cluster-bootstrap 95% CI | seed sign-flip p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['split']} | {fmt(row['typed_model_auprc']['mean'])} +/- {fmt(row['typed_model_auprc']['std'])} | "
            f"{fmt(row['laign_auprc']['mean'])} +/- {fmt(row['laign_auprc']['std'])} | "
            f"{fmt(row['auprc_delta']['mean'])} +/- {fmt(row['auprc_delta']['std'])} | "
            f"{fmt_ci(row['auprc_delta_cluster_bootstrap_95_ci'])} | {row['seed_signflip_auprc_p']:.4g} |"
        )
    lines += [
        "",
        "## Per-Seed Paired Tests",
        "",
        "| seed | split | graphs | residues | typed model AUPRC | LAIGN AUPRC | delta | delta bootstrap 95% CI | graph permutation p |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['split']} | {row['graphs']} | {row['residues']} | "
            f"{fmt(row['point']['typed_model_auprc'])} | {fmt(row['point']['laign_auprc'])} | "
            f"{fmt(row['point']['auprc_delta'])} | {fmt_ci(row['bootstrap_95_ci']['auprc_delta'])} | "
            f"{row['paired_graph_auprc_permutation_p']:.4g} |"
        )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(lines) + "\n")
    print(json.dumps({"json": str(args.out_json), "table": str(args.table)}))


if __name__ == "__main__":
    main()
