#!/usr/bin/env python3
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.model import evaluate_fixed_contact_baseline as contact_eval  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate distance, interaction-only, and range-interaction hybrid "
            "scores for PLIP active-interaction residue localization."
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["external_val", "external_test"],
        choices=["external_val", "external_test"],
    )
    parser.add_argument("--candidate-radii", nargs="+", type=float, default=[5.0, 12.0, 999.0])
    parser.add_argument(
        "--positive-interactions",
        nargs="+",
        default=contact_eval.DEFAULT_ACTIVE_INTERACTIONS,
        choices=contact_eval.train_visnet.INTERACTION_TYPES,
    )
    parser.add_argument("--tune-split", default="external_val", choices=["external_val", "external_test"])
    parser.add_argument("--tune-radius", type=float, default=999.0)
    parser.add_argument(
        "--alpha-grid",
        nargs="+",
        type=float,
        default=[i / 20 for i in range(21)],
        help="Hybrid score alpha: alpha * interaction + (1-alpha) * range_prior.",
    )
    parser.add_argument(
        "--tau-grid",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0],
        help="Exponential range prior tau in Angstrom.",
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


def radius_key(radius):
    return f"{float(radius):g}"


def range_prior(distances, tau):
    return np.exp(-np.asarray(distances, dtype=np.float64) / float(tau))


def hybrid_score(visnet_scores, distances, alpha, tau):
    visnet_scores = np.asarray(visnet_scores, dtype=np.float64)
    prior = range_prior(distances, tau)
    return float(alpha) * visnet_scores + (1.0 - float(alpha)) * prior


def method_scores(record, method, alpha=None, tau=None):
    if method == "distance":
        return -record["distances"]
    if method == "visnet":
        return record["visnet_scores"]
    if method == "hybrid":
        return hybrid_score(record["visnet_scores"], record["distances"], alpha, tau)
    raise ValueError(method)


def summarize_entries(entries):
    if not entries:
        return {
            "graphs": 0,
            "residues": 0,
            "positives": 0,
            "prevalence": math.nan,
            "auprc": math.nan,
            "auroc": math.nan,
            "mean_graph_auprc": describe([]),
            "mean_graph_auroc": describe([]),
            "mean_graph_positive_rate": describe([]),
        }
    labels = np.concatenate([entry["labels"] for entry in entries])
    scores = np.concatenate([entry["scores"] for entry in entries])
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
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()) if labels.size else math.nan,
        "auprc": safe_metric(average_precision_score, labels, scores),
        "auroc": safe_metric(roc_auc_score, labels, scores),
        "mean_graph_auprc": describe(graph_auprc),
        "mean_graph_auroc": describe(graph_auroc),
        "mean_graph_positive_rate": describe(graph_positive_rate),
    }


def collect_entries(scored_records, split, radius, method, alpha=None, tau=None):
    entries = []
    for record in scored_records:
        if record["split"] != split:
            continue
        mask = record["distances"] <= float(radius)
        if not mask.any():
            continue
        entries.append(
            {
                "sample_id": record["sample_id"],
                "labels": record["labels"][mask],
                "scores": method_scores(record, method, alpha=alpha, tau=tau)[mask],
            }
        )
    return entries


def tune_hybrid(scored_records, args):
    best = None
    trace = []
    for tau in sorted(set(float(value) for value in args.tau_grid)):
        for alpha in sorted(set(float(value) for value in args.alpha_grid)):
            entries = collect_entries(
                scored_records,
                args.tune_split,
                args.tune_radius,
                "hybrid",
                alpha=alpha,
                tau=tau,
            )
            metrics = summarize_entries(entries)
            row = {
                "alpha": alpha,
                "tau": tau,
                "auprc": metrics["auprc"],
                "auroc": metrics["auroc"],
                "mean_graph_auprc": metrics["mean_graph_auprc"]["mean"],
            }
            trace.append(row)
            score = row["auprc"]
            if math.isnan(score):
                continue
            if best is None or score > best["auprc"] + 1e-12:
                best = row
            elif best is not None and abs(score - best["auprc"]) <= 1e-12:
                # Prefer the simpler range-heavy mixture if validation performance ties.
                if (alpha, tau) < (best["alpha"], best["tau"]):
                    best = row
    if best is None:
        raise RuntimeError("No valid hybrid tuning metric")
    return best, trace


@torch.no_grad()
def score_records(records, args, model, class_indices, device):
    scoring_args = copy.copy(args)
    scoring_args.model = "visnet"
    scored = []
    for record in records:
        visnet_scores = contact_eval.score_record(record, scoring_args, model, class_indices, device)
        scored.append(
            {
                "sample_id": record["sample_id"],
                "split": record["split"],
                "labels": record["labels"],
                "distances": record["distances"],
                "visnet_scores": np.asarray(visnet_scores, dtype=np.float64),
            }
        )
    return scored


def evaluate_methods(scored_records, args, best):
    methods = {
        "distance": {"method": "distance"},
        "visnet": {"method": "visnet"},
        "hybrid": {"method": "hybrid", "alpha": best["alpha"], "tau": best["tau"]},
    }
    out = {}
    for method_name, params in methods.items():
        out[method_name] = {}
        for split in args.splits:
            out[method_name][split] = {}
            for radius in args.candidate_radii:
                entries = collect_entries(
                    scored_records,
                    split,
                    radius,
                    params["method"],
                    alpha=params.get("alpha"),
                    tau=params.get("tau"),
                )
                out[method_name][split][radius_key(radius)] = summarize_entries(entries)
    return out


def fmt(value):
    return "nan" if math.isnan(float(value)) else f"{float(value):.4f}"


def write_table(metrics, path):
    lines = [
        "# PLIP Range-Interaction Hybrid Residue Localization",
        "",
        f"- checkpoint: `{metrics['checkpoint']}`",
        f"- tuned on: `{metrics['tuning']['split']}` radius `{metrics['tuning']['radius']}`",
        f"- selected alpha: `{metrics['tuning']['best']['alpha']:.2f}`",
        f"- selected tau: `{metrics['tuning']['best']['tau']:.2f}`",
        "",
        "| method | split | radius | graphs | residues | positives | prevalence | residue AUPRC | residue AUROC | mean graph AUPRC | mean graph AUROC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ["distance", "visnet", "hybrid"]:
        for split in metrics["splits"]:
            for radius, row in metrics["splits"][split].items():
                method_row = metrics["methods"][method][split][radius]
                lines.append(
                    f"| {method} | {split} | {radius} | {method_row['graphs']} | "
                    f"{method_row['residues']} | {method_row['positives']} | "
                    f"{fmt(method_row['prevalence'])} | {fmt(method_row['auprc'])} | "
                    f"{fmt(method_row['auroc'])} | {fmt(method_row['mean_graph_auprc']['mean'])} | "
                    f"{fmt(method_row['mean_graph_auroc']['mean'])} |"
                )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.candidate_radii = sorted(set(float(radius) for radius in args.candidate_radii))
    if args.tune_radius not in args.candidate_radii:
        args.candidate_radii.append(float(args.tune_radius))
        args.candidate_radii = sorted(set(args.candidate_radii))
    max_radius = max(args.candidate_radii)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    class_indices = contact_eval.active_indices(args.positive_interactions)
    model, history = contact_eval.load_model(args.checkpoint, args, device)
    records, skipped = contact_eval.load_candidate_records(args, max_radius)
    scored_records = score_records(records, args, model, class_indices, device)
    best, trace = tune_hybrid(scored_records, args)
    method_metrics = evaluate_methods(scored_records, args, best)

    split_shape = {}
    for split in args.splits:
        split_shape[split] = {}
        for radius in args.candidate_radii:
            entries = collect_entries(scored_records, split, radius, "distance")
            split_shape[split][radius_key(radius)] = summarize_entries(entries)

    metrics = {
        "checkpoint": args.checkpoint,
        "split_manifest": args.split_manifest,
        "interaction_labels": args.interaction_labels,
        "positive_interactions": args.positive_interactions,
        "candidate_radii": args.candidate_radii,
        "records_loaded": int(len(records)),
        "records_skipped": skipped,
        "device": str(device),
        "finite_representation_guard": bool(args.finite_representation_guard),
        "representation_clip": float(args.representation_clip),
        "tuning": {
            "split": args.tune_split,
            "radius": radius_key(args.tune_radius),
            "alpha_grid": sorted(set(float(value) for value in args.alpha_grid)),
            "tau_grid": sorted(set(float(value) for value in args.tau_grid)),
            "best": best,
            "trace": trace,
        },
        "history": history,
        "splits": split_shape,
        "methods": method_metrics,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_table(metrics, args.table)
    print(json.dumps({"metrics": str(metrics_path), "table": args.table, "best": best}))


if __name__ == "__main__":
    main()
