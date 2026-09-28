#!/usr/bin/env python3
"""Summarize Calibration Stage-2 consensus calibration audit."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


SEEDS = [2401, 2402, 2403]
SPLITS = ["external_val", "external_test"]
METHODS = ["trained_distance", "trained_visnet", "trained_stage2"]
CALIBRATIONS = ["raw_probability", "platt_internal_val"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    return parser.parse_args()


def load(path):
    return json.loads(Path(path).read_text())


def mean_std(values):
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not clean:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": statistics.fmean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
    }


def metric(run, method, split, key):
    return run["methods"][method][split][key]


def calibration_metric(run, method, split, calibration, key):
    return run["methods"][method][split].get("calibration", {}).get(calibration, {}).get(key)


def fmt(value):
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(value):
        return "NA"
    return f"{value:.4f}"


def fmt_mean_std(block):
    if math.isnan(float(block["mean"])):
        return "NA"
    return f"{block['mean']:.4f} +/- {block['std']:.4f}"


def main():
    args = parse_args()
    baseline = {seed: load(f"outputs/stage2_consensus_laign_full_seed{seed}/metrics.json") for seed in SEEDS}
    calibrated = {
        seed: load(f"outputs/stage2_consensus_calibration_calibration_full_seed{seed}/metrics.json") for seed in SEEDS
    }
    smoke = load("outputs/stage2_consensus_calibration_calibration_smoke_seed2401/metrics.json")

    paired_rows = []
    for seed in SEEDS:
        for split in SPLITS:
            row = {
                "seed": seed,
                "split": split,
                "laign_stage2_auprc": metric(baseline[seed], "trained_stage2", split, "auprc"),
                "calibration_stage2_auprc": metric(calibrated[seed], "trained_stage2", split, "auprc"),
                "laign_stage2_auroc": metric(baseline[seed], "trained_stage2", split, "auroc"),
                "calibration_stage2_auroc": metric(calibrated[seed], "trained_stage2", split, "auroc"),
            }
            row["calibration_minus_laign_auprc"] = row["calibration_stage2_auprc"] - row["laign_stage2_auprc"]
            row["calibration_minus_laign_auroc"] = row["calibration_stage2_auroc"] - row["laign_stage2_auroc"]
            paired_rows.append(row)

    paired_summary = []
    for split in SPLITS:
        selected = [row for row in paired_rows if row["split"] == split]
        paired_summary.append(
            {
                "split": split,
                "laign_stage2_auprc": mean_std(row["laign_stage2_auprc"] for row in selected),
                "calibration_stage2_auprc": mean_std(row["calibration_stage2_auprc"] for row in selected),
                "calibration_minus_laign_auprc": mean_std(row["calibration_minus_laign_auprc"] for row in selected),
                "laign_stage2_auroc": mean_std(row["laign_stage2_auroc"] for row in selected),
                "calibration_stage2_auroc": mean_std(row["calibration_stage2_auroc"] for row in selected),
                "calibration_minus_laign_auroc": mean_std(row["calibration_minus_laign_auroc"] for row in selected),
            }
        )

    calibration_summary = []
    for method in METHODS:
        for split in SPLITS:
            row = {"method": method, "split": split}
            for calibration in CALIBRATIONS:
                for key in ["brier", "ece_15", "nll", "risk_coverage_auc"]:
                    row[f"{calibration}_{key}"] = mean_std(
                        calibration_metric(calibrated[seed], method, split, calibration, key) for seed in SEEDS
                    )
            calibration_summary.append(row)

    smoke_row = {
        "external_val_stage2_auprc": metric(smoke, "trained_stage2", "external_val", "auprc"),
        "external_test_stage2_auprc": metric(smoke, "trained_stage2", "external_test", "auprc"),
        "external_test_raw_ece": calibration_metric(
            smoke, "trained_stage2", "external_test", "raw_probability", "ece_15"
        ),
        "external_test_platt_ece": calibration_metric(
            smoke, "trained_stage2", "external_test", "platt_internal_val", "ece_15"
        ),
    }

    output = {
        "title": "Calibration Stage-2 Consensus Calibration Audit",
        "seeds": SEEDS,
        "paired_rows": paired_rows,
        "paired_summary": paired_summary,
        "calibration_summary": calibration_summary,
        "smoke": smoke_row,
        "decision": "use_calibration_for_calibration_reporting_keep_laign_ranking_claim",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Calibration Stage-2 Consensus Calibration Audit",
        "",
        "- model/scoring setup: LAIGN three-checkpoint consensus typed multi-scale Stage-2",
        "- purpose: add S9 calibration metrics without changing the LAIGN ranking claim",
        f"- decision: `{output['decision']}`",
        "",
        "## Ranking Consistency Check",
        "",
        "| split | LAIGN AUPRC | Calibration AUPRC | delta | LAIGN AUROC | Calibration AUROC | delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in paired_summary:
        lines.append(
            f"| {row['split']} | {fmt_mean_std(row['laign_stage2_auprc'])} | "
            f"{fmt_mean_std(row['calibration_stage2_auprc'])} | {fmt_mean_std(row['calibration_minus_laign_auprc'])} | "
            f"{fmt_mean_std(row['laign_stage2_auroc'])} | {fmt_mean_std(row['calibration_stage2_auroc'])} | "
            f"{fmt_mean_std(row['calibration_minus_laign_auroc'])} |"
        )
    lines += [
        "",
        "## Main Stage-2 Calibration",
        "",
        "| split | raw Brier | raw ECE-15 | raw NLL | raw risk-coverage AUC | Platt Brier | Platt ECE-15 | Platt NLL | Platt risk-coverage AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    stage2_rows = [row for row in calibration_summary if row["method"] == "trained_stage2"]
    for row in stage2_rows:
        lines.append(
            f"| {row['split']} | "
            f"{fmt_mean_std(row['raw_probability_brier'])} | {fmt_mean_std(row['raw_probability_ece_15'])} | "
            f"{fmt_mean_std(row['raw_probability_nll'])} | {fmt_mean_std(row['raw_probability_risk_coverage_auc'])} | "
            f"{fmt_mean_std(row['platt_internal_val_brier'])} | {fmt_mean_std(row['platt_internal_val_ece_15'])} | "
            f"{fmt_mean_std(row['platt_internal_val_nll'])} | {fmt_mean_std(row['platt_internal_val_risk_coverage_auc'])} |"
        )
    lines += [
        "",
        "## Calibration Ablation Context",
        "",
        "| method | split | raw ECE-15 | Platt ECE-15 | raw NLL | Platt NLL |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in calibration_summary:
        lines.append(
            f"| {row['method']} | {row['split']} | "
            f"{fmt_mean_std(row['raw_probability_ece_15'])} | {fmt_mean_std(row['platt_internal_val_ece_15'])} | "
            f"{fmt_mean_std(row['raw_probability_nll'])} | {fmt_mean_std(row['platt_internal_val_nll'])} |"
        )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(lines) + "\n")
    print(json.dumps({"decision": output["decision"], "table": str(args.table)}))


if __name__ == "__main__":
    main()
