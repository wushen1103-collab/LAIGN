#!/usr/bin/env python3
"""Distance-conditioned validation on experimentally assayed Platinum sites.

The analysis uses only assayed sites. It reports conventional site metrics,
near-ligand subsets, grouped out-of-fold incremental models, and within-complex
distance-matched positive/negative pairs. Uncertainty is blocked by complex.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


METHOD_COLUMNS = {
    "LAIGN": "LAIGN",
    "Fixed fusion": "fixed fusion",
    "Distance": "distance_score",
    "Raw interaction": "raw interaction",
    "PLIP count": "PLIP count",
    "ProLIF count": "ProLIF count",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=root
        / "source_data"
        / "validation"
        / "platinum_assayed_site_scores.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "analysis" / "results"
    )
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--cv-repeats", type=int, default=20)
    parser.add_argument(
        "--match-calipers", type=float, nargs="+", default=[0.05, 0.10, 0.25, 0.50]
    )
    return parser.parse_args()


def safe_metrics(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return float(average_precision_score(y, score)), float(roc_auc_score(y, score))


def subset_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    subsets: list[tuple[str, pd.Series]] = [
        ("All assayed sites", pd.Series(True, index=frame.index)),
        ("Distance <= 6 A", frame["min_ligand_distance"] <= 6.0),
        ("Distance <= 8 A", frame["min_ligand_distance"] <= 8.0),
        ("Distance <= 4 A", frame["min_ligand_distance"] <= 4.0),
        (
            "Distance (4, 6] A",
            (frame["min_ligand_distance"] > 4.0)
            & (frame["min_ligand_distance"] <= 6.0),
        ),
        (
            "Distance (6, 8] A",
            (frame["min_ligand_distance"] > 6.0)
            & (frame["min_ligand_distance"] <= 8.0),
        ),
        ("Distance > 8 A", frame["min_ligand_distance"] > 8.0),
    ]
    rows = []
    for subset_name, mask in subsets:
        part = frame.loc[mask]
        y = part["outcome"].to_numpy(dtype=int)
        for method, column in METHOD_COLUMNS.items():
            ap, auc = safe_metrics(y, part[column].to_numpy(dtype=float))
            rows.append(
                {
                    "subset": subset_name,
                    "method": method,
                    "complexes": int(part["sample_id"].nunique()),
                    "sites": int(len(part)),
                    "positives": int(y.sum()),
                    "prevalence": float(y.mean()) if len(y) else np.nan,
                    "AUPRC": ap,
                    "AUROC": auc,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_subset_effects(
    frame: pd.DataFrame, iterations: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset_masks: list[tuple[str, pd.Series]] = [
        ("All assayed sites", pd.Series(True, index=frame.index)),
        ("Distance <= 6 A", frame["min_ligand_distance"] <= 6.0),
        ("Distance <= 8 A", frame["min_ligand_distance"] <= 8.0),
        ("Distance <= 4 A", frame["min_ligand_distance"] <= 4.0),
        (
            "Distance (4, 6] A",
            (frame["min_ligand_distance"] > 4.0)
            & (frame["min_ligand_distance"] <= 6.0),
        ),
        (
            "Distance (6, 8] A",
            (frame["min_ligand_distance"] > 6.0)
            & (frame["min_ligand_distance"] <= 8.0),
        ),
        ("Distance > 8 A", frame["min_ligand_distance"] > 8.0),
    ]
    rng = np.random.default_rng(seed)
    bootstrap_rows = []
    summary_rows = []
    for subset_name, mask in subset_masks:
        part = frame.loc[mask].copy()
        group_codes, group_names = pd.factorize(part["sample_id"].astype(str))
        n_groups = len(group_names)
        y = part["outcome"].to_numpy(dtype=int)
        laign_score = part["LAIGN"].to_numpy(dtype=float)
        fixed_score = part["fixed fusion"].to_numpy(dtype=float)
        distance_score = part["distance_score"].to_numpy(dtype=float)
        point_ap = {
            method: safe_metrics(y, part[column].to_numpy(dtype=float))[0]
            for method, column in METHOD_COLUMNS.items()
        }
        subset_boot = []
        for iteration in range(iterations):
            sampled = rng.integers(0, n_groups, size=n_groups)
            group_weights = np.bincount(sampled, minlength=n_groups)
            weights = group_weights[group_codes]
            if np.sum(weights[y == 1]) == 0 or np.sum(weights[y == 0]) == 0:
                continue
            ap_laign = average_precision_score(y, laign_score, sample_weight=weights)
            ap_fixed = average_precision_score(y, fixed_score, sample_weight=weights)
            ap_distance = average_precision_score(
                y, distance_score, sample_weight=weights
            )
            row = {
                "subset": subset_name,
                "iteration": iteration + 1,
                "delta_AUPRC_LAIGN_vs_distance": float(ap_laign - ap_distance),
                "delta_AUPRC_LAIGN_vs_fixed": float(ap_laign - ap_fixed),
            }
            bootstrap_rows.append(row)
            subset_boot.append(row)
        subset_boot_frame = pd.DataFrame(subset_boot)
        for comparison, point in [
            ("LAIGN - Distance", point_ap["LAIGN"] - point_ap["Distance"]),
            ("LAIGN - Fixed fusion", point_ap["LAIGN"] - point_ap["Fixed fusion"]),
        ]:
            column = (
                "delta_AUPRC_LAIGN_vs_distance"
                if comparison.endswith("Distance")
                else "delta_AUPRC_LAIGN_vs_fixed"
            )
            values = subset_boot_frame[column]
            summary_rows.append(
                {
                    "subset": subset_name,
                    "comparison": comparison,
                    "complexes": int(part["sample_id"].nunique()),
                    "sites": int(len(part)),
                    "positives": int(part["outcome"].sum()),
                    "AUPRC_difference": float(point),
                    "CI95_low": float(values.quantile(0.025)),
                    "CI95_high": float(values.quantile(0.975)),
                    "bootstrap_probability_positive": float(np.mean(values > 0)),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(bootstrap_rows)


def grouped_oof_predictions(
    frame: pd.DataFrame, repeats: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_sets = {
        "Distance only": ["min_ligand_distance"],
        "Distance + LAIGN": ["min_ligand_distance", "LAIGN"],
        "Distance + fixed fusion": ["min_ligand_distance", "fixed fusion"],
    }
    y = frame["outcome"].to_numpy(dtype=int)
    groups = frame["sample_id"].astype(str).to_numpy()
    prediction_sums = {name: np.zeros(len(frame), dtype=float) for name in feature_sets}
    prediction_counts = np.zeros(len(frame), dtype=int)
    fold_rows: list[dict[str, float | int | str]] = []

    for repeat in range(repeats):
        cv = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=seed + repeat
        )
        for fold, (train_idx, test_idx) in enumerate(
            cv.split(np.zeros(len(frame)), y, groups), start=1
        ):
            prediction_counts[test_idx] += 1
            for model_name, features in feature_sets.items():
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=2000,
                        solver="lbfgs",
                    ),
                )
                model.fit(frame.iloc[train_idx][features], y[train_idx])
                pred = model.predict_proba(frame.iloc[test_idx][features])[:, 1]
                prediction_sums[model_name][test_idx] += pred
                ap, auc = safe_metrics(y[test_idx], pred)
                fold_rows.append(
                    {
                        "repeat": repeat + 1,
                        "fold": fold,
                        "model": model_name,
                        "test_complexes": int(len(np.unique(groups[test_idx]))),
                        "test_sites": int(len(test_idx)),
                        "test_positives": int(y[test_idx].sum()),
                        "AUPRC": ap,
                        "AUROC": auc,
                    }
                )

    if not np.all(prediction_counts == repeats):
        raise RuntimeError("Each site must receive one held-out prediction per repeat.")
    oof = frame[["sample_id", "outcome", "min_ligand_distance", "LAIGN", "fixed fusion"]].copy()
    for model_name, values in prediction_sums.items():
        oof[model_name] = values / prediction_counts
    return oof, pd.DataFrame(fold_rows)


def cluster_bootstrap_oof(
    oof: pd.DataFrame, iterations: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    group_codes, group_names = pd.factorize(oof["sample_id"].astype(str))
    n_groups = len(group_names)
    models = ["Distance only", "Distance + LAIGN", "Distance + fixed fusion"]
    point_rows = []
    y = oof["outcome"].to_numpy(dtype=int)
    for model in models:
        ap, auc = safe_metrics(y, oof[model].to_numpy(dtype=float))
        point_rows.append({"model": model, "AUPRC": ap, "AUROC": auc})

    boot_rows = []
    for index in range(iterations):
        sampled = rng.integers(0, n_groups, size=n_groups)
        group_weights = np.bincount(sampled, minlength=n_groups)
        weights = group_weights[group_codes]
        row: dict[str, float | int] = {"iteration": index + 1}
        for model in models:
            score = oof[model].to_numpy(dtype=float)
            ap = average_precision_score(y, score, sample_weight=weights)
            auc = roc_auc_score(y, score, sample_weight=weights)
            key = model.lower().replace(" ", "_").replace("+", "plus")
            row[f"{key}_AUPRC"] = ap
            row[f"{key}_AUROC"] = auc
        row["delta_AUPRC_LAIGN_vs_distance"] = (
            row["distance_plus_laign_AUPRC"] - row["distance_only_AUPRC"]
        )
        row["delta_AUROC_LAIGN_vs_distance"] = (
            row["distance_plus_laign_AUROC"] - row["distance_only_AUROC"]
        )
        row["delta_AUPRC_LAIGN_vs_fixed"] = (
            row["distance_plus_laign_AUPRC"]
            - row["distance_plus_fixed_fusion_AUPRC"]
        )
        boot_rows.append(row)
    return pd.DataFrame(point_rows), pd.DataFrame(boot_rows)


def match_distance_pairs(
    frame: pd.DataFrame, max_distance: float, caliper: float
) -> pd.DataFrame:
    rows = []
    for sample_id, group in frame.groupby("sample_id", sort=False):
        group = group[group["min_ligand_distance"] <= max_distance]
        positive = group[group["outcome"] == 1]
        negative = group[group["outcome"] == 0]
        if positive.empty or negative.empty:
            continue
        costs = np.abs(
            positive["min_ligand_distance"].to_numpy()[:, None]
            - negative["min_ligand_distance"].to_numpy()[None, :]
        )
        positive_idx, negative_idx = linear_sum_assignment(costs)
        for p_idx, n_idx in zip(positive_idx, negative_idx):
            gap = float(costs[p_idx, n_idx])
            if gap > caliper:
                continue
            p = positive.iloc[p_idx]
            n = negative.iloc[n_idx]
            row = {
                "sample_id": sample_id,
                "max_distance": max_distance,
                "caliper": caliper,
                "distance_gap": gap,
                "positive_distance": float(p["min_ligand_distance"]),
                "negative_distance": float(n["min_ligand_distance"]),
            }
            for method, column in METHOD_COLUMNS.items():
                row[f"{method}_difference"] = float(p[column] - n[column])
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pairs.empty:
        return pd.DataFrame(rows)
    for (max_distance, caliper), part in pairs.groupby(["max_distance", "caliper"]):
        for method in METHOD_COLUMNS:
            values = part[f"{method}_difference"].to_numpy(dtype=float)
            pair_accuracy = float(np.mean((values > 0) + 0.5 * (values == 0)))
            try:
                p_value = float(wilcoxon(values, alternative="greater").pvalue)
            except ValueError:
                p_value = np.nan
            rows.append(
                {
                    "max_distance": float(max_distance),
                    "caliper": float(caliper),
                    "method": method,
                    "complexes": int(part["sample_id"].nunique()),
                    "matched_pairs": int(len(part)),
                    "mean_distance_gap": float(part["distance_gap"].mean()),
                    "median_score_difference": float(np.median(values)),
                    "mean_score_difference": float(np.mean(values)),
                    "pair_accuracy": pair_accuracy,
                    "one_sided_wilcoxon_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def percentile_interval(values: pd.Series) -> list[float]:
    clean = values.dropna().to_numpy(dtype=float)
    return [float(np.quantile(clean, 0.025)), float(np.quantile(clean, 0.975))]


def write_report(
    output: Path,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    oof_points: pd.DataFrame,
    bootstrap: pd.DataFrame,
    subset_effects: pd.DataFrame,
    pair_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    primary = metrics[
        metrics["subset"].isin(["All assayed sites", "Distance <= 6 A", "Distance <= 8 A"])
        & metrics["method"].isin(["LAIGN", "Fixed fusion", "Distance"])
    ].copy()
    delta_ap = bootstrap["delta_AUPRC_LAIGN_vs_distance"]
    delta_auc = bootstrap["delta_AUROC_LAIGN_vs_distance"]
    lines = [
        "# Platinum distance-conditioned validation",
        "",
        "## Protocol",
        "",
        f"- Assayed sites: {len(frame)} from {frame['sample_id'].nunique()} complexes.",
        f"- Disruptive sites at 1 kcal mol^-1: {int(frame['outcome'].sum())}.",
        "- Site metrics use only experimentally assayed residues.",
        f"- Incremental models use {args.cv_repeats} repeated five-fold stratified group splits by complex.",
        f"- Confidence intervals use {args.bootstrap} complex-blocked bootstrap samples.",
        "- Distance-matched pairs are formed within complex across 0.05, 0.10, 0.25 and 0.50 A calipers.",
        "",
        "## Near-ligand site metrics",
        "",
        primary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Distance-stratified paired AUPRC effects",
        "",
        subset_effects.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Distance-conditioned out-of-fold models",
        "",
        oof_points.to_markdown(index=False, floatfmt=".4f"),
        "",
        (
            "Adding LAIGN to distance changes out-of-fold AUPRC by "
            f"{delta_ap.mean():.4f} (95% complex-blocked bootstrap CI "
            f"[{percentile_interval(delta_ap)[0]:.4f}, {percentile_interval(delta_ap)[1]:.4f}]) "
            "and AUROC by "
            f"{delta_auc.mean():.4f} (95% CI "
            f"[{percentile_interval(delta_auc)[0]:.4f}, {percentile_interval(delta_auc)[1]:.4f}])."
        ),
        "",
        "## Within-complex distance-matched comparisons",
        "",
        pair_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "The analysis is descriptive validation of incremental association; all model comparisons use identical assayed sites and grouped resampling.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)
    required = {
        "sample_id",
        "min_ligand_distance",
        "disruptive_ge_1",
        "LAIGN",
        "fixed fusion",
        "raw interaction",
        "PLIP count",
        "ProLIF count",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    frame = frame.dropna(subset=list(required)).copy()
    frame["outcome"] = frame["disruptive_ge_1"].astype(int)
    frame["distance_score"] = -frame["min_ligand_distance"].astype(float)

    metrics = subset_metrics(frame)
    subset_effects, subset_effect_bootstrap = bootstrap_subset_effects(
        frame, args.bootstrap, args.seed + 5_000
    )
    oof, fold_metrics = grouped_oof_predictions(frame, args.cv_repeats, args.seed)
    oof_points, bootstrap = cluster_bootstrap_oof(
        oof, args.bootstrap, args.seed + 10_000
    )
    pairs = pd.concat(
        [
            match_distance_pairs(frame, max_distance, caliper)
            for max_distance in (6.0, 8.0)
            for caliper in args.match_calipers
        ],
        ignore_index=True,
    )
    pair_summary = summarize_pairs(pairs)

    metrics.to_csv(args.output_dir / "platinum_distance_strata_metrics.csv", index=False)
    subset_effects.to_csv(
        args.output_dir / "platinum_distance_strata_paired_effects.csv", index=False
    )
    subset_effect_bootstrap.to_csv(
        args.output_dir / "platinum_distance_strata_bootstrap.csv", index=False
    )
    oof.to_csv(args.output_dir / "platinum_distance_conditioned_oof.csv", index=False)
    fold_metrics.to_csv(
        args.output_dir / "platinum_distance_conditioned_cv_folds.csv", index=False
    )
    bootstrap.to_csv(
        args.output_dir / "platinum_distance_conditioned_bootstrap.csv", index=False
    )
    pairs.to_csv(args.output_dir / "platinum_distance_matched_pairs.csv", index=False)
    pair_summary.to_csv(
        args.output_dir / "platinum_distance_matched_pair_summary.csv", index=False
    )

    summary = {
        "sites": int(len(frame)),
        "complexes": int(frame["sample_id"].nunique()),
        "positives": int(frame["outcome"].sum()),
        "cv_repeats": args.cv_repeats,
        "bootstrap_iterations": args.bootstrap,
        "match_calipers_angstrom": args.match_calipers,
        "distance_strata_paired_effects": subset_effects.to_dict(orient="records"),
        "oof_metrics": oof_points.to_dict(orient="records"),
        "delta_AUPRC_LAIGN_vs_distance": {
            "mean": float(bootstrap["delta_AUPRC_LAIGN_vs_distance"].mean()),
            "ci95": percentile_interval(bootstrap["delta_AUPRC_LAIGN_vs_distance"]),
            "probability_positive": float(
                np.mean(bootstrap["delta_AUPRC_LAIGN_vs_distance"] > 0)
            ),
        },
        "delta_AUROC_LAIGN_vs_distance": {
            "mean": float(bootstrap["delta_AUROC_LAIGN_vs_distance"].mean()),
            "ci95": percentile_interval(bootstrap["delta_AUROC_LAIGN_vs_distance"]),
            "probability_positive": float(
                np.mean(bootstrap["delta_AUROC_LAIGN_vs_distance"] > 0)
            ),
        },
        "matched_pair_summary": pair_summary.to_dict(orient="records"),
    }
    (args.output_dir / "platinum_distance_conditioned_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_report(
        args.output_dir / "PLATINUM_DISTANCE_CONDITIONED_REPORT.md",
        frame,
        metrics,
        oof_points,
        bootstrap,
        subset_effects,
        pair_summary,
        args,
    )


if __name__ == "__main__":
    main()
