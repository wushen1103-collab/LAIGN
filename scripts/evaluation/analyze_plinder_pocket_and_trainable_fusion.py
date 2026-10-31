#!/usr/bin/env python3
"""Pocket-restricted PLINDER-200 and trainable two-input fusion analysis.

The script consumes the three Stage-2 prediction archives already produced by
the frozen experiment. It does not retrain the interaction scorer or LAIGN.
A two-input logistic model is fitted on BioLiP2 training predictions using the
same raw interaction maximum and exponential distance prior as fixed fusion;
regularization is selected on the internal-validation split. The selected
model is then evaluated on exactly the same PLINDER-200 complexes as LAIGN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def dataframe_to_markdown(frame: pd.DataFrame, float_digits: int = 4) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]
    rows = []
    for values in frame.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.{float_digits}f}")
            else:
                rendered.append(str(value))
        rows.append(rendered)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


SPLITS = ("external_val", "external_test")
METHODS = ("Distance", "Fixed fusion", "Trainable fusion", "LAIGN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, nargs=3, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=8241)
    return parser.parse_args()


def require_keys(archive: np.lib.npyio.NpzFile, keys: list[str], name: Path) -> None:
    missing = [key for key in keys if key not in archive.files]
    if missing:
        raise KeyError(f"{name} is missing required arrays: {missing}")


def validate_archives(archives: list[np.lib.npyio.NpzFile], paths: list[Path]) -> None:
    required = []
    for split in ("train", "internal_val", *SPLITS):
        required.extend(
            [
                f"{split}__labels",
                f"{split}__sample_index",
                f"{split}__sample_ids",
                f"{split}__score__raw_distance",
                f"{split}__score__raw_visnet",
                f"{split}__score__fixed_hybrid",
                f"{split}__score__trained_stage2",
            ]
        )
    for archive, path in zip(archives, paths):
        require_keys(archive, required, path)
    for split in SPLITS:
        for suffix in ("labels", "sample_index", "sample_ids", "score__raw_distance"):
            key = f"{split}__{suffix}"
            reference = archives[0][key]
            for archive in archives[1:]:
                if not np.array_equal(reference, archive[key]):
                    raise ValueError(f"Archive alignment differs for {key}.")


def distance_from_archive(archive: np.lib.npyio.NpzFile, split: str) -> np.ndarray:
    distance = -archive[f"{split}__score__raw_distance"].astype(np.float64)
    if not np.all(np.isfinite(distance)) or np.any(distance < -1e-8):
        raise ValueError(f"Invalid reconstructed distances for {split}.")
    return np.maximum(distance, 0.0)


def two_input_features(archive: np.lib.npyio.NpzFile, split: str) -> np.ndarray:
    interaction = archive[f"{split}__score__raw_visnet"].astype(np.float64)
    distance_prior = np.exp(-distance_from_archive(archive, split) / 12.0)
    return np.column_stack([interaction, distance_prior])


def mean_graph_ap(labels: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> float:
    values = []
    for group in np.unique(groups):
        mask = groups == group
        y = labels[mask]
        if y.sum() == 0:
            continue
        values.append(average_precision_score(y, scores[mask]))
    return float(np.mean(values))


def fit_trainable_fusion(
    archive: np.lib.npyio.NpzFile, c_grid: list[float], seed: int
) -> tuple[object, pd.DataFrame]:
    train_x = two_input_features(archive, "train")
    train_y = archive["train__labels"].astype(np.int8)
    val_x = two_input_features(archive, "internal_val")
    val_y = archive["internal_val__labels"].astype(np.int8)
    val_groups = archive["internal_val__sample_index"].astype(np.int32)
    trace = []
    selected = None
    for c_value in sorted(set(c_grid)):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                random_state=seed,
            ),
        )
        model.fit(train_x, train_y)
        score = model.predict_proba(val_x)[:, 1]
        graph_ap = mean_graph_ap(val_y, score, val_groups)
        pooled_ap = float(average_precision_score(val_y, score))
        trace.append(
            {
                "C": c_value,
                "internal_validation_graph_AP": graph_ap,
                "internal_validation_pooled_AUPRC": pooled_ap,
            }
        )
        candidate = (pooled_ap, -c_value, model)
        if selected is None or candidate[:2] > selected[:2]:
            selected = candidate
    if selected is None:
        raise RuntimeError("No trainable fusion model was fitted.")
    return selected[2], pd.DataFrame(trace)


def graph_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    positives = int(labels.sum())
    if positives == 0 or labels.size == 0:
        return {key: np.nan for key in ("graph_AP", "IoU_at_P", "recall_at_10", "precision_at_10")}
    order = np.argsort(-scores, kind="stable")
    truth = set(np.flatnonzero(labels).tolist())
    selected = set(order[:positives].tolist())
    intersection = len(truth & selected)
    union = len(truth | selected)
    top = order[: min(10, labels.size)]
    recovered = float(labels[top].sum())
    return {
        "graph_AP": float(average_precision_score(labels, scores)),
        "IoU_at_P": float(intersection / max(union, 1)),
        "recall_at_10": float(recovered / positives),
        "precision_at_10": float(recovered / len(top)),
    }


def load_selected(path: Path) -> dict[str, set[str]]:
    frame = pd.read_csv(path, usecols=["sample_id", "split"])
    return {
        split: set(frame.loc[frame["split"] == split, "sample_id"].astype(str))
        for split in SPLITS
    }


def collect_external_rows(
    archives: list[np.lib.npyio.NpzFile], models: list[object], selected: dict[str, set[str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    residue_rows = []
    for split in SPLITS:
        first = archives[0]
        labels = first[f"{split}__labels"].astype(np.int8)
        groups = first[f"{split}__sample_index"].astype(np.int32)
        sample_ids = first[f"{split}__sample_ids"].astype(str)
        distances = distance_from_archive(first, split)
        scores = {
            "Distance": first[f"{split}__score__raw_distance"].astype(np.float64),
            "Fixed fusion": np.mean(
                [a[f"{split}__score__fixed_hybrid"].astype(np.float64) for a in archives],
                axis=0,
            ),
            "Trainable fusion": np.mean(
                [
                    model.predict_proba(two_input_features(archive, split))[:, 1]
                    for model, archive in zip(models, archives)
                ],
                axis=0,
            ),
            "LAIGN": np.mean(
                [a[f"{split}__score__trained_stage2"].astype(np.float64) for a in archives],
                axis=0,
            ),
        }
        for graph_index, sample_id in enumerate(sample_ids):
            if sample_id not in selected[split]:
                continue
            graph_mask = groups == graph_index
            for radius in (np.inf, 6.0, 8.0):
                mask = graph_mask & (distances <= radius)
                if not np.any(mask):
                    continue
                radius_name = "All residues" if np.isinf(radius) else f"Distance <= {int(radius)} A"
                for method in METHODS:
                    row = graph_metrics(labels[mask], scores[method][mask])
                    metric_rows.append(
                        {
                            "sample_id": sample_id,
                            "split": split,
                            "subset": radius_name,
                            "method": method,
                            "residues": int(mask.sum()),
                            "positives": int(labels[mask].sum()),
                            **row,
                        }
                    )
            indices = np.flatnonzero(graph_mask)
            for local_index, global_index in enumerate(indices):
                residue_rows.append(
                    {
                        "sample_id": sample_id,
                        "split": split,
                        "residue_index": local_index,
                        "label": int(labels[global_index]),
                        "distance": float(distances[global_index]),
                        **{method: float(scores[method][global_index]) for method in METHODS},
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(residue_rows)


def summarize_metrics(metrics: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    value_columns = ["graph_AP", "IoU_at_P", "recall_at_10", "precision_at_10"]
    rng = np.random.default_rng(seed)
    rows = []
    for (subset, method), part in metrics.groupby(["subset", "method"], sort=False):
        complexes = part[["split", "sample_id"]].drop_duplicates().shape[0]
        row = {
            "subset": subset,
            "method": method,
            "complexes": int(complexes),
            "mean_residues": float(part["residues"].mean()),
            "mean_positives": float(part["positives"].mean()),
        }
        for column in value_columns:
            values = part[column].dropna().to_numpy(dtype=float)
            draws = values[
                rng.integers(0, len(values), size=(bootstrap, len(values)))
            ].mean(axis=1)
            row[column] = float(values.mean())
            row[f"{column}_bootstrap_SD"] = float(draws.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_effects(
    metrics: pd.DataFrame, bootstrap: int, permutations: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    comparisons = ("Distance", "Fixed fusion", "Trainable fusion")
    rows = []
    boot_rows = []
    for subset in metrics["subset"].unique():
        part = metrics[metrics["subset"] == subset]
        pivot = part.pivot(
            index=["split", "sample_id"], columns="method", values="graph_AP"
        ).dropna()
        for comparison in comparisons:
            delta = (pivot["LAIGN"] - pivot[comparison]).to_numpy(dtype=float)
            draws = np.asarray(
                [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(bootstrap)]
            )
            signs = rng.choice(np.asarray([-1.0, 1.0]), size=(permutations, len(delta)))
            null = np.mean(signs * delta[None, :], axis=1)
            observed = float(delta.mean())
            p_value = float((1 + np.sum(np.abs(null) >= abs(observed))) / (permutations + 1))
            rows.append(
                {
                    "subset": subset,
                    "comparison": f"LAIGN - {comparison}",
                    "complexes": int(len(delta)),
                    "graph_AP_difference": observed,
                    "CI95_low": float(np.quantile(draws, 0.025)),
                    "CI95_high": float(np.quantile(draws, 0.975)),
                    "sign_flip_p": p_value,
                    "win_rate": float(np.mean(delta > 0)),
                }
            )
            boot_rows.extend(
                {
                    "subset": subset,
                    "comparison": f"LAIGN - {comparison}",
                    "iteration": index + 1,
                    "graph_AP_difference": float(value),
                }
                for index, value in enumerate(draws)
            )
    return pd.DataFrame(rows), pd.DataFrame(boot_rows)


def distance_matched_pairs(residues: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for radius in (6.0, 8.0):
        for caliper in (0.05, 0.10, 0.25, 0.50):
            for (split, sample_id), graph in residues.groupby(
                ["split", "sample_id"], sort=False
            ):
                graph = graph[graph["distance"] <= radius]
                positive = graph[graph["label"] == 1]
                negative = graph[graph["label"] == 0]
                if positive.empty or negative.empty:
                    continue
                costs = np.abs(
                    positive["distance"].to_numpy()[:, None]
                    - negative["distance"].to_numpy()[None, :]
                )
                p_indices, n_indices = linear_sum_assignment(costs)
                for p_index, n_index in zip(p_indices, n_indices):
                    gap = float(costs[p_index, n_index])
                    if gap > caliper:
                        continue
                    p = positive.iloc[p_index]
                    n = negative.iloc[n_index]
                    row = {
                        "sample_id": sample_id,
                        "split": split,
                        "radius": radius,
                        "caliper": caliper,
                        "distance_gap": gap,
                    }
                    for method in METHODS:
                        row[f"{method}_difference"] = float(p[method] - n[method])
                    rows.append(row)
    pairs = pd.DataFrame(rows)
    summary_rows = []
    if not pairs.empty:
        for (radius, caliper), part in pairs.groupby(["radius", "caliper"]):
            for method in METHODS:
                delta = part[f"{method}_difference"].to_numpy(dtype=float)
                summary_rows.append(
                    {
                        "radius": radius,
                        "caliper": caliper,
                        "method": method,
                        "complexes": int(
                            part[["split", "sample_id"]].drop_duplicates().shape[0]
                        ),
                        "matched_pairs": int(len(part)),
                        "mean_distance_gap": float(part["distance_gap"].mean()),
                        "pair_accuracy": float(np.mean((delta > 0) + 0.5 * (delta == 0))),
                        "mean_score_difference": float(delta.mean()),
                    }
                )
    return pairs, pd.DataFrame(summary_rows)


def distance_matched_effects(
    pairs: pd.DataFrame, bootstrap: int, permutations: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare pair discrimination with complexes as the resampling unit."""
    rng = np.random.default_rng(seed)
    rows = []
    boot_rows = []
    for (radius, caliper), part in pairs.groupby(["radius", "caliper"], sort=True):
        per_pair = part[["split", "sample_id"]].copy()
        for method in METHODS:
            delta = part[f"{method}_difference"].to_numpy(dtype=float)
            per_pair[method] = (delta > 0).astype(float) + 0.5 * (delta == 0)
        per_complex = per_pair.groupby(["split", "sample_id"], sort=False)[list(METHODS)].mean()
        for comparison in ("Distance", "Fixed fusion", "Trainable fusion"):
            delta = (per_complex["LAIGN"] - per_complex[comparison]).to_numpy(dtype=float)
            draws = np.asarray(
                [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(bootstrap)]
            )
            signs = rng.choice(np.asarray([-1.0, 1.0]), size=(permutations, len(delta)))
            null = np.mean(signs * delta[None, :], axis=1)
            observed = float(delta.mean())
            p_value = float((1 + np.sum(np.abs(null) >= abs(observed))) / (permutations + 1))
            comparison_name = f"LAIGN - {comparison}"
            rows.append(
                {
                    "radius": float(radius),
                    "caliper": float(caliper),
                    "comparison": comparison_name,
                    "complexes": int(len(delta)),
                    "complex_mean_pair_accuracy_difference": observed,
                    "CI95_low": float(np.quantile(draws, 0.025)),
                    "CI95_high": float(np.quantile(draws, 0.975)),
                    "sign_flip_p": p_value,
                    "win_rate": float(np.mean(delta > 0)),
                }
            )
            boot_rows.extend(
                {
                    "radius": float(radius),
                    "caliper": float(caliper),
                    "comparison": comparison_name,
                    "iteration": index + 1,
                    "complex_mean_pair_accuracy_difference": float(value),
                }
                for index, value in enumerate(draws)
            )
    return pd.DataFrame(rows), pd.DataFrame(boot_rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archives = [np.load(path, allow_pickle=False) for path in args.archives]
    validate_archives(archives, args.archives)
    selected = load_selected(args.subset)
    models = []
    traces = []
    for archive_index, archive in enumerate(archives):
        model, trace = fit_trainable_fusion(
            archive, args.c_grid, args.seed + archive_index
        )
        trace.insert(0, "archive", args.archives[archive_index].name)
        trace.insert(1, "run", archive_index + 1)
        models.append(model)
        traces.append(trace)
    trace = pd.concat(traces, ignore_index=True)
    metrics, residues = collect_external_rows(archives, models, selected)
    summary = summarize_metrics(metrics, bootstrap=1000, seed=args.seed)
    effects, boot = paired_effects(
        metrics, args.bootstrap, args.permutations, args.seed + 1
    )
    pairs, pair_summary = distance_matched_pairs(residues)
    pair_effects, pair_boot = distance_matched_effects(
        pairs, args.bootstrap, args.permutations, args.seed + 2
    )

    trace.to_csv(args.output_dir / "trainable_fusion_selection.csv", index=False)
    metrics.to_csv(args.output_dir / "plinder200_pocket_per_complex.csv", index=False)
    residues.to_csv(args.output_dir / "plinder200_pocket_residue_scores.csv.gz", index=False)
    summary.to_csv(args.output_dir / "plinder200_pocket_summary.csv", index=False)
    effects.to_csv(args.output_dir / "plinder200_pocket_paired_effects.csv", index=False)
    boot.to_csv(args.output_dir / "plinder200_pocket_bootstrap.csv.gz", index=False)
    pairs.to_csv(args.output_dir / "plinder200_distance_matched_pairs.csv.gz", index=False)
    pair_summary.to_csv(
        args.output_dir / "plinder200_distance_matched_pair_summary.csv", index=False
    )
    pair_effects.to_csv(
        args.output_dir / "plinder200_distance_matched_pair_effects.csv", index=False
    )
    pair_boot.to_csv(
        args.output_dir / "plinder200_distance_matched_pair_bootstrap.csv.gz", index=False
    )

    selected_rows = (
        trace.sort_values(
            ["run", "internal_validation_pooled_AUPRC", "C"],
            ascending=[True, False, True],
        )
        .groupby("run", as_index=False)
        .first()
    )
    payload = {
        "archives": [str(path) for path in args.archives],
        "selected_complexes": {split: len(ids) for split, ids in selected.items()},
        "trainable_fusion": {
            "inputs": ["raw interaction maximum", "exp(-nearest distance / 12 A)"],
            "classifier": "class-balanced L2 logistic regression",
            "selected_runs": selected_rows[
                ["archive", "run", "C", "internal_validation_graph_AP", "internal_validation_pooled_AUPRC"]
            ].to_dict(orient="records"),
        },
        "summary": summary.to_dict(orient="records"),
        "paired_effects": effects.to_dict(orient="records"),
        "distance_matched_summary": pair_summary.to_dict(orient="records"),
        "distance_matched_effects": pair_effects.to_dict(orient="records"),
    }
    (args.output_dir / "plinder200_pocket_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    report = [
        "# PLINDER-200 pocket-restricted and trainable-fusion analysis",
        "",
        "## Trainable fusion selection",
        "",
        dataframe_to_markdown(trace),
        "",
        "## PLINDER-200 summary",
        "",
        dataframe_to_markdown(summary),
        "",
        "## Paired graph-AP effects",
        "",
        dataframe_to_markdown(effects),
        "",
        "## Distance-matched residue pairs",
        "",
        dataframe_to_markdown(pair_summary),
        "",
        "## Distance-matched paired effects (complex-level resampling)",
        "",
        dataframe_to_markdown(pair_effects),
    ]
    (args.output_dir / "PLINDER200_POCKET_AND_TRAINABLE_FUSION_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "selected_C": [float(value) for value in selected_rows["C"]],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
