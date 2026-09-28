#!/usr/bin/env python3
"""Summarize the frozen Post-freeze temporal blind benchmark with paired statistics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[2]
METRICS = ("graph_ap", "iou_at_p", "recall_at_10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "outputs/postfreeze_holdout"
    parser.add_argument(
        "--archives",
        nargs=3,
        default=[str(base / f"replicate_{name}/predictions.npz") for name in "ABC"],
    )
    parser.add_argument(
        "--manifest", default=str(base / "postfreeze_split_manifest.csv.gz")
    )
    parser.add_argument(
        "--site-labels", default=str(base / "postfreeze_site_labels.csv.gz")
    )
    parser.add_argument(
        "--sequence-audit", default=str(base / "sequence_identity_audit.csv")
    )
    parser.add_argument(
        "--included", default=str(base / "included_complexes.csv")
    )
    parser.add_argument(
        "--fpocket",
        default=str(base / "baselines_geometry_full/predictions_fpocket.csv.gz"),
    )
    parser.add_argument(
        "--p2rank",
        default=str(base / "baselines_geometry_full/predictions_p2rank.csv.gz"),
    )
    parser.add_argument(
        "--prolif",
        default=str(base / "baseline_prolif_ccd_full/predictions_prolif.csv.gz"),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=8001)
    parser.add_argument("--out-json", default=str(base / "postfreeze_summary.json"))
    parser.add_argument("--per-complex", default=str(base / "postfreeze_per_complex_metrics.csv"))
    parser.add_argument("--stratified", default=str(base / "postfreeze_stratified_metrics.csv"))
    parser.add_argument(
        "--table", default=str(ROOT / "outputs/tables/table_postfreeze.md")
    )
    return parser.parse_args()


def graph_metrics(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if labels.size == 0 or positives == 0:
        return np.full(3, np.nan, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    selected = set(order[:positives].tolist())
    truth = set(np.flatnonzero(labels).tolist())
    overlap = len(selected & truth)
    union = len(selected | truth)
    return np.asarray(
        [
            average_precision_score(labels, scores),
            overlap / max(1, union),
            labels[order[: min(10, labels.size)]].sum() / positives,
        ],
        dtype=np.float64,
    )


def describe(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "n": int(values.size),
        "mean": float(values.mean()) if values.size else float("nan"),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
    }


def paired_test(delta: np.ndarray, draws: int, seed: int) -> dict[str, object]:
    delta = np.asarray(delta, dtype=np.float64)
    delta = delta[np.isfinite(delta)]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, delta.size, size=(draws, delta.size))
    bootstrap = delta[indices].mean(axis=1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, delta.size))
    null = (signs * delta[None, :]).mean(axis=1)
    observed = float(delta.mean())
    return {
        "n": int(delta.size),
        "mean_difference": observed,
        "ci_95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "sign_flip_p": float(
            (1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (draws + 1)
        ),
        "win_rate": float(np.mean(delta > 1e-12)),
        "tie_rate": float(np.mean(np.abs(delta) <= 1e-12)),
    }


def cluster_paired_test(
    delta: pd.Series, groups: pd.Series, draws: int, seed: int
) -> dict[str, object]:
    frame = pd.DataFrame({"delta": delta, "group": groups}).dropna()
    grouped = [
        values.to_numpy(dtype=np.float64)
        for _, values in frame.groupby("group", sort=True)["delta"]
    ]
    group_sums = np.asarray([values.sum() for values in grouped], dtype=np.float64)
    group_sizes = np.asarray([values.size for values in grouped], dtype=np.int32)
    n_groups = len(grouped)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n_groups, size=(draws, n_groups))
    bootstrap = group_sums[sampled].sum(axis=1) / group_sizes[sampled].sum(axis=1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, n_groups))
    null = (signs * group_sums[None, :]).sum(axis=1) / group_sizes.sum()
    observed = float(frame["delta"].mean())
    return {
        "complexes": int(len(frame)),
        "receptor_sequence_groups": int(n_groups),
        "mean_difference": observed,
        "ci_95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "cluster_sign_flip_p": float(
            (1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (draws + 1)
        ),
    }


def load_archives(paths: list[Path]) -> list[dict[str, np.ndarray]]:
    archives = []
    reference = None
    required = [
        "external_test__labels",
        "external_test__sample_index",
        "external_test__sample_ids",
        "external_test__score__raw_distance",
        "external_test__score__fixed_hybrid",
        "external_test__score__trained_stage2",
    ]
    for path in paths:
        loaded = np.load(path, allow_pickle=True)
        archive = {key: np.asarray(loaded[key]) for key in required}
        if reference is None:
            reference = archive
        else:
            for key in required[:3]:
                if not np.array_equal(reference[key], archive[key]):
                    raise ValueError(f"Archive order mismatch for {key}: {path}")
        archives.append(archive)
    return archives


def endpoint_labels(
    archive: dict[str, np.ndarray], manifest: pd.DataFrame, site: pd.DataFrame
) -> dict[str, np.ndarray]:
    sample_ids = archive["external_test__sample_ids"].astype(str)
    sample_index = archive["external_test__sample_index"].astype(np.int32)
    active = archive["external_test__labels"].astype(np.int32)
    system_by_sample = dict(
        zip(
            manifest.loc[manifest["supervision_split"] == "external_test", "sample_id"].astype(str),
            manifest.loc[manifest["supervision_split"] == "external_test", "system_id"].astype(str),
        )
    )
    site_by_system = {str(key): frame for key, frame in site.groupby("system_id", sort=False)}
    contact = np.empty_like(active)
    for index, sample_id in enumerate(sample_ids):
        mask = sample_index == index
        system_id = system_by_sample[sample_id]
        rows = site_by_system[system_id]
        expected = rows["has_plip_interaction"].to_numpy(dtype=np.int32)
        if not np.array_equal(active[mask], expected):
            raise ValueError(f"Residue-label order mismatch for {sample_id}")
        contact[mask] = rows["site_label"].to_numpy(dtype=np.int32)
    return {"plip_active": active, "contact_5A": contact}


def archive_rows(
    archive: dict[str, np.ndarray], labels: np.ndarray, score_key: str
) -> pd.DataFrame:
    sample_ids = archive["external_test__sample_ids"].astype(str)
    sample_index = archive["external_test__sample_index"].astype(np.int32)
    scores = archive[score_key].astype(np.float64)
    rows = []
    for index, sample_id in enumerate(sample_ids):
        mask = sample_index == index
        values = graph_metrics(labels[mask], scores[mask])
        rows.append({"sample_id": sample_id, **dict(zip(METRICS, values))})
    return pd.DataFrame(rows).set_index("sample_id")


def baseline_rows(
    path: Path,
    endpoint: str,
    manifest: pd.DataFrame,
    site: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame["sample_id"] = frame["sample_id"].astype(str)
    if endpoint == "plip_active":
        label_column = "label"
    else:
        system_map = dict(zip(manifest["sample_id"].astype(str), manifest["system_id"].astype(str)))
        frame["system_id"] = frame["sample_id"].map(system_map)
        lookup = site[
            [
                "system_id",
                "protein_chain",
                "protein_residue_number",
                "protein_residue_type",
                "site_label",
            ]
        ].rename(
            columns={
                "protein_chain": "chain",
                "protein_residue_number": "residue_number",
                "protein_residue_type": "residue_type",
            }
        )
        frame = frame.merge(
            lookup,
            on=["system_id", "chain", "residue_number", "residue_type"],
            how="left",
            validate="one_to_one",
        )
        if frame["site_label"].isna().any():
            raise ValueError(f"Missing 5A labels after joining {path}")
        label_column = "site_label"
    rows = []
    for sample_id, group in frame.groupby("sample_id", sort=False):
        values = graph_metrics(
            group[label_column].to_numpy(dtype=np.int32),
            group["score"].to_numpy(dtype=np.float64),
        )
        rows.append({"sample_id": sample_id, **dict(zip(METRICS, values))})
    return pd.DataFrame(rows).set_index("sample_id")


def add_metadata(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    sequence: pd.DataFrame,
    included: pd.DataFrame,
) -> pd.DataFrame:
    external = manifest[manifest["supervision_split"] == "external_test"].copy()
    training_ligands = set(
        manifest.loc[manifest["supervision_split"] == "train", "ligand_ccd_code"]
        .dropna()
        .astype(str)
        .str.upper()
    )
    external["ligand_ccd_code"] = external["ligand_ccd_code"].astype(str).str.upper()
    external["ligand_status"] = np.where(
        external["ligand_ccd_code"].isin(training_ligands), "seen_CCD", "unseen_CCD"
    )
    meta = external[["sample_id", "system_id", "pdb_id", "ligand_ccd_code", "ligand_status"]]
    meta = meta.merge(sequence[["system_id", "max_train_identity_percent"]], on="system_id")
    meta = meta.merge(
        included[
            [
                "pdb_id",
                "resolution",
                "ligand_heavy_atoms",
                "receptor_residues",
                "active_interaction_residues",
            ]
        ],
        on="pdb_id",
        validate="many_to_one",
    )
    meta["identity_group"] = pd.cut(
        meta["max_train_identity_percent"],
        bins=[-np.inf, 70.0, 95.0, np.inf],
        labels=["identity_lt70", "identity_70_to_95", "identity_ge95"],
        right=False,
    ).astype(str)
    meta["resolution_group"] = pd.cut(
        meta["resolution"],
        bins=[-np.inf, 2.0, 2.5, np.inf],
        labels=["resolution_le2", "resolution_2_to_2.5", "resolution_gt2.5"],
        right=True,
    ).astype(str)
    meta["ligand_size_group"] = pd.cut(
        meta["ligand_heavy_atoms"],
        bins=[-np.inf, 20, 40, np.inf],
        labels=["ligand_le20", "ligand_21_to_40", "ligand_gt40"],
        right=True,
    ).astype(str)
    meta["receptor_size_group"] = pd.cut(
        meta["receptor_residues"],
        bins=[-np.inf, 250, 500, np.inf],
        labels=["receptor_le250", "receptor_251_to_500", "receptor_gt500"],
        right=True,
    ).astype(str)
    return meta.set_index("sample_id").join(frame, how="inner")


def main() -> int:
    args = parse_args()
    archive_paths = [Path(path) for path in args.archives]
    archives = load_archives(archive_paths)
    manifest = pd.read_csv(args.manifest, low_memory=False)
    site = pd.read_csv(args.site_labels, low_memory=False)
    sequence = pd.read_csv(args.sequence_audit)
    included = pd.read_csv(args.included)
    endpoints = endpoint_labels(archives[0], manifest, site)

    all_rows: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)
    method_specs = {
        "raw_distance": "external_test__score__raw_distance",
        "fixed_hybrid": "external_test__score__fixed_hybrid",
        "LAIGN_A": "external_test__score__trained_stage2",
    }
    for endpoint, labels in endpoints.items():
        for method, key in method_specs.items():
            all_rows[endpoint][method] = archive_rows(archives[0], labels, key)
        for replicate, archive in zip("ABC", archives):
            all_rows[endpoint][f"LAIGN_{replicate}"] = archive_rows(
                archive, labels, "external_test__score__trained_stage2"
            )
            all_rows[endpoint][f"fixed_hybrid_{replicate}"] = archive_rows(
                archive, labels, "external_test__score__fixed_hybrid"
            )
        for method, path in {
            "fpocket": args.fpocket,
            "P2Rank": args.p2rank,
            "ProLIF": args.prolif,
        }.items():
            all_rows[endpoint][method] = baseline_rows(Path(path), endpoint, manifest, site)

    reference_ids = all_rows["plip_active"]["LAIGN_A"].index
    for endpoint, methods in all_rows.items():
        for method, rows in methods.items():
            if set(rows.index) != set(reference_ids):
                raise ValueError(f"Coverage mismatch for {endpoint}:{method}")
            methods[method] = rows.loc[reference_ids]

    summaries = {}
    paired = {}
    paired_clustered = {}
    external_manifest = manifest[manifest["supervision_split"] == "external_test"]
    receptor_groups = pd.Series(
        external_manifest["receptor_sequence"].astype(str).to_numpy(),
        index=external_manifest["sample_id"].astype(str),
    ).loc[reference_ids]
    for endpoint, methods in all_rows.items():
        summaries[endpoint] = {
            method: {metric: describe(rows[metric].to_numpy()) for metric in METRICS}
            for method, rows in methods.items()
        }
        paired[endpoint] = {}
        paired_clustered[endpoint] = {}
        for method in ("raw_distance", "fixed_hybrid", "fpocket", "P2Rank", "ProLIF"):
            paired[endpoint][method] = {
                metric: paired_test(
                    methods["LAIGN_A"][metric].to_numpy() - methods[method][metric].to_numpy(),
                    args.bootstrap_draws,
                    args.seed + 100 * list(all_rows).index(endpoint) + 10 * list(methods).index(method) + index,
                )
                for index, metric in enumerate(METRICS)
            }
            paired_clustered[endpoint][method] = {
                metric: cluster_paired_test(
                    methods["LAIGN_A"][metric] - methods[method][metric],
                    receptor_groups,
                    args.bootstrap_draws,
                    args.seed + 5000 + 100 * list(all_rows).index(endpoint) + 10 * list(methods).index(method) + index,
                )
                for index, metric in enumerate(METRICS)
            }

    replicate_summary = {}
    matched_replicate_deltas = {}
    for endpoint in endpoints:
        replicate_summary[endpoint] = {"LAIGN": {}, "fixed_hybrid": {}, "paired_delta": {}}
        matched_replicate_deltas[endpoint] = {}
        for metric in METRICS:
            laign_values = np.asarray(
                [summaries[endpoint][f"LAIGN_{rep}"][metric]["mean"] for rep in "ABC"]
            )
            control_values = np.asarray(
                [summaries[endpoint][f"fixed_hybrid_{rep}"][metric]["mean"] for rep in "ABC"]
            )
            deltas = laign_values - control_values
            replicate_summary[endpoint]["LAIGN"][metric] = describe(laign_values)
            replicate_summary[endpoint]["fixed_hybrid"][metric] = describe(control_values)
            replicate_summary[endpoint]["paired_delta"][metric] = describe(deltas)
            matched_replicate_deltas[endpoint][metric] = {
                replicate: float(value) for replicate, value in zip("ABC", deltas)
            }

    per_complex = all_rows["plip_active"]["LAIGN_A"].add_prefix("LAIGN_A_")
    for method in (
        "LAIGN_B",
        "LAIGN_C",
        "fixed_hybrid",
        "fixed_hybrid_B",
        "fixed_hybrid_C",
        "fpocket",
        "P2Rank",
        "ProLIF",
    ):
        per_complex = per_complex.join(all_rows["plip_active"][method].add_prefix(f"{method}_"))
    enriched = add_metadata(per_complex, manifest, sequence, included)
    Path(args.per_complex).parent.mkdir(parents=True, exist_ok=True)
    enriched.reset_index().to_csv(args.per_complex, index=False)

    stratified_rows = []
    for dimension in (
        "identity_group",
        "ligand_status",
        "resolution_group",
        "ligand_size_group",
        "receptor_size_group",
    ):
        for group, rows in enriched.groupby(dimension, observed=True):
            entry = {"dimension": dimension, "group": str(group), "graphs": int(len(rows))}
            for method in (
                "LAIGN_A",
                "LAIGN_B",
                "LAIGN_C",
                "fixed_hybrid",
                "fixed_hybrid_B",
                "fixed_hybrid_C",
                "P2Rank",
                "ProLIF",
            ):
                for metric in METRICS:
                    entry[f"{method}_{metric}"] = float(rows[f"{method}_{metric}"].mean())
            stratified_rows.append(entry)
    stratified = pd.DataFrame(stratified_rows)
    stratified.to_csv(args.stratified, index=False)

    stratified_complete_replicates = []
    for dimension in (
        "identity_group",
        "ligand_status",
        "resolution_group",
        "ligand_size_group",
        "receptor_size_group",
    ):
        for group, rows in enriched.groupby(dimension, observed=True):
            laign_values = np.asarray(
                [rows[f"LAIGN_{replicate}_graph_ap"].mean() for replicate in "ABC"]
            )
            control_values = np.asarray(
                [
                    rows[
                        "fixed_hybrid_graph_ap"
                        if replicate == "A"
                        else f"fixed_hybrid_{replicate}_graph_ap"
                    ].mean()
                    for replicate in "ABC"
                ]
            )
            stratified_complete_replicates.append(
                {
                    "dimension": dimension,
                    "group": str(group),
                    "graphs": int(len(rows)),
                    "LAIGN_graph_ap": describe(laign_values),
                    "fixed_hybrid_graph_ap": describe(control_values),
                    "paired_delta_graph_ap": describe(laign_values - control_values),
                    "per_replicate_delta": {
                        replicate: float(value)
                        for replicate, value in zip("ABC", laign_values - control_values)
                    },
                }
            )

    exploratory_stratified_tests = []
    dimensions = (
        "identity_group",
        "ligand_status",
        "resolution_group",
        "ligand_size_group",
        "receptor_size_group",
    )
    test_index = 0
    for dimension in dimensions:
        for group, rows in enriched.groupby(dimension, observed=True):
            for comparator in ("fixed_hybrid", "P2Rank", "ProLIF"):
                result = paired_test(
                    rows["LAIGN_A_graph_ap"].to_numpy()
                    - rows[f"{comparator}_graph_ap"].to_numpy(),
                    args.bootstrap_draws,
                    args.seed + 10000 + test_index,
                )
                exploratory_stratified_tests.append(
                    {
                        "dimension": dimension,
                        "group": str(group),
                        "comparator": comparator,
                        **result,
                    }
                )
                test_index += 1

    failure_columns = [
        "system_id",
        "pdb_id",
        "ligand_ccd_code",
        "ligand_status",
        "max_train_identity_percent",
        "resolution",
        "ligand_heavy_atoms",
        "receptor_residues",
        "active_interaction_residues",
        "LAIGN_A_graph_ap",
        "fixed_hybrid_graph_ap",
        "ProLIF_graph_ap",
        "P2Rank_graph_ap",
    ]
    failures = enriched.copy()
    failures["delta_LAIGN_minus_fixed_graph_ap"] = (
        failures["LAIGN_A_graph_ap"] - failures["fixed_hybrid_graph_ap"]
    )
    failures = failures.sort_values("delta_LAIGN_minus_fixed_graph_ap").head(10)
    failure_path = Path(args.per_complex).with_name("postfreeze_largest_fixed_hybrid_losses.csv")
    failures.reset_index()[
        ["sample_id", *failure_columns, "delta_LAIGN_minus_fixed_graph_ap"]
    ].to_csv(failure_path, index=False)

    payload = {
        "hypothesis": "Post-freeze",
        "protocol": "frozen post-2026-07-08 RCSB temporal blind test",
        "graphs": int(len(reference_ids)),
        "archives": [str(path.relative_to(ROOT)) for path in archive_paths],
        "endpoints": {
            "plip_active": "five active PLIP interaction residue types",
            "contact_5A": "residue has any protein atom within 5 A of the bound ligand",
        },
        "summaries": summaries,
        "complete_pipeline_replicates": replicate_summary,
        "matched_complete_pipeline_deltas": matched_replicate_deltas,
        "paired_LAIGN_A_minus_baseline": paired,
        "receptor_clustered_paired_LAIGN_A_minus_baseline": paired_clustered,
        "exploratory_stratified_graph_AP_tests": exploratory_stratified_tests,
        "stratified_complete_pipeline_replicates": stratified_complete_replicates,
        "artifacts": {
            "per_complex": str(Path(args.per_complex).relative_to(ROOT)),
            "stratified": str(Path(args.stratified).relative_to(ROOT)),
            "largest_fixed_hybrid_losses": str(failure_path.relative_to(ROOT)),
        },
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Post-freeze Frozen Temporal Blind Benchmark",
        "",
        "| endpoint | method | graph AP | IoU@P | recall@10 |",
        "|---|---|---:|---:|---:|",
    ]
    order = ("LAIGN_A", "LAIGN_B", "LAIGN_C", "fixed_hybrid", "raw_distance", "ProLIF", "P2Rank", "fpocket")
    for endpoint, methods in summaries.items():
        for method in order:
            row = methods[method]
            lines.append(
                f"| {endpoint} | {method} | {row['graph_ap']['mean']:.4f} | "
                f"{row['iou_at_p']['mean']:.4f} | {row['recall_at_10']['mean']:.4f} |"
            )
    lines.extend(["", "## Complete-pipeline replication", ""])
    lines.append("| endpoint | quantity | metric | mean +/- SD across matched A/B/C |")
    lines.append("|---|---|---|---:|")
    for endpoint, quantity_rows in replicate_summary.items():
        for quantity, metric_rows in quantity_rows.items():
            for metric, row in metric_rows.items():
                lines.append(
                    f"| {endpoint} | {quantity} | {metric} | "
                    f"{row['mean']:.4f} +/- {row['std']:.4f} |"
                )
    lines.extend(["", "## Paired primary-endpoint tests", ""])
    lines.append("| baseline | metric | delta | 95% CI | sign-flip p | win rate |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for method, metric_rows in paired["plip_active"].items():
        for metric, row in metric_rows.items():
            low, high = row["ci_95"]
            lines.append(
                f"| {method} | {metric} | {row['mean_difference']:+.4f} | "
                f"[{low:+.4f}, {high:+.4f}] | {row['sign_flip_p']:.4g} | {row['win_rate']:.3f} |"
            )
    lines.extend(["", "## Exact-receptor-sequence clustered tests", ""])
    lines.append("| baseline | metric | groups | delta | cluster 95% CI | cluster sign-flip p |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for method, metric_rows in paired_clustered["plip_active"].items():
        for metric, row in metric_rows.items():
            low, high = row["ci_95"]
            lines.append(
                f"| {method} | {metric} | {row['receptor_sequence_groups']} | "
                f"{row['mean_difference']:+.4f} | [{low:+.4f}, {high:+.4f}] | "
                f"{row['cluster_sign_flip_p']:.4g} |"
            )
    table = Path(args.table)
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(out_json), "table": str(table), "graphs": len(reference_ids)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
