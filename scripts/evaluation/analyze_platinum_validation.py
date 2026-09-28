#!/usr/bin/env python3
"""Analyze LAIGN scores against independent Platinum mutation-affinity endpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.data import build_interaction_graph_dataset as graph_builder  # noqa: E402
from scripts.data import build_interaction_pair_features as pair_features  # noqa: E402


ACTIVE_INTERACTIONS = {
    "hydrophobic_contact",
    "hydrogen_bond",
    "salt_bridge",
    "pi_stacking",
    "pi_cation",
}
THRESHOLDS = (0.5, 1.0, 2.0)
METHODS = {
    "LAIGN": "trained_stage2",
    "fixed fusion": "fixed_hybrid",
    "distance": "raw_distance",
    "raw interaction": "raw_visnet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["frozen", "purged"])
    parser.add_argument("--archives", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=8231)
    return parser.parse_args()


def clean_residue_number(value: object) -> str:
    return pair_features.clean_resnr(value)


def candidate_keys(row: dict[str, object]) -> list[tuple[str, str, str]]:
    _, residue_atoms = pair_features.parse_pdb_atoms(str(row["plip_pdb"]))
    keys = sorted(
        key
        for key in residue_atoms
        if key[2] in pair_features.AA3
    )
    kept = []
    occupied = set()
    for key in keys:
        atoms = graph_builder.get_residue_atoms(residue_atoms, key)
        if not atoms:
            continue
        centroid = np.stack([atom["xyz"] for atom in atoms], axis=0).mean(axis=0).astype(np.float32)
        position = tuple(centroid.tolist())
        if position in occupied:
            continue
        occupied.add(position)
        kept.append(
            (
                pair_features.clean_str(key[0]),
                clean_residue_number(key[1]),
                pair_features.clean_str(key[2]).upper(),
            )
        )
    return kept


def load_structural_counts(base: Path) -> tuple[dict[tuple[str, str, str, str], int], dict[tuple[str, str, str, str], float]]:
    manifest = pd.read_csv(base / "platinum_manifest.csv.gz", low_memory=False)
    sample_by_system = {
        (str(row["system_id"]), str(row["ligand_id"])): str(row["sample_id"])
        for row in manifest.to_dict("records")
    }
    interactions = pd.read_csv(base / "platinum_plip_interactions.csv.gz", low_memory=False)
    interactions = interactions[interactions["interaction_type"].isin(ACTIVE_INTERACTIONS)].copy()
    plip_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for row in interactions.to_dict("records"):
        key = (
            sample_by_system[(str(row["system_id"]), str(row["ligand_id"]))],
            pair_features.clean_str(row["protein_chain"]),
            clean_residue_number(row["protein_residue_number"]),
            pair_features.clean_str(row["protein_residue_type"]).upper(),
        )
        plip_counts[key] += 1

    prolif_path = ROOT / "experiments/validation/results/platinum_prolif/platinum_prolif_counts.csv.gz"
    prolif = pd.read_csv(prolif_path, low_memory=False)
    prolif_counts = {}
    for row in prolif.to_dict("records"):
        key = (
            str(row["sample_id"]),
            pair_features.clean_str(row["protein_chain"]),
            clean_residue_number(row["protein_residue_number"]),
            pair_features.clean_str(row["protein_residue_type"]).upper(),
        )
        prolif_counts[key] = float(row["prolif_count"])
    return plip_counts, prolif_counts


def load_candidate_scores(
    archives: list[str], manifest: pd.DataFrame, base: Path
) -> pd.DataFrame:
    manifest_by_id = {
        str(row["sample_id"]): row for row in manifest.to_dict("records")
    }
    plip_counts, prolif_counts = load_structural_counts(base)
    rows = []
    for run_index, archive in enumerate(archives):
        path = Path(archive)
        z = np.load(path, allow_pickle=True)
        sample_ids = [str(value) for value in z["external_test__sample_ids"]]
        sample_index = z["external_test__sample_index"].astype(np.int32)
        labels = z["external_test__labels"].astype(np.int32)
        score_arrays = {
            name: z[f"external_test__score__{archive_name}"].astype(np.float64)
            for name, archive_name in METHODS.items()
        }
        for graph_index, sample_id in enumerate(sample_ids):
            row = manifest_by_id[sample_id]
            keys = candidate_keys(row)
            idx = np.flatnonzero(sample_index == graph_index)
            if len(keys) != idx.size:
                raise RuntimeError(
                    f"candidate count mismatch for {sample_id}: keys={len(keys)}, archive={idx.size}"
                )
            for local_index, (chain, number, residue_type) in enumerate(keys):
                identity = (sample_id, chain, number, residue_type)
                item = {
                    "stage": "",
                    "run": run_index,
                    "sample_id": sample_id,
                    "protein_chain": chain,
                    "protein_residue_number": number,
                    "protein_residue_type": residue_type,
                    "plip_label": int(labels[idx[local_index]]),
                    "PLIP count": float(plip_counts.get(identity, 0)),
                    "ProLIF count": float(prolif_counts.get(identity, 0.0)),
                }
                for method, values in score_arrays.items():
                    item[method] = float(values[idx[local_index]])
                if int(item["PLIP count"] > 0) != item["plip_label"]:
                    raise RuntimeError(f"PLIP alignment mismatch for {identity}")
                rows.append(item)
    return pd.DataFrame(rows)


def load_site_table(base: Path) -> pd.DataFrame:
    mutations = pd.read_csv(base / "platinum_mutation_records.csv", low_memory=False)
    mutations["protein_residue_number"] = mutations["protein_residue_number"].map(clean_residue_number)
    group_columns = [
        "sample_id",
        "protein_chain",
        "protein_residue_number",
        "protein_residue_type",
        "wt_pdb_id",
        "ligand_ccd_code",
        "uniprot_id",
    ]
    sites = (
        mutations.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            mutation_count=("mutation", "size"),
            mutations=("mutation", lambda values: ";".join(sorted(set(map(str, values))))),
            max_ddg=("ddg_kcal_mol_298K", "max"),
            mean_ddg=("ddg_kcal_mol_298K", "mean"),
            min_ligand_distance=("min_mutation_ligand_distance", "min"),
            pmids=("pmid", lambda values: ";".join(sorted(set(map(str, values))))),
        )
    )
    for threshold in THRESHOLDS:
        sites[f"disruptive_ge_{threshold:g}"] = (sites["max_ddg"] >= threshold).astype(int)

    overlap = pd.read_csv(base / "platinum_exact_overlap_audit.csv", low_memory=False)
    overlap = overlap.drop_duplicates("sample_id")
    identity = pd.read_csv(base / "platinum_sequence_identity.csv", low_memory=False)
    identity = identity.drop_duplicates("sample_id")
    sites = sites.merge(
        overlap[
            ["sample_id", "exact_train_pdb_overlap", "exact_train_uniprot_overlap"]
        ],
        on="sample_id",
        how="left",
        validate="many_to_one",
    ).merge(
        identity[["sample_id", "max_train_identity_percent"]],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    raw = pd.read_csv(
        ROOT / "experiments/validation/data/raw/platinum/platinum_flat_file.csv",
        low_memory=False,
    ).reset_index(names="source_row")
    metadata = raw[
        [
            "source_row",
            "prot.molecule_name",
            "prot.organism",
            "prot.protein_class",
            "affin.lig_name",
            "lig.mol_weight",
            "lig.num_rings",
        ]
    ].copy()
    mutation_source = pd.read_csv(base / "platinum_mutation_records.csv", low_memory=False)[
        ["sample_id", "protein_chain", "protein_residue_number", "protein_residue_type", "source_row"]
    ]
    mutation_source["protein_residue_number"] = mutation_source["protein_residue_number"].map(clean_residue_number)
    mutation_source = mutation_source.sort_values("source_row").drop_duplicates(
        ["sample_id", "protein_chain", "protein_residue_number", "protein_residue_type"]
    )
    mutation_source = mutation_source.merge(metadata, on="source_row", how="left")
    sites = sites.merge(
        mutation_source.drop(columns="source_row"),
        on=["sample_id", "protein_chain", "protein_residue_number", "protein_residue_type"],
        how="left",
        validate="one_to_one",
    )
    return sites


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else math.nan


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else math.nan


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(spearmanr(x, y).statistic)


def subset_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    identity = frame["max_train_identity_percent"].fillna(-1)
    return {
        "all": pd.Series(True, index=frame.index),
        "no exact PDB": ~frame["exact_train_pdb_overlap"].fillna(False),
        "no exact UniProt": ~frame["exact_train_uniprot_overlap"].fillna(False),
        "sequence identity <70%": identity.lt(70),
        "sequence identity <30%": identity.lt(30),
    }


def classification_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    methods = [*METHODS, "PLIP count", "ProLIF count"]
    for run, run_frame in frame.groupby("run", sort=True):
        for subset, mask in subset_masks(run_frame).items():
            selected = run_frame[mask]
            for threshold in THRESHOLDS:
                label = f"disruptive_ge_{threshold:g}"
                y = selected[label].to_numpy(dtype=np.int32)
                for method in methods:
                    score = selected[method].to_numpy(dtype=np.float64)
                    graph_ap = []
                    graph_auc = []
                    graph_rho = []
                    for _, group in selected.groupby("sample_id", sort=False):
                        gy = group[label].to_numpy(dtype=np.int32)
                        gs = group[method].to_numpy(dtype=np.float64)
                        graph_ap.append(safe_ap(gy, gs))
                        graph_auc.append(safe_auc(gy, gs))
                        graph_rho.append(
                            safe_spearman(
                                gs, group["max_ddg"].to_numpy(dtype=np.float64)
                            )
                        )
                    rows.append(
                        {
                            "run": int(run),
                            "subset": subset,
                            "threshold_kcal_mol": threshold,
                            "method": method,
                            "complexes": int(selected["sample_id"].nunique()),
                            "sites": int(len(selected)),
                            "positives": int(y.sum()),
                            "prevalence": float(y.mean()) if y.size else math.nan,
                            "site_auprc": safe_ap(y, score),
                            "site_auroc": safe_auc(y, score),
                            "spearman_ddg": safe_spearman(
                                score, selected["max_ddg"].to_numpy(dtype=np.float64)
                            ),
                            "macro_complex_auprc": float(np.nanmean(graph_ap))
                            if np.isfinite(graph_ap).any()
                            else math.nan,
                            "macro_complex_auroc": float(np.nanmean(graph_auc))
                            if np.isfinite(graph_auc).any()
                            else math.nan,
                            "macro_complex_spearman": float(np.nanmean(graph_rho))
                            if np.isfinite(graph_rho).any()
                            else math.nan,
                        }
                    )
    return pd.DataFrame(rows)


def topk_metrics(site_scores: pd.DataFrame, candidate_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    methods = [*METHODS, "PLIP count", "ProLIF count"]
    key_columns = [
        "sample_id",
        "protein_chain",
        "protein_residue_number",
        "protein_residue_type",
    ]
    for run, candidates in candidate_scores.groupby("run", sort=True):
        sites = site_scores[site_scores["run"].eq(run)]
        for threshold in THRESHOLDS:
            positive = sites[sites[f"disruptive_ge_{threshold:g}"].eq(1)]
            truth = {
                sample_id: set(map(tuple, group[key_columns[1:]].to_numpy()))
                for sample_id, group in positive.groupby("sample_id")
            }
            for method in methods:
                recalls = {5: [], 10: []}
                hits = {5: [], 10: []}
                for sample_id, keys in truth.items():
                    group = candidates[candidates["sample_id"].eq(sample_id)].copy()
                    group = group.sort_values(
                        [method, "protein_chain", "protein_residue_number", "protein_residue_type"],
                        ascending=[False, True, True, True],
                        kind="mergesort",
                    )
                    ranked = list(map(tuple, group[key_columns[1:]].to_numpy()))
                    for k in (5, 10):
                        recovered = len(keys.intersection(ranked[:k]))
                        recalls[k].append(recovered / len(keys))
                        hits[k].append(float(recovered > 0))
                rows.append(
                    {
                        "run": int(run),
                        "threshold_kcal_mol": threshold,
                        "method": method,
                        "complexes_with_disruptive_site": len(truth),
                        "recall_at_5_all_residues": float(np.mean(recalls[5])) if recalls[5] else math.nan,
                        "recall_at_10_all_residues": float(np.mean(recalls[10])) if recalls[10] else math.nan,
                        "hit_at_5_all_residues": float(np.mean(hits[5])) if hits[5] else math.nan,
                        "hit_at_10_all_residues": float(np.mean(hits[10])) if hits[10] else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_comparisons(
    averaged_sites: pd.DataFrame, n_bootstrap: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = "disruptive_ge_1"
    groups = {
        sample_id: group.copy()
        for sample_id, group in averaged_sites.groupby("sample_id", sort=False)
    }
    sample_ids = np.asarray(sorted(groups), dtype=object)
    methods = ["fixed fusion", "distance", "PLIP count", "ProLIF count"]
    observed = {}
    y = averaged_sites[label].to_numpy(dtype=np.int32)
    laign = averaged_sites["LAIGN"].to_numpy(dtype=np.float64)
    for method in methods:
        observed[method] = safe_ap(y, laign) - safe_ap(
            y, averaged_sites[method].to_numpy(dtype=np.float64)
        )
    values = {method: [] for method in methods}
    for _ in range(n_bootstrap):
        sampled = rng.choice(sample_ids, size=len(sample_ids), replace=True)
        frame = pd.concat([groups[sample_id] for sample_id in sampled], ignore_index=True)
        by = frame[label].to_numpy(dtype=np.int32)
        bs = frame["LAIGN"].to_numpy(dtype=np.float64)
        if len(np.unique(by)) < 2:
            continue
        laign_ap = safe_ap(by, bs)
        for method in methods:
            values[method].append(
                laign_ap - safe_ap(by, frame[method].to_numpy(dtype=np.float64))
            )
    rows = []
    for method in methods:
        delta = np.asarray(values[method], dtype=np.float64)
        rows.append(
            {
                "comparator": method,
                "observed_delta_auprc": observed[method],
                "ci95_low": float(np.quantile(delta, 0.025)),
                "ci95_high": float(np.quantile(delta, 0.975)),
                "bootstrap_replicates": int(delta.size),
                "one_sided_p_delta_le_0": float((np.sum(delta <= 0) + 1) / (delta.size + 1)),
            }
        )
    return pd.DataFrame(rows)


def budget_bootstrap(
    averaged_sites: pd.DataFrame,
    candidate_average: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    key_columns = [
        "sample_id",
        "protein_chain",
        "protein_residue_number",
        "protein_residue_type",
    ]
    positive = averaged_sites[averaged_sites["disruptive_ge_1"].eq(1)]
    truth = {
        sample_id: set(map(tuple, group[key_columns[1:]].astype(str).to_numpy()))
        for sample_id, group in positive.groupby("sample_id")
    }
    methods = ["LAIGN", "fixed fusion", "distance", "PLIP count", "ProLIF count"]
    per_complex: dict[tuple[int, str], list[float]] = defaultdict(list)
    for sample_id, true_keys in truth.items():
        group = candidate_average[candidate_average["sample_id"].eq(sample_id)]
        for method in methods:
            ranked = list(
                map(
                    tuple,
                    group.sort_values(
                        [method, *key_columns[1:]],
                        ascending=[False, True, True, True],
                        kind="mergesort",
                    )[key_columns[1:]]
                    .astype(str)
                    .to_numpy(),
                )
            )
            for k in (5, 10):
                recovered = len(true_keys.intersection(ranked[:k]))
                per_complex[(k, method)].append(recovered / len(true_keys))
    rng = np.random.default_rng(seed + 91)
    rows = []
    for k in (5, 10):
        laign = np.asarray(per_complex[(k, "LAIGN")], dtype=np.float64)
        for comparator in methods[1:]:
            other = np.asarray(per_complex[(k, comparator)], dtype=np.float64)
            delta = laign - other
            bootstrap_mean = np.mean(
                rng.choice(delta, size=(n_bootstrap, delta.size), replace=True), axis=1
            )
            rows.append(
                {
                    "budget_k": k,
                    "comparator": comparator,
                    "complexes": int(delta.size),
                    "observed_delta_recall": float(delta.mean()),
                    "ci95_low": float(np.quantile(bootstrap_mean, 0.025)),
                    "ci95_high": float(np.quantile(bootstrap_mean, 0.975)),
                    "one_sided_p_delta_le_0": float(
                        (np.sum(bootstrap_mean <= 0) + 1) / (n_bootstrap + 1)
                    ),
                }
            )
    return pd.DataFrame(rows)


def averaged_site_scores(site_scores: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "sample_id",
        "protein_chain",
        "protein_residue_number",
        "protein_residue_type",
    ]
    method_columns = [*METHODS, "PLIP count", "ProLIF count"]
    scores = site_scores.groupby(keys, as_index=False)[method_columns].mean()
    metadata = site_scores.drop(columns=[*method_columns, "run"]).drop_duplicates(keys)
    return metadata.merge(scores, on=keys, validate="one_to_one")


def select_cases(averaged: pd.DataFrame, candidate_average: pd.DataFrame) -> pd.DataFrame:
    frame = averaged.copy()
    frame["laign_rank_assayed"] = frame.groupby("sample_id")["LAIGN"].rank(
        method="min", ascending=False
    )
    frame["distance_rank_assayed"] = frame.groupby("sample_id")["distance"].rank(
        method="min", ascending=False
    )
    frame["laign_percentile_assayed"] = frame.groupby("sample_id")["LAIGN"].rank(
        pct=True, ascending=True
    )
    positive = frame[frame["disruptive_ge_1"].eq(1)].copy()
    positive["success_margin"] = positive["distance_rank_assayed"] - positive["laign_rank_assayed"]

    selected: list[tuple[str, pd.Series]] = []
    cofactor_codes = {
        "ATP", "ADP", "AMP", "FAD", "FMN", "NAD", "NAP", "NAI", "COA", "SAM", "SAH", "HEM"
    }
    cofactor = positive[positive["ligand_ccd_code"].isin(cofactor_codes)].sort_values(
        ["success_margin", "max_ddg"], ascending=False
    )
    if len(cofactor):
        selected.append(("enzyme/cofactor", cofactor.iloc[0]))

    text = (
        frame["prot.molecule_name"].fillna("").astype(str)
        + " "
        + frame["prot.organism"].fillna("").astype(str)
        + " "
        + frame["prot.protein_class"].fillna("").astype(str)
    ).str.upper()
    inhibitor = positive[text.loc[positive.index].str.contains("PROTEASE|KINASE|RECEPTOR|HUMAN IMMUNODEFICIENCY")]
    inhibitor = inhibitor.sort_values(["success_margin", "max_ddg"], ascending=False)
    if len(inhibitor):
        selected.append(("drug-target/inhibitor", inhibitor.iloc[0]))

    unusual = positive.copy()
    unusual["ligand_mw_numeric"] = pd.to_numeric(unusual["lig.mol_weight"], errors="coerce")
    unusual = unusual.sort_values(["ligand_mw_numeric", "success_margin"], ascending=False)
    if len(unusual):
        selected.append(("chemically unusual/high-mass ligand", unusual.iloc[0]))

    pair_candidates = []
    for sample_id, group in frame.groupby("sample_id"):
        pos = group[group["disruptive_ge_1"].eq(1)]
        neg = group[group["disruptive_ge_1"].eq(0)]
        for _, prow in pos.iterrows():
            for _, nrow in neg.iterrows():
                distance_gap = abs(float(prow["min_ligand_distance"]) - float(nrow["min_ligand_distance"]))
                score_gap = float(prow["LAIGN"]) - float(nrow["LAIGN"])
                if distance_gap <= 0.75 and score_gap > 0:
                    pair_candidates.append((score_gap - 0.1 * distance_gap, prow, nrow))
    if pair_candidates:
        _, prow, nrow = max(pair_candidates, key=lambda item: item[0])
        pair = prow.copy()
        pair["paired_control_residue"] = (
            f"{nrow['protein_chain']}:{nrow['protein_residue_type']}{nrow['protein_residue_number']}"
        )
        pair["paired_control_ddg"] = nrow["max_ddg"]
        pair["paired_control_laign"] = nrow["LAIGN"]
        pair["paired_control_distance"] = nrow["min_ligand_distance"]
        selected.append(("distance-matched discrimination", pair))

    failure = positive.sort_values(
        ["laign_percentile_assayed", "max_ddg"], ascending=[True, False]
    )
    if len(failure):
        selected.append(("failure case", failure.iloc[0]))

    rows = []
    seen = set()
    for case_type, row in selected:
        key = (
            case_type,
            row["sample_id"],
            row["protein_chain"],
            row["protein_residue_number"],
        )
        if key in seen:
            continue
        seen.add(key)
        item = {"case_type": case_type, **row.to_dict()}
        rows.append(item)
    return pd.DataFrame(rows)


def fmt(value: object, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def write_report(
    args: argparse.Namespace,
    site_scores: pd.DataFrame,
    metrics: pd.DataFrame,
    topk: pd.DataFrame,
    bootstrap: pd.DataFrame,
    budget_ci: pd.DataFrame,
    cases: pd.DataFrame,
    output: Path,
) -> None:
    primary = metrics[
        metrics["subset"].eq("all") & metrics["threshold_kcal_mol"].eq(1.0)
    ]
    summary = (
        primary.groupby("method")
        .agg(
            auprc_mean=("site_auprc", "mean"),
            auprc_std=("site_auprc", "std"),
            auroc_mean=("site_auroc", "mean"),
            auroc_std=("site_auroc", "std"),
            rho_mean=("spearman_ddg", "mean"),
            rho_std=("spearman_ddg", "std"),
            complexes=("complexes", "max"),
            sites=("sites", "max"),
            positives=("positives", "max"),
        )
        .reset_index()
    )
    top_primary = topk[topk["threshold_kcal_mol"].eq(1.0)]
    top_summary = (
        top_primary.groupby("method")
        .agg(
            recall5=("recall_at_5_all_residues", "mean"),
            recall10=("recall_at_10_all_residues", "mean"),
            hit5=("hit_at_5_all_residues", "mean"),
            hit10=("hit_at_10_all_residues", "mean"),
        )
        .reset_index()
    )
    lines = [
        f"# Platinum Mutation-Affinity Validation ({args.stage})",
        "",
        "## Protocol",
        "",
        "- Endpoint: experimental affinity loss after a single mutation, expressed as ΔΔG = RT ln(Kmut/Kwt) at 298.15 K.",
        "- Primary positive: site has at least one assayed substitution with ΔΔG ≥ 1.0 kcal/mol; 0.5 and 2.0 kcal/mol are prespecified sensitivity thresholds.",
        "- Classification uses assayed residue sites only. Unassayed residues are never treated as negatives.",
        "- Top-5/top-10 recovery ranks all standard residues, then measures recovery of known disruptive assayed sites; unknown sites remain unlabeled.",
        "- LAIGN checkpoints and Stage-2 hyperparameters are frozen relative to Platinum.",
        f"- Runs: {site_scores['run'].nunique()}; accepted complexes: {site_scores['sample_id'].nunique()}; assayed sites: {site_scores.drop_duplicates(['sample_id','protein_chain','protein_residue_number','protein_residue_type']).shape[0]}.",
        "",
        "## Primary endpoint (ΔΔG ≥ 1.0 kcal/mol)",
        "",
        "| method | complexes | assayed sites | disruptive sites | AUPRC | AUROC | Spearman ρ with max ΔΔG |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["LAIGN", "fixed fusion", "distance", "raw interaction", "PLIP count", "ProLIF count"]
    for method in order:
        row = summary[summary["method"].eq(method)].iloc[0]
        lines.append(
            f"| {method} | {int(row['complexes'])} | {int(row['sites'])} | {int(row['positives'])} | "
            f"{fmt(row['auprc_mean'])} ± {fmt(row['auprc_std'])} | {fmt(row['auroc_mean'])} ± {fmt(row['auroc_std'])} | "
            f"{fmt(row['rho_mean'])} ± {fmt(row['rho_std'])} |"
        )
    lines.extend(
        [
            "",
            "## Experimental prioritization budget",
            "",
            "| method | recall@5 | recall@10 | hit@5 | hit@10 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in order:
        row = top_summary[top_summary["method"].eq(method)].iloc[0]
        lines.append(
            f"| {method} | {fmt(row['recall5'])} | {fmt(row['recall10'])} | {fmt(row['hit5'])} | {fmt(row['hit10'])} |"
        )
    lines.extend(
        [
            "",
            "Paired uncertainty for recovery of disruptive sites:",
            "",
            "| budget | comparator | LAIGN − comparator recall | 95% CI | one-sided bootstrap p |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in budget_ci.to_dict("records"):
        lines.append(
            f"| {int(row['budget_k'])} | {row['comparator']} | {fmt(row['observed_delta_recall'])} | "
            f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | {fmt(row['one_sided_p_delta_le_0'])} |"
        )
    lines.extend(
        [
            "",
            "## Complex-blocked bootstrap (primary endpoint)",
            "",
            "| comparator | LAIGN − comparator AUPRC | 95% CI | one-sided bootstrap p |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in bootstrap.to_dict("records"):
        lines.append(
            f"| {row['comparator']} | {fmt(row['observed_delta_auprc'])} | "
            f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | {fmt(row['one_sided_p_delta_le_0'])} |"
        )
    lines.extend(
        [
            "",
            "## Leakage and identity strata",
            "",
            "| subset | complexes | sites | positives | LAIGN AUPRC | fixed-fusion AUPRC | distance AUPRC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    strata = metrics[metrics["threshold_kcal_mol"].eq(1.0)]
    for subset in ["all", "no exact PDB", "no exact UniProt", "sequence identity <70%", "sequence identity <30%"]:
        group = strata[strata["subset"].eq(subset)]
        pivot = group.groupby("method")["site_auprc"].mean()
        row = group.iloc[0]
        lines.append(
            f"| {subset} | {int(row['complexes'])} | {int(row['sites'])} | {int(row['positives'])} | "
            f"{fmt(pivot.get('LAIGN', math.nan))} | {fmt(pivot.get('fixed fusion', math.nan))} | "
            f"{fmt(pivot.get('distance', math.nan))} |"
        )
    lines.extend(
        [
            "",
            "The <30% identity stratum is reported for transparency; its very small number of complexes makes it descriptive rather than inferential.",
            "",
            "## Threshold sensitivity",
            "",
            "| threshold (kcal/mol) | positives | prevalence | LAIGN AUPRC | fixed-fusion AUPRC | distance AUPRC |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    all_metrics = metrics[metrics["subset"].eq("all")]
    for threshold in THRESHOLDS:
        group = all_metrics[all_metrics["threshold_kcal_mol"].eq(threshold)]
        pivot = group.groupby("method")["site_auprc"].mean()
        row = group.iloc[0]
        lines.append(
            f"| {threshold:.1f} | {int(row['positives'])} | {fmt(row['prevalence'])} | "
            f"{fmt(pivot.get('LAIGN', math.nan))} | {fmt(pivot.get('fixed fusion', math.nan))} | "
            f"{fmt(pivot.get('distance', math.nan))} |"
        )
    lines.extend(
        [
            "",
            "## Mutation-affinity case candidates",
            "",
            "| type | PDB / ligand | protein | residue | mutations | max ΔΔG | LAIGN | distance (Å) | note |",
            "|---|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in cases.to_dict("records"):
        note = ""
        if row["case_type"] == "distance-matched discrimination":
            note = (
                f"control {row.get('paired_control_residue', 'NA')}: ΔΔG={fmt(row.get('paired_control_ddg'), 2)}, "
                f"LAIGN={fmt(row.get('paired_control_laign'))}, distance={fmt(row.get('paired_control_distance'), 2)} Å"
            )
        elif row["case_type"] == "failure case":
            note = "experimentally disruptive site ranked comparatively low by LAIGN"
        lines.append(
            f"| {row['case_type']} | {row['wt_pdb_id']} / {row['ligand_ccd_code']} | "
            f"{str(row.get('prot.molecule_name', 'NA')).replace('|', '/')} | "
            f"{row['protein_chain']}:{row['protein_residue_type']}{row['protein_residue_number']} | "
            f"{row['mutations']} | {fmt(row['max_ddg'], 2)} | {fmt(row['LAIGN'])} | "
            f"{fmt(row['min_ligand_distance'], 2)} | {note} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This endpoint is independent of PLIP: PLIP and ProLIF appear only as structural comparators, while the outcome is experimentally measured affinity change. The analysis evaluates residue prioritization rather than mutant-amino-acid effect prediction. Exact-PDB/UniProt and sequence-identity strata are kept separate so that biological validation strength is not conflated with ordinary benchmark performance.",
            "",
            "## Artifacts",
            "",
            "- `assayed_site_scores.csv.gz`: one row per assayed site and run.",
            "- `classification_metrics.csv`: endpoint metrics by run, threshold, and leakage stratum.",
            "- `topk_metrics.csv`: full-protein top-5/top-10 recovery.",
            "- `blocked_bootstrap.csv`: complex-blocked paired uncertainty.",
            "- `budget_bootstrap.csv`: paired uncertainty for top-5/top-10 disruptive-site recovery.",
            "- `case_candidates.csv`: auditable case-study selection.",
            "- `candidate_residue_scores.csv.gz`: all-residue scores used for ranking.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    base = ROOT / "experiments/validation/data/processed/platinum"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(base / "platinum_manifest.csv.gz", low_memory=False)
    candidates = load_candidate_scores(args.archives, manifest, base)
    candidates["stage"] = args.stage
    sites = load_site_table(base)
    keys = [
        "sample_id",
        "protein_chain",
        "protein_residue_number",
        "protein_residue_type",
    ]
    site_scores = candidates.merge(sites, on=keys, how="inner", validate="many_to_one")
    mapped = site_scores[keys].drop_duplicates().shape[0]
    if mapped != len(sites):
        missing = sites.merge(site_scores[keys].drop_duplicates(), on=keys, how="left", indicator=True)
        missing = missing[missing["_merge"].eq("left_only")]
        missing.to_csv(out_dir / "unmapped_assayed_sites.csv", index=False)
        raise RuntimeError(f"mapped {mapped}/{len(sites)} assayed sites")

    metrics = classification_metrics(site_scores)
    topk = topk_metrics(site_scores, candidates)
    averaged = averaged_site_scores(site_scores)
    candidate_average = candidates.groupby(keys, as_index=False)[
        [*METHODS, "PLIP count", "ProLIF count"]
    ].mean()
    bootstrap = bootstrap_comparisons(averaged, args.bootstrap, args.seed)
    budget_ci = budget_bootstrap(
        averaged, candidate_average, args.bootstrap, args.seed
    )
    cases = select_cases(averaged, candidate_average)

    candidates.to_csv(out_dir / "candidate_residue_scores.csv.gz", index=False)
    site_scores.to_csv(out_dir / "assayed_site_scores.csv.gz", index=False)
    averaged.to_csv(out_dir / "assayed_site_scores_ensemble_mean.csv", index=False)
    metrics.to_csv(out_dir / "classification_metrics.csv", index=False)
    topk.to_csv(out_dir / "topk_metrics.csv", index=False)
    bootstrap.to_csv(out_dir / "blocked_bootstrap.csv", index=False)
    budget_ci.to_csv(out_dir / "budget_bootstrap.csv", index=False)
    cases.to_csv(out_dir / "case_candidates.csv", index=False)
    write_report(
        args,
        site_scores,
        metrics,
        topk,
        bootstrap,
        budget_ci,
        cases,
        out_dir / "PLATINUM_RESULTS.md",
    )
    audit = {
        "stage": args.stage,
        "archives": args.archives,
        "complexes": int(averaged["sample_id"].nunique()),
        "assayed_sites": int(len(averaged)),
        "mutation_records": int(averaged["mutation_count"].sum()),
        "candidate_residues": int(candidate_average.shape[0]),
        "runs": int(site_scores["run"].nunique()),
        "bootstrap_replicates": args.bootstrap,
        "unmapped_sites": 0,
    }
    (out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
