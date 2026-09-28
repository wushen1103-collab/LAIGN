#!/usr/bin/env python3
"""Evaluate frozen Post-freeze residue rankings separately for five interaction types."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluation import summarize_postfreeze_holdout as postfreeze  # noqa: E402


INTERACTION_TYPES = (
    "hydrophobic_contact",
    "hydrogen_bond",
    "salt_bridge",
    "pi_stacking",
    "pi_cation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "outputs/postfreeze_holdout"
    parser.add_argument(
        "--archives",
        nargs=3,
        default=[str(base / f"replicate_{name}/predictions.npz") for name in "ABC"],
    )
    parser.add_argument("--manifest", default=str(base / "postfreeze_split_manifest.csv.gz"))
    parser.add_argument("--site-labels", default=str(base / "postfreeze_site_labels.csv.gz"))
    parser.add_argument(
        "--interactions", default=str(base / "postfreeze_plip_interactions.csv.gz")
    )
    parser.add_argument(
        "--p2rank", default=str(base / "baselines_geometry_full/predictions_p2rank.csv.gz")
    )
    parser.add_argument(
        "--prolif", default=str(base / "baseline_prolif_ccd_full/predictions_prolif.csv.gz")
    )
    parser.add_argument("--out-json", default=str(base / "interaction_type_summary.json"))
    parser.add_argument(
        "--table", default=str(ROOT / "outputs/tables/table_postfreeze_interaction_types.md")
    )
    return parser.parse_args()


def normalize_residue_number(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def residue_key(chain: object, number: object, residue_type: object) -> tuple[str, str, str]:
    return (
        str(chain).strip(),
        normalize_residue_number(number),
        str(residue_type).strip().upper(),
    )


def load_archive(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path, allow_pickle=True)
    keys = [
        "external_test__labels",
        "external_test__sample_index",
        "external_test__sample_ids",
        "external_test__score__fixed_hybrid",
        "external_test__score__trained_stage2",
    ]
    return {key: np.asarray(loaded[key]) for key in keys}


def build_type_labels(
    archive: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    site: pd.DataFrame,
    interactions: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, list[tuple[str, str, str]]]]:
    sample_ids = archive["external_test__sample_ids"].astype(str)
    sample_index = archive["external_test__sample_index"].astype(np.int32)
    active_labels = archive["external_test__labels"].astype(np.int32)
    external = manifest[manifest["supervision_split"] == "external_test"]
    system_by_sample = dict(zip(external["sample_id"].astype(str), external["system_id"].astype(str)))
    site_by_system = {str(key): frame for key, frame in site.groupby("system_id", sort=False)}

    interaction_lookup: dict[tuple[str, tuple[str, str, str]], set[str]] = {}
    for row in interactions.to_dict("records"):
        key = (
            str(row["system_id"]),
            residue_key(
                row["protein_chain"],
                row["protein_residue_number"],
                row["protein_residue_type"],
            ),
        )
        interaction_lookup.setdefault(key, set()).add(str(row["interaction_type"]))

    labels = np.zeros((active_labels.size, len(INTERACTION_TYPES)), dtype=np.int8)
    keys_by_sample = {}
    for index, sample_id in enumerate(sample_ids):
        mask = sample_index == index
        system_id = system_by_sample[sample_id]
        rows = site_by_system[system_id]
        residue_keys = [
            residue_key(row.protein_chain, row.protein_residue_number, row.protein_residue_type)
            for row in rows.itertuples(index=False)
        ]
        if len(residue_keys) != int(mask.sum()):
            raise ValueError(f"Residue count mismatch for {sample_id}")
        keys_by_sample[sample_id] = residue_keys
        positions = np.flatnonzero(mask)
        for local_index, key in enumerate(residue_keys):
            present = interaction_lookup.get((system_id, key), set())
            for type_index, interaction_type in enumerate(INTERACTION_TYPES):
                labels[positions[local_index], type_index] = int(interaction_type in present)
    if not np.array_equal(labels.max(axis=1), active_labels):
        raise ValueError("Union of type labels does not reproduce the frozen active label")
    return labels, keys_by_sample


def baseline_vector(
    path: Path,
    sample_ids: np.ndarray,
    sample_index: np.ndarray,
    keys_by_sample: dict[str, list[tuple[str, str, str]]],
) -> np.ndarray:
    frame = pd.read_csv(path, low_memory=False)
    output = np.empty(sample_index.size, dtype=np.float64)
    for index, sample_id in enumerate(sample_ids.astype(str)):
        group = frame[frame["sample_id"].astype(str) == sample_id]
        lookup = {
            residue_key(row.chain, row.residue_number, row.residue_type): float(row.score)
            for row in group.itertuples(index=False)
        }
        keys = keys_by_sample[sample_id]
        missing = [key for key in keys if key not in lookup]
        if missing:
            raise ValueError(f"Baseline residue mismatch for {sample_id}: {missing[:3]}")
        output[sample_index == index] = np.asarray([lookup[key] for key in keys])
    return output


def per_graph_rows(
    labels: np.ndarray,
    scores: np.ndarray,
    sample_index: np.ndarray,
    sample_ids: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for index, sample_id in enumerate(sample_ids.astype(str)):
        mask = sample_index == index
        if int(labels[mask].sum()) == 0:
            continue
        values = postfreeze.graph_metrics(labels[mask], scores[mask])
        rows.append({"sample_id": sample_id, **dict(zip(postfreeze.METRICS, values))})
    return pd.DataFrame(rows).set_index("sample_id")


def main() -> int:
    args = parse_args()
    archives = [load_archive(Path(path)) for path in args.archives]
    for archive in archives[1:]:
        for key in (
            "external_test__labels",
            "external_test__sample_index",
            "external_test__sample_ids",
        ):
            if not np.array_equal(archives[0][key], archive[key]):
                raise ValueError(f"Archive order mismatch for {key}")
    manifest = pd.read_csv(args.manifest, low_memory=False)
    site = pd.read_csv(args.site_labels, low_memory=False)
    interactions = pd.read_csv(args.interactions, low_memory=False)
    type_labels, keys_by_sample = build_type_labels(archives[0], manifest, site, interactions)
    sample_ids = archives[0]["external_test__sample_ids"].astype(str)
    sample_index = archives[0]["external_test__sample_index"].astype(np.int32)
    scores = {
        "LAIGN_A": archives[0]["external_test__score__trained_stage2"].astype(np.float64),
        "LAIGN_B": archives[1]["external_test__score__trained_stage2"].astype(np.float64),
        "LAIGN_C": archives[2]["external_test__score__trained_stage2"].astype(np.float64),
        "fixed_hybrid": archives[0]["external_test__score__fixed_hybrid"].astype(np.float64),
        "fixed_hybrid_A": archives[0]["external_test__score__fixed_hybrid"].astype(np.float64),
        "fixed_hybrid_B": archives[1]["external_test__score__fixed_hybrid"].astype(np.float64),
        "fixed_hybrid_C": archives[2]["external_test__score__fixed_hybrid"].astype(np.float64),
        "P2Rank": baseline_vector(Path(args.p2rank), sample_ids, sample_index, keys_by_sample),
        "ProLIF": baseline_vector(Path(args.prolif), sample_ids, sample_index, keys_by_sample),
    }

    summaries = {}
    per_graph = {}
    for type_index, interaction_type in enumerate(INTERACTION_TYPES):
        labels = type_labels[:, type_index]
        per_graph[interaction_type] = {
            method: per_graph_rows(labels, method_scores, sample_index, sample_ids)
            for method, method_scores in scores.items()
        }
        summaries[interaction_type] = {
            method: {
                metric: postfreeze.describe(rows[metric].to_numpy()) for metric in postfreeze.METRICS
            }
            for method, rows in per_graph[interaction_type].items()
        }

    replicate = {}
    for interaction_type in INTERACTION_TYPES:
        replicate[interaction_type] = {"LAIGN": {}, "fixed_hybrid": {}, "paired_delta": {}}
        for metric in postfreeze.METRICS:
            laign_values = np.asarray(
                [
                    summaries[interaction_type][f"LAIGN_{name}"][metric]["mean"]
                    for name in "ABC"
                ]
            )
            control_values = np.asarray(
                [
                    summaries[interaction_type][f"fixed_hybrid_{name}"][metric]["mean"]
                    for name in "ABC"
                ]
            )
            replicate[interaction_type]["LAIGN"][metric] = postfreeze.describe(laign_values)
            replicate[interaction_type]["fixed_hybrid"][metric] = postfreeze.describe(control_values)
            replicate[interaction_type]["paired_delta"][metric] = postfreeze.describe(
                laign_values - control_values
            )
    paired = {}
    for type_index, interaction_type in enumerate(INTERACTION_TYPES):
        paired[interaction_type] = {}
        for method in ("fixed_hybrid", "P2Rank", "ProLIF"):
            common = per_graph[interaction_type]["LAIGN_A"].index.intersection(
                per_graph[interaction_type][method].index
            )
            paired[interaction_type][method] = postfreeze.paired_test(
                per_graph[interaction_type]["LAIGN_A"].loc[common, "graph_ap"].to_numpy()
                - per_graph[interaction_type][method].loc[common, "graph_ap"].to_numpy(),
                10000,
                8401 + 10 * type_index + list(scores).index(method),
            )

    payload = {
        "hypothesis": "Interaction-type analysis",
        "interaction_types": list(INTERACTION_TYPES),
        "type_label_union_matches_primary_label": True,
        "summaries": summaries,
        "complete_pipeline_replicates": replicate,
        "paired_graph_AP_LAIGN_A_minus_method": paired,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Interaction-type analysis Post-freeze Interaction-Type Performance",
        "",
        "| interaction | complexes | method | graph AP | recall@10 |",
        "|---|---:|---|---:|---:|",
    ]
    for interaction_type in INTERACTION_TYPES:
        for method in ("LAIGN_A", "LAIGN_B", "LAIGN_C", "fixed_hybrid", "ProLIF", "P2Rank"):
            row = summaries[interaction_type][method]
            lines.append(
                f"| {interaction_type} | {row['graph_ap']['n']} | {method} | "
                f"{row['graph_ap']['mean']:.4f} | {row['recall_at_10']['mean']:.4f} |"
            )
    table = Path(args.table)
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(out), "table": str(table)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
