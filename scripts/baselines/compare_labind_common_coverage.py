#!/usr/bin/env python3
"""Compare Post-freeze methods on the 51 complexes supported by released LABind resources."""

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
    parser.add_argument("--labind", default=str(base / "labind_eval/predictions_labind.csv.gz"))
    parser.add_argument(
        "--fpocket", default=str(base / "baselines_geometry_full/predictions_fpocket.csv.gz")
    )
    parser.add_argument(
        "--p2rank", default=str(base / "baselines_geometry_full/predictions_p2rank.csv.gz")
    )
    parser.add_argument(
        "--prolif", default=str(base / "baseline_prolif_ccd_full/predictions_prolif.csv.gz")
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=8301)
    parser.add_argument("--out-json", default=str(base / "labind_common_coverage_summary.json"))
    parser.add_argument(
        "--table", default=str(ROOT / "outputs/tables/table_postfreeze_labind_common_coverage.md")
    )
    return parser.parse_args()


def as_archive(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path, allow_pickle=True)
    keys = [
        "external_test__labels",
        "external_test__sample_index",
        "external_test__sample_ids",
        "external_test__score__raw_distance",
        "external_test__score__fixed_hybrid",
        "external_test__score__trained_stage2",
    ]
    return {key: np.asarray(loaded[key]) for key in keys}


def labind_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    rows = []
    for sample_id, group in frame.groupby("sample_id", sort=False):
        values = postfreeze.graph_metrics(
            group["label"].to_numpy(dtype=np.int32),
            group["score"].to_numpy(dtype=np.float64),
        )
        rows.append({"sample_id": str(sample_id), **dict(zip(postfreeze.METRICS, values))})
    return pd.DataFrame(rows).set_index("sample_id")


def main() -> int:
    args = parse_args()
    archives = [as_archive(Path(path)) for path in args.archives]
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
    labels = postfreeze.endpoint_labels(archives[0], manifest, site)["plip_active"]

    methods = {
        "raw_distance": postfreeze.archive_rows(
            archives[0], labels, "external_test__score__raw_distance"
        ),
        "fixed_hybrid": postfreeze.archive_rows(
            archives[0], labels, "external_test__score__fixed_hybrid"
        ),
        "fixed_hybrid_A": postfreeze.archive_rows(
            archives[0], labels, "external_test__score__fixed_hybrid"
        ),
        "fixed_hybrid_B": postfreeze.archive_rows(
            archives[1], labels, "external_test__score__fixed_hybrid"
        ),
        "fixed_hybrid_C": postfreeze.archive_rows(
            archives[2], labels, "external_test__score__fixed_hybrid"
        ),
        "LAIGN_A": postfreeze.archive_rows(
            archives[0], labels, "external_test__score__trained_stage2"
        ),
        "LAIGN_B": postfreeze.archive_rows(
            archives[1], labels, "external_test__score__trained_stage2"
        ),
        "LAIGN_C": postfreeze.archive_rows(
            archives[2], labels, "external_test__score__trained_stage2"
        ),
        "fpocket": postfreeze.baseline_rows(Path(args.fpocket), "plip_active", manifest, site),
        "P2Rank": postfreeze.baseline_rows(Path(args.p2rank), "plip_active", manifest, site),
        "ProLIF": postfreeze.baseline_rows(Path(args.prolif), "plip_active", manifest, site),
        "LABind": labind_rows(Path(args.labind)),
    }
    common_ids = methods["LABind"].index
    for method, rows in methods.items():
        missing = set(common_ids) - set(rows.index)
        if missing:
            raise ValueError(f"{method} misses {len(missing)} common-coverage samples")
        methods[method] = rows.loc[common_ids]

    summaries = {
        method: {metric: postfreeze.describe(rows[metric].to_numpy()) for metric in postfreeze.METRICS}
        for method, rows in methods.items()
    }
    paired = {}
    for method in ("raw_distance", "fixed_hybrid", "fpocket", "P2Rank", "ProLIF", "LABind"):
        paired[method] = {
            metric: postfreeze.paired_test(
                methods["LAIGN_A"][metric].to_numpy() - methods[method][metric].to_numpy(),
                args.bootstrap_draws,
                args.seed + 10 * list(methods).index(method) + metric_index,
            )
            for metric_index, metric in enumerate(postfreeze.METRICS)
        }

    external = manifest[manifest["supervision_split"] == "external_test"]
    sequence_groups = pd.Series(
        external["receptor_sequence"].astype(str).to_numpy(),
        index=external["sample_id"].astype(str),
    ).loc[common_ids]
    clustered = {
        method: {
            metric: postfreeze.cluster_paired_test(
                methods["LAIGN_A"][metric] - methods[method][metric],
                sequence_groups,
                args.bootstrap_draws,
                args.seed + 5000 + 10 * list(methods).index(method) + metric_index,
            )
            for metric_index, metric in enumerate(postfreeze.METRICS)
        }
        for method in paired
    }
    replicate = {"LAIGN": {}, "fixed_hybrid": {}, "paired_delta": {}}
    for metric in postfreeze.METRICS:
        laign_values = np.asarray(
            [summaries[f"LAIGN_{name}"][metric]["mean"] for name in "ABC"]
        )
        control_values = np.asarray(
            [summaries[f"fixed_hybrid_{name}"][metric]["mean"] for name in "ABC"]
        )
        replicate["LAIGN"][metric] = postfreeze.describe(laign_values)
        replicate["fixed_hybrid"][metric] = postfreeze.describe(control_values)
        replicate["paired_delta"][metric] = postfreeze.describe(laign_values - control_values)
    payload = {
        "hypothesis": "Common-coverage",
        "coverage": {
            "common_complexes": int(len(common_ids)),
            "full_postfreeze_complexes": 70,
            "fraction": float(len(common_ids) / 70),
            "restriction": "released LABind CCD ligand resources",
        },
        "summaries": summaries,
        "complete_pipeline_replicates": replicate,
        "paired_LAIGN_A_minus_method": paired,
        "receptor_sequence_clustered_tests": clustered,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Common-coverage Post-freeze LABind Common-Coverage Comparison",
        "",
        "| method | N | graph AP | IoU@P | recall@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ("LAIGN_A", "LAIGN_B", "LAIGN_C", "fixed_hybrid", "raw_distance", "ProLIF", "P2Rank", "fpocket", "LABind"):
        row = summaries[method]
        lines.append(
            f"| {method} | {len(common_ids)} | {row['graph_ap']['mean']:.4f} | "
            f"{row['iou_at_p']['mean']:.4f} | {row['recall_at_10']['mean']:.4f} |"
        )
    lines.extend(["", "LABind coverage is 51/70 because 19 Post-freeze complexes use CCD codes absent from its released ligand resources."])
    table = Path(args.table)
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(out), "table": str(table), "common": len(common_ids)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
