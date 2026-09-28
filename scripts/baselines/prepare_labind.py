#!/usr/bin/env python3
"""Prepare every release-supported Post-freeze complex for official LABind inference."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import prepare_labind_plinder_smoke as labind_prep  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest",
        default=str(
            ROOT / "outputs/postfreeze_holdout/postfreeze_split_manifest.csv.gz"
        ),
    )
    parser.add_argument(
        "--labind-root", default=str(ROOT / ".tools/deep_baselines/src/LABind-main")
    )
    parser.add_argument("--max-samples", type=int, default=70)
    parser.add_argument("--max-sequence-length", type=int, default=1500)
    parser.add_argument(
        "--out-dir", default=str(ROOT / "outputs/postfreeze_holdout/labind_prep")
    )
    parser.add_argument(
        "--table", default=str(ROOT / "outputs/tables/table_postfreeze_labind_prep.md")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labind_root = Path(args.labind_root)
    with (labind_root / "tools/ligand.pkl").open("rb") as handle:
        ligand_dict = pickle.load(handle)
    ligand_keys = {str(key).upper() for key in ligand_dict}
    smiles = labind_prep.load_smiles(labind_root / "tools/smiles.txt")

    manifest = pd.read_csv(args.split_manifest, low_memory=False)
    external = manifest[manifest["supervision_split"] == "external_test"].copy()
    external["ligand_ccd_code"] = external["ligand_ccd_code"].astype(str).str.upper()
    external["stable_rank"] = external["sample_id"].astype(str).map(labind_prep.stable_rank)
    external = external.sort_values(["stable_rank", "sample_id"])

    rows = []
    exclusions = []
    for row in external.to_dict("records"):
        ligand = str(row["ligand_ccd_code"])
        if ligand not in ligand_keys or ligand not in smiles:
            exclusions.append(
                {
                    "sample_id": row["sample_id"],
                    "ligand_ccd_code": ligand,
                    "reason": "absent_from_official_LABind_ligand_resources",
                }
            )
            continue
        sequence, chains, parsed_length = labind_prep.pdb_sequence_and_chains(
            str(row["receptor_path"])
        )
        if len(chains) != 1 or not sequence or len(sequence) > args.max_sequence_length:
            exclusions.append(
                {
                    "sample_id": row["sample_id"],
                    "ligand_ccd_code": ligand,
                    "reason": "official_LABind_single_chain_or_length_constraint",
                }
            )
            continue
        row["labind_sequence"] = sequence
        row["labind_chains"] = ",".join(chains)
        row["labind_sequence_length"] = parsed_length
        rows.append(row)
        if len(rows) >= args.max_samples:
            break
    if not rows:
        raise RuntimeError("No Post-freeze complexes are supported by the released LABind resources")

    summary = labind_prep.write_outputs(rows, args, ligand_dict, smiles)
    out_dir = Path(args.out_dir)
    pd.DataFrame(exclusions).to_csv(out_dir / "excluded_samples.csv", index=False)
    summary.update(
        {
            "hypothesis": "Common-coverage",
            "requested_postfreeze_samples": int(len(external)),
            "official_resource_coverage": len(rows) / len(external),
            "excluded_samples": len(exclusions),
            "coverage_policy": "No ligand embeddings or SMILES are synthesized for unsupported CCD codes.",
        }
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Common-coverage LABind Post-freeze Preparation",
        "",
        f"- Post-freeze complexes: `{len(external)}`",
        f"- supported by released LABind ligand resources: `{len(rows)}`",
        f"- unsupported: `{len(exclusions)}`",
        f"- common-coverage fraction: `{len(rows) / len(external):.4f}`",
        "- unsupported ligand features are not imputed or synthesized",
    ]
    table = Path(args.table)
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
