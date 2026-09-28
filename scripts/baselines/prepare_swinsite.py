#!/usr/bin/env python3
"""Prepare the External subset PLINDER smoke subset for SwinSite inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
AA3 = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-samples",
        default=str(ROOT / "outputs/external_labind_plinder_external_subset_smoke_prep/selected_samples.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "outputs/external_swinsite_smoke"),
    )
    parser.add_argument(
        "--table",
        default=str(ROOT / "outputs/tables/table_swinsite_plinder_swinsite_smoke_prep.md"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_clean_pdb(source: Path, target: Path) -> int:
    residues = set()
    lines = []
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM  "):
                continue
            resname = line[17:20].strip().upper()
            if resname not in AA3:
                continue
            chain = line[21:22].strip()
            resnr = line[22:26].strip() + line[26:27].strip()
            residues.add((chain, resnr, resname))
            lines.append(line.rstrip("\n"))
    if not lines:
        raise ValueError(f"No standard protein ATOM records found in {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\nTER\nEND\n", encoding="utf-8")
    return len(residues)


def prepare(args: argparse.Namespace) -> dict:
    selected = pd.read_csv(args.selected_samples)
    out_dir = Path(args.out_dir)
    input_dir = out_dir / "work" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in selected.to_dict("records"):
        name = str(row["labind_name"])
        sample_dir = input_dir / name
        target = sample_dir / "protein.pdb"
        if args.force or not target.exists() or target.stat().st_size == 0:
            residue_count = write_clean_pdb(Path(str(row["receptor_path"])), target)
        else:
            residue_count = write_clean_pdb(Path(str(row["receptor_path"])), target)
        current = dict(row)
        current["swinsite_name"] = name
        current["swinsite_pdb_path"] = str(target)
        current["swinsite_residues"] = residue_count
        rows.append(current)

    manifest_path = out_dir / "selected_samples_swinsite.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    summary = {
        "selected_samples": int(len(rows)),
        "splits": sorted({str(row["split"]) for row in rows}),
        "input_dir": str(input_dir),
        "manifest": str(manifest_path),
        "source_subset": str(args.selected_samples),
        "protocol": "External subset PLINDER single-chain subset; SwinSite input keeps standard protein ATOM records with original residue numbering.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def write_table(summary: dict, path: Path) -> None:
    lines = [
        "# SwinSite SwinSite PLINDER Smoke Prep",
        "",
        f"- selected samples: `{summary['selected_samples']}`",
        f"- splits: `{summary['splits']}`",
        f"- input dir: `{summary['input_dir']}`",
        f"- manifest: `{summary['manifest']}`",
        "- model: SwinSite official 4-fold checkpoint, evaluated as an external rerun.",
        "- protocol: same External subset single-chain PLINDER subset; input PDBs are protein-only `ATOM` records with original residue numbering.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = prepare(args)
    write_table(summary, Path(args.table))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
