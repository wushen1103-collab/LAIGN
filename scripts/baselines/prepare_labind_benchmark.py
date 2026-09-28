#!/usr/bin/env python3
"""Prepare a deterministic PLINDER subset for LABind feature-generation smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest",
        default=str(
            ROOT
            / "data/processed/structure_supervision/structure_supervision_split_manifest.csv.gz"
        ),
    )
    parser.add_argument(
        "--labind-root",
        default=str(ROOT / ".tools/deep_baselines/src/LABind-main"),
    )
    parser.add_argument("--splits", nargs="+", default=["external_val", "external_test"])
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-sequence-length", type=int, default=1500)
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "outputs/external_labind_plinder_external_subset_smoke_prep"),
    )
    parser.add_argument(
        "--table",
        default=str(ROOT / "outputs/tables/table_labind_plinder_external_subset_smoke_prep.md"),
    )
    return parser.parse_args()


def stable_rank(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_name(index: int, row: dict) -> str:
    pdb = str(row["pdb_id"]).lower()
    lig = str(row["ligand_ccd_code"]).upper()
    digest = hashlib.sha1(str(row["sample_id"]).encode("utf-8")).hexdigest()[:8]
    return f"plinder_{index:03d}_{pdb}_{lig}_{digest}"


def pdb_sequence_and_chains(path: str) -> tuple[str, list[str], int]:
    residues = []
    seen = set()
    chains = []
    chain_seen = set()
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM  "):
                continue
            resname = line[17:20].strip().upper()
            if resname not in AA1:
                continue
            chain = line[21:22].strip() or "_"
            resnr = line[22:27].strip()
            key = (chain, resnr, resname)
            if chain not in chain_seen:
                chains.append(chain)
                chain_seen.add(chain)
            if key in seen:
                continue
            seen.add(key)
            residues.append((chain, resname))
    seq = "".join(AA1[resname] for _, resname in residues)
    return seq, chains, len(residues)


def load_smiles(path: Path) -> dict[str, str]:
    smiles = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2:
                smiles[parts[0].upper()] = parts[1]
    return smiles


def select_rows(args: argparse.Namespace, ligand_keys: set[str], smiles: dict[str, str]) -> list[dict]:
    manifest = pd.read_csv(args.split_manifest, low_memory=False)
    frame = manifest[
        (manifest["source"] == "plinder")
        & (manifest["supervision_split"].isin(args.splits))
        & (manifest["structure_complete"] == 1)
        & (manifest["ligand_ccd_code"].notna())
    ].copy()
    frame["ligand_ccd_code"] = frame["ligand_ccd_code"].astype(str).str.upper()
    frame = frame[
        frame["ligand_ccd_code"].isin(ligand_keys)
        & frame["ligand_ccd_code"].isin(smiles)
    ].copy()
    frame["stable_rank"] = frame["sample_id"].astype(str).map(stable_rank)
    frame = frame.sort_values(["supervision_split", "stable_rank", "sample_id"])

    selected = []
    per_split = max(1, args.max_samples // max(1, len(args.splits)))
    for split in args.splits:
        split_rows = []
        for row in frame[frame["supervision_split"] == split].to_dict("records"):
            seq, chains, parsed_len = pdb_sequence_and_chains(str(row["receptor_path"]))
            if len(chains) != 1:
                continue
            if not seq or len(seq) > args.max_sequence_length:
                continue
            row["labind_sequence"] = seq
            row["labind_chains"] = ",".join(chains)
            row["labind_sequence_length"] = parsed_len
            split_rows.append(row)
            if len(split_rows) >= per_split:
                break
        selected.extend(split_rows)
    if len(selected) < args.max_samples:
        seen = {row["sample_id"] for row in selected}
        for row in frame.to_dict("records"):
            if row["sample_id"] in seen:
                continue
            seq, chains, parsed_len = pdb_sequence_and_chains(str(row["receptor_path"]))
            if len(chains) != 1:
                continue
            if not seq or len(seq) > args.max_sequence_length:
                continue
            row["labind_sequence"] = seq
            row["labind_chains"] = ",".join(chains)
            row["labind_sequence_length"] = parsed_len
            selected.append(row)
            seen.add(row["sample_id"])
            if len(selected) >= args.max_samples:
                break
    return selected


def write_outputs(rows: list[dict], args: argparse.Namespace, ligand_dict: dict, smiles: dict[str, str]) -> dict:
    out_dir = Path(args.out_dir)
    work = out_dir / "work" / "out"
    pdb_dir = work / "pdb"
    for path in [out_dir, work, pdb_dir, work / "dssp", work / "pos", work / "ankh"]:
        path.mkdir(parents=True, exist_ok=True)

    fasta_lines = []
    smiles_lines = []
    manifest_rows = []
    subset_ligands = {}
    for index, row in enumerate(rows, start=1):
        name = safe_name(index, row)
        ligand = str(row["ligand_ccd_code"]).upper()
        fasta_lines.extend([f">{name} {ligand}", row["labind_sequence"]])
        smiles_lines.append(f"{ligand} {smiles[ligand]}")
        subset_ligands[ligand] = ligand_dict[ligand]
        shutil.copy2(str(row["receptor_path"]), pdb_dir / f"{name}.pdb")
        manifest_rows.append(
            {
                "labind_name": name,
                "sample_id": row["sample_id"],
                "split": row["supervision_split"],
                "pdb_id": row["pdb_id"],
                "ligand_ccd_code": ligand,
                "sequence_length": row["labind_sequence_length"],
                "chain": row["labind_chains"],
                "interaction_pairs": row["interaction_pairs"],
                "positive_site_residues": row["positive_site_residues"],
                "receptor_path": row["receptor_path"],
                "ligand_path": row["ligand_path"],
            }
        )
    (work / "protein.fa").write_text("\n".join(fasta_lines) + "\n", encoding="utf-8")
    (work / "smiles.txt").write_text("\n".join(smiles_lines) + "\n", encoding="utf-8")
    with (work / "ligand.pkl").open("wb") as handle:
        pickle.dump(subset_ligands, handle)
    manifest_path = out_dir / "selected_samples.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    summary = {
        "selected_samples": len(manifest_rows),
        "splits": sorted(set(row["split"] for row in manifest_rows)),
        "ligands": sorted(set(row["ligand_ccd_code"] for row in manifest_rows)),
        "work_dir": str(work),
        "manifest": str(manifest_path),
        "next_required_feature": "ankh embeddings; dssp/pos can be generated with LABind tools",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def write_table(summary: dict, args: argparse.Namespace) -> None:
    lines = [
        "# External subset LABind PLINDER Smoke Prep",
        "",
        f"- selected samples: `{summary['selected_samples']}`",
        f"- splits: `{summary['splits']}`",
        f"- covered LABind ligands: `{summary['ligands']}`",
        f"- work dir: `{summary['work_dir']}`",
        f"- manifest: `{summary['manifest']}`",
        "- status: `prepared_fasta_pdb_smiles_ligand_embeddings; ankh_embeddings_pending`",
        "",
        "This is a preparation artifact only. It does not fabricate missing Ankh features.",
    ]
    path = Path(args.table)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    labind_root = Path(args.labind_root)
    with (labind_root / "tools/ligand.pkl").open("rb") as handle:
        ligand_dict = pickle.load(handle)
    ligand_keys = {str(key).upper() for key in ligand_dict}
    smiles = load_smiles(labind_root / "tools/smiles.txt")
    rows = select_rows(args, ligand_keys, smiles)
    if not rows:
        raise SystemExit("No eligible PLINDER rows found for LABind smoke prep")
    summary = write_outputs(rows, args, ligand_dict, smiles)
    write_table(summary, args)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
