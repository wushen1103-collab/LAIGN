#!/usr/bin/env python3
"""Run ProLIF interaction-count baseline on accepted Platinum complexes."""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import MDAnalysis as mda
import pandas as pd
import prolif as plf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="experiments/validation/data/processed/platinum/platinum_manifest.csv.gz",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/validation/results/platinum_prolif",
    )
    parser.add_argument("--workers", type=int, default=48)
    return parser.parse_args()


def clean(value: object) -> str:
    value = str(value).strip()
    return "" if value.lower() == "nan" else value


def shifted_receptor(source: str, target: Path, offset: int = 1000) -> Path:
    lines = []
    for line in Path(source).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            try:
                number = int(line[22:26]) + offset
                line = f"{line[:22]}{number:4d}{line[26:]}"
            except ValueError:
                pass
        lines.append(line)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def compute_fingerprint(
    row: dict[str, object], receptor_path: str, residue_offset: int = 0
) -> tuple[dict[tuple[str, str, str], float], dict[str, int]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        receptor_u = mda.Universe(receptor_path)
        ligand_u = mda.Universe(str(row["ligand_path"]))
        protein = plf.Molecule.from_mda(
            receptor_u.select_atoms("protein"), NoImplicit=False, force=True
        )
        ligand = plf.Molecule.from_mda(
            ligand_u.atoms, NoImplicit=False, force=True
        )
        fingerprint = plf.Fingerprint(
            [
                "Hydrophobic",
                "PiStacking",
                "Anionic",
                "Cationic",
                "CationPi",
                "PiCation",
                "VdWContact",
            ],
            parameters={"VdWContact": {"preset": "csd"}},
            count=True,
            implicit_hydrogens=True,
            vicinity_cutoff=8.0,
        )
        ifp = fingerprint.generate(ligand, protein, residues=None, metadata=True)
    scores: dict[tuple[str, str, str], float] = defaultdict(float)
    type_counts: dict[str, int] = defaultdict(int)
    for interaction in ifp.interactions():
        residue = interaction.protein
        number = int(residue.number) - residue_offset
        key = (clean(residue.chain), str(number), clean(residue.name).upper())
        scores[key] += 1.0
        type_counts[str(interaction.interaction)] += 1
    return scores, type_counts


def run_one(row: dict[str, object], raw_dir: str) -> dict[str, object]:
    sample_id = str(row["sample_id"])
    slug = sample_id.replace(":", "_").replace("/", "_")
    output = Path(raw_dir) / f"{slug}.json"
    if output.exists():
        cached = json.loads(output.read_text(encoding="utf-8"))
        if cached.get("ok"):
            return cached
    try:
        try:
            scores, type_counts = compute_fingerprint(
                row, str(row["receptor_path"]), residue_offset=0
            )
        except OverflowError:
            shifted = shifted_receptor(
                str(row["receptor_path"]), output.with_suffix(".shifted.pdb")
            )
            scores, type_counts = compute_fingerprint(
                row, str(shifted), residue_offset=1000
            )
        result = {
            "sample_id": sample_id,
            "ok": True,
            "scores": {"|".join(key): value for key, value in scores.items()},
            "type_counts": dict(sorted(type_counts.items())),
        }
    except Exception as exc:
        result = {"sample_id": sample_id, "ok": False, "error": repr(exc)}
    output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, low_memory=False)
    records = manifest.sort_values("sample_id").to_dict("records")
    outputs: dict[str, dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(run_one, row, str(raw_dir)): str(row["sample_id"])
            for row in records
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                outputs[sample_id] = future.result()
            except Exception as exc:
                outputs[sample_id] = {
                    "sample_id": sample_id,
                    "ok": False,
                    "error": repr(exc),
                }

    rows = []
    failures = []
    for record in records:
        sample_id = str(record["sample_id"])
        result = outputs[sample_id]
        if not result.get("ok"):
            failures.append(
                {"sample_id": sample_id, "error": str(result.get("error", "unknown"))}
            )
            continue
        for key, score in result["scores"].items():
            chain, residue_number, residue_type = key.split("|", 2)
            rows.append(
                {
                    "sample_id": sample_id,
                    "protein_chain": chain,
                    "protein_residue_number": residue_number,
                    "protein_residue_type": residue_type,
                    "prolif_count": float(score),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "platinum_prolif_counts.csv.gz", index=False)
    pd.DataFrame(failures).to_csv(out_dir / "platinum_prolif_failures.csv", index=False)
    summary = {
        "requested_complexes": len(records),
        "successful_complexes": len(records) - len(failures),
        "failed_complexes": len(failures),
        "residues_with_interactions": len(rows),
        "fingerprint": [
            "Hydrophobic",
            "PiStacking",
            "Anionic",
            "Cationic",
            "CationPi",
            "PiCation",
            "VdWContact (CSD preset)",
        ],
        "failures": failures,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
