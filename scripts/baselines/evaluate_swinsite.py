#!/usr/bin/env python3
"""Evaluate SwinSite PLINDER smoke predictions on frozen PLIP active residues."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_INTERACTIONS = {
    "hydrophobic_contact",
    "hydrogen_bond",
    "salt_bridge",
    "pi_stacking",
    "pi_cation",
}
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
SCORE_RE = re.compile(r"_score_([0-9]+(?:\.[0-9]+)?)\.mol2$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-samples",
        default=str(ROOT / "outputs/external_swinsite_smoke/selected_samples_swinsite.csv"),
    )
    parser.add_argument(
        "--prediction-dir",
        default=str(ROOT / "outputs/external_swinsite_smoke/work/predictions/input"),
    )
    parser.add_argument(
        "--interaction-labels",
        default=str(ROOT / "data/processed/plinder/plinder_plip_interaction_labels.csv.gz"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "outputs/external_swinsite_smoke_eval"),
    )
    parser.add_argument(
        "--table",
        default=str(ROOT / "outputs/tables/table_swinsite_plinder_swinsite_smoke_eval.md"),
    )
    parser.add_argument("--coord-tolerance", type=float, default=0.35)
    return parser.parse_args()


def clean_resnr(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def clean_key(chain: object, residue_number: object, residue_type: object) -> tuple[str, str, str]:
    return (str(chain).strip(), clean_resnr(residue_number), str(residue_type).strip().upper())


def is_hydrogen(line: str) -> bool:
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    atom_name = line[12:16].strip().upper()
    return element == "H" or atom_name.startswith("H")


def protein_atoms(path: str) -> tuple[list[tuple[str, str, str]], np.ndarray, np.ndarray]:
    residues = []
    residue_index = {}
    atom_coords = []
    atom_residue_idx = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM  "):
                continue
            resname = line[17:20].strip().upper()
            if resname not in AA3:
                continue
            chain = line[21:22].strip()
            resnr = line[22:26].strip() + line[26:27].strip()
            key = (chain, resnr, resname)
            if key not in residue_index:
                residue_index[key] = len(residues)
                residues.append(key)
            if is_hydrogen(line):
                continue
            try:
                coord = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            atom_coords.append(coord)
            atom_residue_idx.append(residue_index[key])
    coords = np.asarray(atom_coords, dtype=np.float64)
    indices = np.asarray(atom_residue_idx, dtype=np.int64)
    return residues, coords, indices


def mol2_atom_coords(path: Path) -> np.ndarray:
    coords = []
    in_atom = False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom = True
                continue
            if line.startswith("@<TRIPOS>") and in_atom:
                break
            if not in_atom:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                coords.append((float(parts[2]), float(parts[3]), float(parts[4])))
            except ValueError:
                continue
    return np.asarray(coords, dtype=np.float64)


def score_from_name(path: Path) -> float:
    match = SCORE_RE.search(path.name)
    return float(match.group(1)) if match else 0.0


def positive_residues(path: str) -> dict[tuple[str, str], set[tuple[str, str, str]]]:
    frame = pd.read_csv(path)
    positives: dict[tuple[str, str], set[tuple[str, str, str]]] = defaultdict(set)
    for row in frame.to_dict("records"):
        if str(row["interaction_type"]) not in ACTIVE_INTERACTIONS:
            continue
        positives[(str(row["system_id"]), str(row["ligand_id"]))].add(
            clean_key(row["protein_chain"], row["protein_residue_number"], row["protein_residue_type"])
        )
    return positives


def safe_metric(fn, labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or labels.min() == labels.max():
        return math.nan
    return float(fn(labels, scores))


def graph_metrics(group: pd.DataFrame) -> dict[str, float]:
    labels = group["label"].to_numpy(dtype=np.int32)
    scores = group["score"].to_numpy(dtype=np.float64)
    positives = int(labels.sum())
    if positives <= 0:
        return {"graph_ap": math.nan, "iou_at_p": math.nan, "recall_at_10": math.nan}
    order = np.argsort(-scores, kind="stable")
    pred = np.zeros_like(labels, dtype=bool)
    pred[order[:positives]] = True
    truth = labels.astype(bool)
    union = int((pred | truth).sum())
    return {
        "graph_ap": safe_metric(average_precision_score, labels, scores),
        "iou_at_p": float((pred & truth).sum() / union) if union else math.nan,
        "recall_at_10": float(labels[order[: min(10, labels.size)]].sum() / positives),
    }


def summarize(rows: pd.DataFrame) -> dict[str, float]:
    labels = rows["label"].to_numpy(dtype=np.int32)
    scores = rows["score"].to_numpy(dtype=np.float64)
    per_graph = [graph_metrics(group) for _, group in rows.groupby("sample_id", sort=False)]
    return {
        "graphs": int(rows["sample_id"].nunique()),
        "residues": int(len(rows)),
        "positives": int(labels.sum()),
        "pooled_auprc": safe_metric(average_precision_score, labels, scores),
        "pooled_auroc": safe_metric(roc_auc_score, labels, scores),
        "graph_ap": float(np.nanmean([row["graph_ap"] for row in per_graph])),
        "iou_at_p": float(np.nanmean([row["iou_at_p"] for row in per_graph])),
        "recall_at_10": float(np.nanmean([row["recall_at_10"] for row in per_graph])),
    }


def swinsite_residue_scores(
    sample_dir: Path,
    residues: list[tuple[str, str, str]],
    coords: np.ndarray,
    atom_residue_idx: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, int]]:
    scores = np.zeros(len(residues), dtype=np.float64)
    stats = {
        "pocket_files": 0,
        "pocket_atoms": 0,
        "matched_atoms": 0,
        "unmatched_atoms": 0,
    }
    if coords.size == 0:
        return scores, stats
    tree = cKDTree(coords)
    for pocket_path in sorted(sample_dir.glob("pocket*_score_*.mol2")):
        pocket_coords = mol2_atom_coords(pocket_path)
        if pocket_coords.size == 0:
            continue
        score = score_from_name(pocket_path)
        stats["pocket_files"] += 1
        stats["pocket_atoms"] += int(len(pocket_coords))
        distances, atom_indices = tree.query(pocket_coords, k=1)
        matched_mask = distances <= tolerance
        stats["matched_atoms"] += int(matched_mask.sum())
        stats["unmatched_atoms"] += int((~matched_mask).sum())
        for atom_index in atom_indices[matched_mask]:
            residue_idx = int(atom_residue_idx[int(atom_index)])
            scores[residue_idx] = max(scores[residue_idx], score)
    return scores, stats


def build_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    selected = pd.read_csv(args.selected_samples)
    positives = positive_residues(args.interaction_labels)
    prediction_dir = Path(args.prediction_dir)
    rows = []
    missing = []
    totals = {
        "pocket_files": 0,
        "pocket_atoms": 0,
        "matched_atoms": 0,
        "unmatched_atoms": 0,
    }

    for row in selected.to_dict("records"):
        residues, coords, atom_residue_idx = protein_atoms(str(row["swinsite_pdb_path"]))
        sample_dir = prediction_dir / str(row["swinsite_name"])
        if not sample_dir.exists():
            missing.append(str(row["swinsite_name"]))
        scores, stats = swinsite_residue_scores(
            sample_dir,
            residues,
            coords,
            atom_residue_idx,
            args.coord_tolerance,
        )
        for key, value in stats.items():
            totals[key] += int(value)
        rest = str(row["sample_id"]).split(":", 1)[1]
        system_id, ligand_id = rest.rsplit(":", 1)
        positive = positives.get((system_id, ligand_id), set())
        for key, score in zip(residues, scores):
            rows.append(
                {
                    "swinsite_name": row["swinsite_name"],
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "chain": key[0],
                    "residue_number": key[1],
                    "residue_type": key[2],
                    "label": int(key in positive),
                    "score": float(score),
                    "method": "SwinSite",
                }
            )
    return pd.DataFrame(rows), missing, totals


def write_table(summary: dict, path: Path) -> None:
    lines = [
        "# SwinSite SwinSite PLINDER Smoke Evaluation",
        "",
        "- protocol: SwinSite official 4-fold checkpoint on External subset 20-sample PLINDER smoke subset.",
        "- score mapping: residue score is the maximum score among predicted SwinSite pocket files whose atoms map back to that residue by coordinate nearest-neighbor matching.",
        f"- coordinate tolerance: `{summary['coord_tolerance']}` Angstrom",
        f"- missing prediction dirs: `{summary['missing_predictions']}`",
        f"- matched pocket atoms: `{summary['mapping_totals']['matched_atoms']}` / `{summary['mapping_totals']['pocket_atoms']}`",
        "",
        "| split | graphs | residues | positives | pooled AUPRC | pooled AUROC | graph AP | IoU@P | recall@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, values in summary["by_split"].items():
        lines.append(
            "| {split} | {graphs} | {residues} | {positives} | {pooled_auprc:.4f} | {pooled_auroc:.4f} | {graph_ap:.4f} | {iou_at_p:.4f} | {recall_at_10:.4f} |".format(
                split=split,
                **values,
            )
        )
    lines.append(
        "| all | {graphs} | {residues} | {positives} | {pooled_auprc:.4f} | {pooled_auroc:.4f} | {graph_ap:.4f} | {iou_at_p:.4f} | {recall_at_10:.4f} |".format(
            **summary["all"]
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows, missing, totals = build_rows(args)
    if rows.empty:
        raise SystemExit("No SwinSite rows to evaluate")
    by_split = {split: summarize(group) for split, group in rows.groupby("split", sort=True)}
    summary = {
        "by_split": by_split,
        "all": summarize(rows),
        "missing_predictions": len(missing),
        "missing_prediction_names": missing,
        "mapping_totals": totals,
        "coord_tolerance": args.coord_tolerance,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_dir / "predictions_swinsite.csv.gz", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_table(summary, Path(args.table))


if __name__ == "__main__":
    main()
