#!/usr/bin/env python3
"""Evaluate LABind PLINDER smoke predictions on frozen PLIP active residues."""

from __future__ import annotations

import argparse
import ast
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-samples",
        default=str(ROOT / "outputs/external_labind_plinder_external_subset_smoke_prep/selected_samples.csv"),
    )
    parser.add_argument(
        "--labind-result",
        default=str(ROOT / "outputs/external_labind_plinder_external_subset_smoke_prep/work/out/RESULT.csv"),
    )
    parser.add_argument(
        "--interaction-labels",
        default=str(ROOT / "data/processed/plinder/plinder_plip_interaction_labels.csv.gz"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "outputs/external_labind_plinder_labind_eval_smoke_eval"),
    )
    parser.add_argument(
        "--table",
        default=str(ROOT / "outputs/tables/table_labind_plinder_labind_eval_smoke_eval.md"),
    )
    return parser.parse_args()


def clean_resnr(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def clean_key(chain: object, residue_number: object, residue_type: object) -> tuple[str, str, str]:
    return (str(chain).strip(), clean_resnr(residue_number), str(residue_type).strip().upper())


def ordered_residues(path: str) -> list[tuple[str, str, str]]:
    out = []
    seen = set()
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
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


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
    denom = int((pred | truth).sum())
    return {
        "graph_ap": safe_metric(average_precision_score, labels, scores),
        "iou_at_p": float((pred & truth).sum() / denom) if denom else math.nan,
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


def build_rows(args: argparse.Namespace) -> pd.DataFrame:
    selected = pd.read_csv(args.selected_samples)
    result = pd.read_csv(args.labind_result, converters={"Binding Site Probability": ast.literal_eval})
    result = result.rename(columns={"Protein Name": "labind_name", "Binding Site Probability": "scores"})
    merged = selected.merge(result[["labind_name", "scores"]], on="labind_name", how="inner")
    positives = positive_residues(args.interaction_labels)
    rows = []
    for row in merged.to_dict("records"):
        residue_keys = ordered_residues(str(row["receptor_path"]))
        scores = [float(value) for value in row["scores"]]
        if len(scores) < len(residue_keys):
            residue_keys = residue_keys[: len(scores)]
        elif len(scores) > len(residue_keys):
            scores = scores[: len(residue_keys)]
        rest = str(row["sample_id"]).split(":", 1)[1]
        system_id, ligand_id = rest.rsplit(":", 1)
        positive = positives.get((system_id, ligand_id), set())
        for key, score in zip(residue_keys, scores):
            rows.append(
                {
                    "labind_name": row["labind_name"],
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "chain": key[0],
                    "residue_number": key[1],
                    "residue_type": key[2],
                    "label": int(key in positive),
                    "score": score,
                    "method": "LABind",
                }
            )
    return pd.DataFrame(rows)


def write_table(summary: dict, path: Path) -> None:
    lines = [
        "# LABind LABind PLINDER Smoke Evaluation",
        "",
        "- protocol: LABind official checkpoint on External subset 20-sample PLINDER smoke subset.",
        "- note: External subset adapter crops DSSP/MSMS arrays to Ankh/FASTA length where PLINDER residue counts differ.",
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
    rows = build_rows(args)
    if rows.empty:
        raise SystemExit("No LABind rows to evaluate")
    by_split = {
        split: summarize(group)
        for split, group in rows.groupby("split", sort=True)
    }
    summary = {"by_split": by_split, "all": summarize(rows)}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_dir / "predictions_labind.csv.gz", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_table(summary, Path(args.table))


if __name__ == "__main__":
    main()
