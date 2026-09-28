#!/usr/bin/env python3
"""Run a CLAPE-SMB sequence-only smoke baseline on the External subset PLINDER subset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_INTERACTIONS = {
    "hydrophobic_contact",
    "hydrogen_bond",
    "salt_bridge",
    "pi_stacking",
    "pi_cation",
}
AA3_TO_1 = {
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
        "--selected-samples",
        default=str(ROOT / "outputs/external_labind_plinder_external_subset_smoke_prep/selected_samples.csv"),
    )
    parser.add_argument(
        "--interaction-labels",
        default=str(ROOT / "data/processed/plinder/plinder_plip_interaction_labels.csv.gz"),
    )
    parser.add_argument(
        "--clape-root",
        default=str(ROOT / ".tools/deep_baselines/src/CLAPE-SMB"),
    )
    parser.add_argument(
        "--checkpoint",
        default="Models/SJC/random_seed/42.ckpt",
        help="Checkpoint path relative to CLAPE-SMB root.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "outputs/external_clape_smoke_eval"),
    )
    parser.add_argument(
        "--prep-table",
        default=str(ROOT / "outputs/tables/table_clape_plinder_clape_smoke_prep.md"),
    )
    parser.add_argument(
        "--eval-table",
        default=str(ROOT / "outputs/tables/table_clape_plinder_clape_smoke_eval.md"),
    )
    parser.add_argument(
        "--comparison-table",
        default=str(ROOT / "outputs/tables/table_clape_plinder_clape_smoke_comparison.md"),
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional sample cap for probe runs.")
    parser.add_argument("--max-length", type=int, default=1022, help="ESM2 practical sequence length limit.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def clean_resnr(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def clean_key(chain: object, residue_number: object, residue_type: object) -> tuple[str, str, str]:
    return (str(chain).strip(), clean_resnr(residue_number), str(residue_type).strip().upper())


def pdb_sequence(path: str) -> tuple[list[tuple[str, str, str]], str]:
    residues = []
    seen = set()
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM  "):
                continue
            resname = line[17:20].strip().upper()
            if resname not in AA3_TO_1:
                continue
            chain = line[21:22].strip()
            resnr = line[22:26].strip() + line[26:27].strip()
            key = (chain, resnr, resname)
            if key in seen:
                continue
            seen.add(key)
            residues.append(key)
    sequence = "".join(AA3_TO_1[key[2]] for key in residues)
    return residues, sequence


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


def summarize(frame: pd.DataFrame) -> dict[str, float]:
    labels = frame["label"].to_numpy(dtype=np.int32)
    scores = frame["score"].to_numpy(dtype=np.float64)
    per_graph = [graph_metrics(group) for _, group in frame.groupby("sample_id", sort=False)]
    return {
        "graphs": int(frame["sample_id"].nunique()),
        "residues": int(len(frame)),
        "positives": int(labels.sum()),
        "pooled_auprc": safe_metric(average_precision_score, labels, scores),
        "pooled_auroc": safe_metric(roc_auc_score, labels, scores),
        "graph_ap": float(np.nanmean([row["graph_ap"] for row in per_graph])),
        "iou_at_p": float(np.nanmean([row["iou_at_p"] for row in per_graph])),
        "recall_at_10": float(np.nanmean([row["recall_at_10"] for row in per_graph])),
    }


def load_clape(args: argparse.Namespace):
    clape_root = Path(args.clape_root)
    sys.path.insert(0, str(clape_root))
    import esm  # noqa: PLC0415
    from model import ContinueModel  # noqa: PLC0415

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model.to(device)
    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()

    predictor = ContinueModel()
    checkpoint = clape_root / args.checkpoint
    predictor.load_state_dict(torch.load(checkpoint, map_location=device))
    predictor.to(device)
    predictor.eval()
    return esm_model, batch_converter, predictor, device


def clape_scores(esm_model, batch_converter, predictor, device, sample_id: str, sequence: str) -> np.ndarray:
    batch = [(sample_id, sequence)]
    _, _, batch_tokens = batch_converter(batch)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        features = results["representations"][33].squeeze(0)[1:-1, :].unsqueeze(0)
        scores = predictor(features)[0].squeeze(0)[:, 1]
    return scores.detach().cpu().numpy().astype(np.float64)


def build_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    selected = pd.read_csv(args.selected_samples)
    if args.limit > 0:
        selected = selected.head(args.limit).copy()
    positives = positive_residues(args.interaction_labels)
    esm_model, batch_converter, predictor, device = load_clape(args)

    rows = []
    manifest_rows = []
    skipped = []
    for row in selected.to_dict("records"):
        residues, sequence = pdb_sequence(str(row["receptor_path"]))
        if len(sequence) > args.max_length:
            skipped.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "sequence_length": int(len(sequence)),
                    "reason": f"exceeds ESM2 max length {args.max_length}",
                }
            )
            continue
        scores = clape_scores(esm_model, batch_converter, predictor, device, str(row["sample_id"]), sequence)
        if len(scores) != len(residues):
            raise ValueError(f"{row['sample_id']}: score length {len(scores)} != residues {len(residues)}")
        rest = str(row["sample_id"]).split(":", 1)[1]
        system_id, ligand_id = rest.rsplit(":", 1)
        positive = positives.get((system_id, ligand_id), set())
        for key, score in zip(residues, scores):
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "chain": key[0],
                    "residue_number": key[1],
                    "residue_type": key[2],
                    "label": int(key in positive),
                    "score": float(score),
                    "method": "CLAPE-SMB",
                }
            )
        current = dict(row)
        current["clape_sequence_length"] = len(sequence)
        manifest_rows.append(current)
    return pd.DataFrame(rows), pd.DataFrame(manifest_rows), skipped


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(array)),
        "std": float(np.nanstd(array, ddof=1)) if len(array) > 1 else 0.0,
        "n": int(len(array)),
    }


def rows_from_csv(path: Path, selected_ids: set[str], method: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["sample_id"].isin(selected_ids)].copy()
    frame["method"] = method
    return frame


def optional_rows(path: Path, selected_ids: set[str], method: str) -> list[pd.DataFrame]:
    if not path.exists():
        return []
    frame = rows_from_csv(path, selected_ids, method)
    return [] if frame.empty else [frame]


def laign_runs(selected_ids: set[str]) -> list[pd.DataFrame]:
    runs = []
    for seed in [2401, 2402, 2403]:
        archive = ROOT / f"outputs/stage2_statistics_statistics_laign_full_seed{seed}/predictions.npz"
        payload = np.load(archive, allow_pickle=True)
        rows = []
        for split in ["external_val", "external_test"]:
            labels = payload[f"{split}__labels"].astype(np.int32)
            scores = payload[f"{split}__score__trained_stage2"].astype(np.float64)
            sample_index = payload[f"{split}__sample_index"].astype(np.int64)
            sample_ids = [str(value) for value in payload[f"{split}__sample_ids"]]
            for graph_index, sample_id in enumerate(sample_ids):
                if sample_id not in selected_ids:
                    continue
                idx = np.flatnonzero(sample_index == graph_index)
                for row_idx in idx:
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "split": split,
                            "label": int(labels[row_idx]),
                            "score": float(scores[row_idx]),
                            "method": "laign",
                        }
                    )
        runs.append(pd.DataFrame(rows))
    return runs


def comparison_summary(clape_rows: pd.DataFrame) -> dict:
    selected_ids = set(clape_rows["sample_id"])
    method_runs = {
        "fpocket": [
            rows_from_csv(ROOT / "outputs/external_pocket_baselines_pocket_baseline_full/predictions_fpocket.csv.gz", selected_ids, "fpocket")
        ],
        "p2rank": [
            rows_from_csv(ROOT / "outputs/external_pocket_baselines_pocket_baseline_full/predictions_p2rank.csv.gz", selected_ids, "p2rank")
        ],
        "prolif": [
            rows_from_csv(ROOT / "outputs/external_prolif_csd_full/predictions_prolif.csv.gz", selected_ids, "prolif")
        ],
        "LABind": optional_rows(ROOT / "outputs/external_labind_plinder_labind_eval_smoke_eval/predictions_labind.csv.gz", selected_ids, "LABind"),
        "UniSite": optional_rows(ROOT / "outputs/external_unisite_plinder_unisite_smoke_eval/predictions_unisite.csv.gz", selected_ids, "UniSite"),
        "SwinSite": optional_rows(ROOT / "outputs/external_swinsite_smoke_eval/predictions_swinsite.csv.gz", selected_ids, "SwinSite"),
        "GrASP": optional_rows(ROOT / "outputs/external_grasp_smoke_eval/predictions_grasp.csv.gz", selected_ids, "GrASP"),
        "CLAPE-SMB": [clape_rows],
        "laign": laign_runs(selected_ids),
    }
    metrics = ["pooled_auprc", "pooled_auroc", "graph_ap", "iou_at_p", "recall_at_10"]
    summary = {}
    for method, runs in method_runs.items():
        if not runs:
            continue
        summary[method] = {}
        for split in ["external_val", "external_test", "all"]:
            values = []
            for frame in runs:
                current = frame if split == "all" else frame[frame["split"] == split]
                if not current.empty:
                    values.append(summarize(current))
            if not values:
                continue
            summary[method][split] = {key: describe([row[key] for row in values]) for key in metrics}
            summary[method][split]["graphs"] = int(values[0]["graphs"])
            summary[method][split]["residues"] = int(values[0]["residues"])
            summary[method][split]["positives"] = int(values[0]["positives"])
    return summary


def fmt_summary(row: dict, key: str) -> str:
    return f"{row[key]['mean']:.4f} +/- {row[key]['std']:.4f}"


def write_prep_table(manifest: pd.DataFrame, args: argparse.Namespace) -> None:
    path = Path(args.prep_table)
    lines = [
        "# CLAPE-SMB CLAPE-SMB PLINDER Smoke Prep",
        "",
        f"- runnable samples: `{len(manifest)}`",
        f"- splits: `{sorted(manifest['split'].astype(str).unique())}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- max sequence length: `{args.max_length}`",
        "- protocol: same External subset single-chain PLINDER subset; sequence is extracted from standard receptor PDB residues in residue order.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_eval_table(summary: dict, args: argparse.Namespace) -> None:
    path = Path(args.eval_table)
    lines = [
        "# CLAPE-SMB CLAPE-SMB PLINDER Smoke Evaluation",
        "",
        "- protocol: CLAPE-SMB official sequence-only checkpoint on External subset PLINDER smoke subset.",
        f"- checkpoint: `{args.checkpoint}`",
        f"- runnable/skipped samples: `{summary['selected_samples']}` / `{summary['skipped_samples']}`",
        "- score mapping: CLAPE sequence-position probability is mapped to the corresponding standard receptor residue by PDB residue order.",
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


def write_comparison_table(summary: dict, args: argparse.Namespace) -> None:
    path = Path(args.comparison_table)
    lines = [
        "# CLAPE-SMB CLAPE-SMB PLINDER Smoke Subset Comparison",
        "",
        "- source subset: External subset 20-sample PLINDER smoke set; LAIGN reports three-seed mean/std.",
        "- comparison rows use the CLAPE-compatible subset after excluding proteins longer than ESM2's practical sequence length limit.",
        "- external methods are self-rerun or smoke-rerun under the frozen PLIP residue protocol.",
        "- CLAPE-SMB uses the official sequence-only SJC random-seed 42 checkpoint from the repository inference default.",
        "",
        "| split | method | runs | graphs | pooled AUPRC | pooled AUROC | graph AP | IoU@P | recall@10 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["fpocket", "p2rank", "prolif", "LABind", "UniSite", "SwinSite", "GrASP", "CLAPE-SMB", "laign"]
    for split in ["external_val", "external_test", "all"]:
        for method in order:
            row = summary.get(method, {}).get(split)
            if not row:
                continue
            lines.append(
                f"| {split} | {method} | {row['pooled_auprc']['n']} | {row['graphs']} | "
                f"{fmt_summary(row, 'pooled_auprc')} | {fmt_summary(row, 'pooled_auroc')} | "
                f"{fmt_summary(row, 'graph_ap')} | {fmt_summary(row, 'iou_at_p')} | "
                f"{fmt_summary(row, 'recall_at_10')} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows, manifest, skipped = build_rows(args)
    if rows.empty:
        raise SystemExit("No CLAPE-compatible rows to evaluate")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions_clape_smb.csv.gz"
    manifest_path = out_dir / "selected_samples_clape.csv"
    skipped_path = out_dir / "skipped_samples_clape.csv"
    rows.to_csv(predictions_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    pd.DataFrame(skipped).to_csv(skipped_path, index=False)
    by_split = {split: summarize(group) for split, group in rows.groupby("split", sort=True)}
    summary = {
        "by_split": by_split,
        "all": summarize(rows),
        "checkpoint": args.checkpoint,
        "selected_samples": int(manifest["sample_id"].nunique()),
        "skipped_samples": int(len(skipped)),
        "skipped": skipped,
        "manifest": str(manifest_path),
        "skipped_manifest": str(skipped_path),
        "predictions": str(predictions_path),
        "device": str(args.device),
        "max_length": int(args.max_length),
    }
    comparison = comparison_summary(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "comparison_summary.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    write_prep_table(manifest, args)
    write_eval_table(summary, args)
    write_comparison_table(comparison, args)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
