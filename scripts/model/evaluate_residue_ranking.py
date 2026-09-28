#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader

from scripts.model import train_interaction_scorer as train_visnet


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINABLE_CLASSES = [0, 1, 3, 4, 5]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate residue-level interaction-site localization from pair scores."
    )
    parser.add_argument(
        "--graph-dir",
        default=str(
            ROOT / "data/processed/structure_supervision/interaction_graphs_hardneg_full_raw_v2"
        ),
    )
    parser.add_argument("--model", choices=["distance", "visnet"], default="distance")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["internal_val", "external_val", "external_test"],
        choices=list(train_visnet.SPLIT_ORDER),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-loader-workers", type=int, default=0)
    parser.add_argument("--finite-representation-guard", action="store_true")
    parser.add_argument("--representation-clip", type=float, default=0.0)
    return parser.parse_args()


def safe_metric(fn, y_true, score):
    y_true = np.asarray(y_true, dtype=np.int32)
    score = np.asarray(score, dtype=np.float64)
    if y_true.size == 0 or y_true.min() == y_true.max():
        return math.nan
    return float(fn(y_true, score))


def describe(values):
    clean = np.asarray([value for value in values if not math.isnan(value)], dtype=np.float64)
    if clean.size == 0:
        return {"n": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "n": int(clean.size),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def load_visnet(checkpoint_path, args, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    model = train_visnet.PairViSNet(
        hidden=int(ckpt_args.get("hidden", 64)),
        layers=int(ckpt_args.get("layers", 3)),
        heads=int(ckpt_args.get("heads", 8)),
        num_rbf=int(ckpt_args.get("num_rbf", 16)),
        cutoff=float(ckpt_args.get("cutoff", 5.0)),
        max_neighbors=int(ckpt_args.get("max_neighbors", 64)),
        dropout=float(ckpt_args.get("dropout", 0.10)),
        finite_representation_guard=args.finite_representation_guard,
        representation_clip=args.representation_clip,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    trainable_classes = checkpoint.get("trainable_classes", DEFAULT_TRAINABLE_CLASSES)
    return model, [int(index) for index in trainable_classes], checkpoint.get("history", [])


def pair_scores(batch, model, active_classes, mode):
    protein_index, ligand_index = batch.pair_index
    labels = batch.y[:, active_classes]
    masks = batch.loss_mask[:, active_classes]
    pair_labels = ((labels * masks).sum(dim=1) > 0).detach().cpu().numpy().astype(np.int32)

    if mode == "distance":
        distance = torch.linalg.norm(batch.pos[protein_index] - batch.pos[ligand_index], dim=1)
        scores = (-distance).detach().cpu().numpy().astype(np.float64)
        return scores, pair_labels

    logits = model(batch)
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite logits in residue-localization evaluation")
    probs = torch.sigmoid(logits[:, active_classes])
    masked_probs = probs.masked_fill(masks <= 0, float("-inf"))
    scores = masked_probs.max(dim=1).values.detach().cpu().numpy().astype(np.float64)
    scores[~np.isfinite(scores)] = 0.0
    return scores, pair_labels


def sample_ids_from_batch(batch):
    sample_ids = batch.sample_id
    if isinstance(sample_ids, (list, tuple)):
        return list(sample_ids)
    return [str(sample_ids)]


def aggregate_residues(batch, scores, labels):
    protein_nodes = batch.pair_index[0].detach().cpu().numpy().astype(np.int64)
    graph_ids = batch.batch[batch.pair_index[0]].detach().cpu().numpy().astype(np.int64)
    sample_ids = sample_ids_from_batch(batch)
    residue_scores = {}
    residue_labels = {}
    residue_graphs = {}
    for node, graph_id, score, label in zip(protein_nodes, graph_ids, scores, labels):
        node = int(node)
        graph_id = int(graph_id)
        key = (graph_id, node)
        residue_scores[key] = max(float(score), residue_scores.get(key, -math.inf))
        residue_labels[key] = max(int(label), residue_labels.get(key, 0))
        residue_graphs[key] = sample_ids[graph_id]
    rows = []
    for key in residue_scores:
        rows.append(
            {
                "sample_id": residue_graphs[key],
                "score": residue_scores[key],
                "label": residue_labels[key],
            }
        )
    return rows


def summarize_split(rows):
    labels = np.asarray([row["label"] for row in rows], dtype=np.int32)
    scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    graph_groups = defaultdict(list)
    for row in rows:
        graph_groups[row["sample_id"]].append(row)
    graph_auprc = []
    graph_auroc = []
    graph_positive_rates = []
    for group in graph_groups.values():
        y = np.asarray([row["label"] for row in group], dtype=np.int32)
        s = np.asarray([row["score"] for row in group], dtype=np.float64)
        graph_positive_rates.append(float(y.mean()))
        if y.min() == y.max():
            continue
        graph_auprc.append(float(average_precision_score(y, s)))
        graph_auroc.append(float(roc_auc_score(y, s)))
    return {
        "graphs": int(len(graph_groups)),
        "residues": int(labels.size),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()) if labels.size else math.nan,
        "auprc": safe_metric(average_precision_score, labels, scores),
        "auroc": safe_metric(roc_auc_score, labels, scores),
        "mean_graph_auprc": describe(graph_auprc),
        "mean_graph_auroc": describe(graph_auroc),
        "mean_graph_positive_rate": describe(graph_positive_rates),
    }


@torch.no_grad()
def evaluate_split(graphs, args, model, active_classes, device):
    loader = DataLoader(
        graphs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_loader_workers,
        pin_memory=device.type == "cuda",
    )
    rows = []
    for batch in loader:
        batch = batch.to(device)
        scores, labels = pair_scores(batch, model, active_classes, args.model)
        rows.extend(aggregate_residues(batch, scores, labels))
    return rows, summarize_split(rows)


def write_table(metrics, path):
    lines = [
        "# Interaction-to-Residue Localization",
        "",
        f"- model: `{metrics['model']}`",
        f"- checkpoint: `{metrics.get('checkpoint') or 'none'}`",
        f"- finite representation guard: `{str(metrics['finite_representation_guard']).lower()}`",
        "",
        "| split | graphs | residues | positives | prevalence | residue AUPRC | residue AUROC | mean graph AUPRC | mean graph AUROC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in metrics["splits"]:
        row = metrics["splits"][split]
        lines.append(
            f"| {split} | {row['graphs']} | {row['residues']} | {row['positives']} | "
            f"{row['prevalence']:.4f} | {row['auprc']:.4f} | {row['auroc']:.4f} | "
            f"{row['mean_graph_auprc']['mean']:.4f} | {row['mean_graph_auroc']['mean']:.4f} |"
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.model == "visnet":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --model visnet")
        model, active_classes, history = load_visnet(args.checkpoint, args, device)
    else:
        model = None
        active_classes = DEFAULT_TRAINABLE_CLASSES
        history = []

    split_metrics = {}
    for split in args.splits:
        graphs = train_visnet.load_graphs(args.graph_dir, split)
        _, split_metrics[split] = evaluate_split(graphs, args, model, active_classes, device)

    metrics = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "graph_dir": args.graph_dir,
        "splits_requested": args.splits,
        "batch_size": args.batch_size,
        "device": str(device),
        "finite_representation_guard": bool(args.finite_representation_guard),
        "representation_clip": float(args.representation_clip),
        "active_interaction_types": [train_visnet.INTERACTION_TYPES[index] for index in active_classes],
        "history": history,
        "splits": split_metrics,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_table(metrics, args.table)
    print(json.dumps({"metrics": str(metrics_path), "table": args.table}))


if __name__ == "__main__":
    main()
