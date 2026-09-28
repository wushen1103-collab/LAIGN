#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from scripts.model import train_interaction_scorer as train


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a raw-coordinate ViSNet checkpoint on graph splits."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--graph-dir",
        default=str(
            ROOT / "data/processed/structure_supervision/interaction_graphs_hardneg_full_raw_v2"
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-loader-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--amp-dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--finite-representation-guard",
        action="store_true",
        help="Replace non-finite ViSNet scalar embeddings before pair decoding.",
    )
    parser.add_argument("--representation-clip", type=float, default=0.0)
    return parser.parse_args()


def checkpoint_value(checkpoint_args, name, default):
    return checkpoint_args.get(name, default)


def write_table(metrics, path):
    lines = [
        "# Raw-Coordinate ViSNet Checkpoint Evaluation",
        "",
        f"- graph dataset: `{metrics['graph_dir']}`",
        f"- checkpoint: `{metrics['checkpoint']}`",
        f"- finite representation guard: `{str(metrics['finite_representation_guard']).lower()}`",
        f"- representation clip: {metrics['representation_clip']}",
        "",
        "| split | graphs | pairs | micro_AUPRC | prevalence | macro_AUPRC | micro_AUROC | macro_AUROC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in train.SPLIT_ORDER:
        row = metrics["splits"][split]
        lines.append(
            f"| {split} | {row['graphs']} | {row['pairs']} | "
            f"{row['micro_auprc']:.4f} | {row['micro_prevalence_baseline']:.4f} | "
            f"{row['macro_auprc']:.4f} | {row['micro_auroc']:.4f} | "
            f"{row['macro_auroc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## External-Test Interaction Types",
            "",
            "| interaction_type | pairs | positives | prevalence | AUPRC | AUROC | F1@0.5 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in metrics["splits"]["external_test"]["classes"].items():
        if not row["evaluated"]:
            continue
        lines.append(
            f"| {name} | {row['pairs']} | {row['positives']} | "
            f"{row['prevalence']:.4f} | {row['auprc']:.4f} | "
            f"{row['auroc']:.4f} | {row['f1_at_0_5']:.4f} |"
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    device_name = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    model = train.PairViSNet(
        hidden=checkpoint_value(checkpoint_args, "hidden", 64),
        layers=checkpoint_value(checkpoint_args, "layers", 3),
        heads=checkpoint_value(checkpoint_args, "heads", 8),
        num_rbf=checkpoint_value(checkpoint_args, "num_rbf", 16),
        cutoff=checkpoint_value(checkpoint_args, "cutoff", 5.0),
        max_neighbors=checkpoint_value(checkpoint_args, "max_neighbors", 64),
        dropout=checkpoint_value(checkpoint_args, "dropout", 0.10),
        finite_representation_guard=args.finite_representation_guard,
        representation_clip=args.representation_clip,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    trainable_classes = checkpoint.get("trainable_classes")
    if trainable_classes is None:
        trainable_classes = list(range(len(train.INTERACTION_TYPES)))

    graphs = {split: train.load_graphs(args.graph_dir, split) for split in train.SPLIT_ORDER}
    loaders = {
        split: DataLoader(
            split_graphs,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_loader_workers,
            pin_memory=device.type == "cuda",
        )
        for split, split_graphs in graphs.items()
    }

    split_metrics = {}
    for split in train.SPLIT_ORDER:
        probability, labels, masks = train.predict(
            model,
            loaders[split],
            device,
            use_amp,
            amp_dtype,
        )
        row = train.evaluate_arrays(probability, labels, masks, trainable_classes)
        row["graphs"] = len(graphs[split])
        row["pairs"] = int(sum(graph.y.shape[0] for graph in graphs[split]))
        split_metrics[split] = row

    metrics = {
        "model_name": "ViSNet",
        "checkpoint": args.checkpoint,
        "graph_dir": args.graph_dir,
        "device": str(device),
        "batch_size": args.batch_size,
        "amp": use_amp,
        "amp_dtype": args.amp_dtype if use_amp else "float32",
        "finite_representation_guard": args.finite_representation_guard,
        "representation_clip": args.representation_clip,
        "contains_plip_ring_features": False,
        "trainable_interaction_types": [
            train.INTERACTION_TYPES[index] for index in trainable_classes
        ],
        "skipped_interaction_types": [
            name for index, name in enumerate(train.INTERACTION_TYPES)
            if index not in trainable_classes
        ],
        "history": checkpoint.get("history", []),
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
