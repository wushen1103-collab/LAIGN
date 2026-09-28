#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import ViSNet


ROOT = Path(__file__).resolve().parents[2]
INTERACTION_TYPES = [
    "hydrophobic_contact",
    "hydrogen_bond",
    "water_bridge",
    "salt_bridge",
    "pi_stacking",
    "pi_cation",
]
SPLIT_ORDER = ("train", "internal_val", "external_val", "external_test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a raw-coordinate ViSNet residue-atom interaction model."
    )
    parser.add_argument(
        "--graph-dir",
        default=str(
            ROOT / "data/processed/structure_supervision/interaction_graphs"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "outputs/interaction_scorer"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(ROOT / "checkpoints/interaction_scorer"),
    )
    parser.add_argument(
        "--table",
        default=str(ROOT / "outputs/tables/interaction_scorer.md"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num-rbf", type=int, default=16)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--max-neighbors", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-loader-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--finite-representation-guard",
        action="store_true",
        help="Replace non-finite ViSNet scalar embeddings before pair decoding.",
    )
    parser.add_argument("--representation-clip", type=float, default=0.0)
    parser.add_argument(
        "--amp-dtype",
        choices=["float16", "bfloat16"],
        default="bfloat16",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_graphs(graph_dir, split):
    paths = sorted((Path(graph_dir) / split).glob("graph_shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No graph shards found for split {split}")
    graphs = []
    for path in paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        for graph in shard:
            forbidden = {
                "ring_geometry",
                "ring_center",
                "ring_normal",
                "ring_offset",
            } & set(graph.keys())
            if forbidden:
                raise RuntimeError(
                    f"Forbidden PLIP-derived fields in {path}: {sorted(forbidden)}"
                )
        graphs.extend(shard)
    return graphs


def radial_features(distance, num_rbf, cutoff):
    centers = torch.linspace(0.0, cutoff, num_rbf, device=distance.device)
    width = cutoff / max(1, num_rbf - 1)
    return torch.exp(
        -torch.square(distance[:, None] - centers[None, :])
        / (2.0 * width * width)
    )


class PairViSNet(nn.Module):
    def __init__(
        self,
        hidden,
        layers,
        heads,
        num_rbf,
        cutoff,
        max_neighbors,
        dropout,
        finite_representation_guard=False,
        representation_clip=0.0,
    ):
        super().__init__()
        model = ViSNet(
            lmax=1,
            num_heads=heads,
            num_layers=layers,
            hidden_channels=hidden,
            num_rbf=num_rbf,
            max_z=128,
            cutoff=cutoff,
            max_num_neighbors=max_neighbors,
            vertex=False,
        )
        self.representation = model.representation_model
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.finite_representation_guard = finite_representation_guard
        self.representation_clip = float(representation_clip)
        pair_dim = hidden * 4 + num_rbf
        self.decoder = nn.Sequential(
            nn.Linear(pair_dim, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(INTERACTION_TYPES)),
        )

    def forward(self, batch):
        scalar, _ = self.representation(batch.z, batch.pos, batch.batch)
        if self.finite_representation_guard:
            if self.representation_clip > 0.0:
                scalar = torch.nan_to_num(
                    scalar,
                    nan=0.0,
                    posinf=self.representation_clip,
                    neginf=-self.representation_clip,
                ).clamp(-self.representation_clip, self.representation_clip)
            else:
                scalar = torch.nan_to_num(scalar, nan=0.0, posinf=0.0, neginf=0.0)
        protein_index, ligand_index = batch.pair_index
        protein = scalar[protein_index]
        ligand = scalar[ligand_index]
        distance = torch.linalg.norm(
            batch.pos[protein_index] - batch.pos[ligand_index],
            dim=1,
        )
        pair_features = torch.cat(
            [
                protein,
                ligand,
                protein * ligand,
                torch.abs(protein - ligand),
                radial_features(distance, self.num_rbf, self.cutoff),
            ],
            dim=1,
        )
        return self.decoder(pair_features)


def compute_pos_weight(graphs):
    positives = torch.zeros(len(INTERACTION_TYPES), dtype=torch.float64)
    available = torch.zeros(len(INTERACTION_TYPES), dtype=torch.float64)
    for graph in graphs:
        positives += (graph.y * graph.loss_mask).sum(dim=0).double()
        available += graph.loss_mask.sum(dim=0).double()
    negatives = available - positives
    weights = torch.ones(len(INTERACTION_TYPES), dtype=torch.float32)
    trainable = []
    for index in range(len(INTERACTION_TYPES)):
        if positives[index] > 0 and available[index] > 0:
            weights[index] = min(50.0, float(negatives[index] / positives[index]))
            trainable.append(index)
    return weights, trainable


def masked_bce(logits, labels, mask, pos_weight, class_mask):
    active_mask = mask * class_mask[None, :]
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
        reduction="none",
    )
    return (loss * active_mask).sum() / active_mask.sum().clamp_min(1.0)


def safe_metric(fn, y, probability):
    if y.size == 0 or y.min() == y.max():
        return math.nan
    return float(fn(y, probability))


def evaluate_arrays(probabilities, labels, masks, trainable_classes):
    class_rows = {}
    macro_auprc = []
    macro_auroc = []
    micro_y = []
    micro_probability = []
    for index, name in enumerate(INTERACTION_TYPES):
        active = (masks[:, index] > 0) & (index in trainable_classes)
        y = labels[active, index].astype(np.int32)
        probability = probabilities[active, index]
        auprc = safe_metric(average_precision_score, y, probability)
        auroc = safe_metric(roc_auc_score, y, probability)
        prediction = (probability >= 0.5).astype(np.int32)
        f1 = safe_metric(f1_score, y, prediction)
        class_rows[name] = {
            "evaluated": bool(index in trainable_classes),
            "pairs": int(active.sum()),
            "positives": int(y.sum()) if y.size else 0,
            "prevalence": float(y.mean()) if y.size else math.nan,
            "auprc": auprc,
            "auroc": auroc,
            "f1_at_0_5": f1,
        }
        if index in trainable_classes and y.size and y.min() != y.max():
            macro_auprc.append(auprc)
            macro_auroc.append(auroc)
            micro_y.append(y)
            micro_probability.append(probability)
    all_y = np.concatenate(micro_y)
    all_probability = np.concatenate(micro_probability)
    return {
        "micro_auprc": safe_metric(
            average_precision_score,
            all_y,
            all_probability,
        ),
        "micro_auroc": safe_metric(roc_auc_score, all_y, all_probability),
        "micro_prevalence_baseline": float(all_y.mean()),
        "macro_auprc": float(np.nanmean(macro_auprc)),
        "macro_auroc": float(np.nanmean(macro_auroc)),
        "classes": class_rows,
    }


@torch.no_grad()
def predict(model, loader, device, use_amp, amp_dtype):
    model.eval()
    probabilities = []
    labels = []
    masks = []
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            logits = model(batch)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite logits encountered during evaluation")
        probabilities.append(torch.sigmoid(logits).float().cpu())
        labels.append(batch.y.cpu())
        masks.append(batch.loss_mask.cpu())
    return (
        torch.cat(probabilities).numpy(),
        torch.cat(labels).numpy(),
        torch.cat(masks).numpy(),
    )


def write_table(metrics, path):
    lines = [
        "# Raw-Coordinate ViSNet Interaction Probe",
        "",
        f"- graph dataset: `{metrics['graph_dir']}`",
        f"- checkpoint: `{metrics['best_checkpoint']}`",
        "- PLIP-derived ring features: forbidden",
        f"- hidden/layers/heads: {metrics['hidden']}/{metrics['layers']}/{metrics['heads']}",
        "",
        "| split | graphs | pairs | micro_AUPRC | prevalence | macro_AUPRC | micro_AUROC | macro_AUROC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLIT_ORDER:
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
    set_seed(args.seed)
    device_name = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if args.amp_dtype == "bfloat16"
        else torch.float16
    )

    graphs = {
        split: load_graphs(args.graph_dir, split)
        for split in SPLIT_ORDER
    }
    loaders = {
        split: DataLoader(
            split_graphs,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_loader_workers,
            pin_memory=device.type == "cuda",
        )
        for split, split_graphs in graphs.items()
    }
    pos_weight, trainable_classes = compute_pos_weight(graphs["train"])
    skipped_classes = [
        INTERACTION_TYPES[index]
        for index in range(len(INTERACTION_TYPES))
        if index not in trainable_classes
    ]
    pos_weight = pos_weight.to(device)
    class_mask = torch.zeros(len(INTERACTION_TYPES), device=device)
    class_mask[trainable_classes] = 1.0

    model = PairViSNet(
        hidden=args.hidden,
        layers=args.layers,
        heads=args.heads,
        num_rbf=args.num_rbf,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        dropout=args.dropout,
        finite_representation_guard=args.finite_representation_guard,
        representation_clip=args.representation_clip,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp and amp_dtype == torch.float16,
    )
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoint_dir / "best.pt"
    best_score = -math.inf
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_weight = 0.0
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                logits = model(batch)
                loss = masked_bce(
                    logits,
                    batch.y,
                    batch.loss_mask,
                    pos_weight,
                    class_mask,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss encountered at epoch {epoch}"
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            weight = float((batch.loss_mask * class_mask[None, :]).sum().cpu())
            total_loss += float(loss.detach().cpu()) * weight
            total_weight += weight

        probability, labels, masks = predict(
            model,
            loaders["internal_val"],
            device,
            use_amp,
            amp_dtype,
        )
        val_metrics = evaluate_arrays(
            probability,
            labels,
            masks,
            trainable_classes,
        )
        score = val_metrics["macro_auprc"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(1.0, total_weight),
                "internal_val_macro_auprc": score,
                "internal_val_micro_auprc": val_metrics["micro_auprc"],
            }
        )
        print(
            f"epoch={epoch} train_loss={history[-1]['train_loss']:.6f} "
            f"val_macro_auprc={score:.4f} "
            f"val_micro_auprc={val_metrics['micro_auprc']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "trainable_classes": trainable_classes,
                    "history": history,
                },
                best_checkpoint,
            )

    checkpoint = torch.load(
        best_checkpoint,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    split_metrics = {}
    for split in SPLIT_ORDER:
        probability, labels, masks = predict(
            model,
            loaders[split],
            device,
            use_amp,
            amp_dtype,
        )
        row = evaluate_arrays(
            probability,
            labels,
            masks,
            trainable_classes,
        )
        row["graphs"] = len(graphs[split])
        row["pairs"] = int(sum(graph.y.shape[0] for graph in graphs[split]))
        split_metrics[split] = row

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model_name": "ViSNet",
        "graph_dir": args.graph_dir,
        "best_checkpoint": str(best_checkpoint),
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "layers": args.layers,
        "heads": args.heads,
        "num_rbf": args.num_rbf,
        "cutoff": args.cutoff,
        "max_neighbors": args.max_neighbors,
        "amp": use_amp,
        "amp_dtype": args.amp_dtype if use_amp else "float32",
        "finite_representation_guard": args.finite_representation_guard,
        "representation_clip": args.representation_clip,
        "contains_plip_ring_features": False,
        "trainable_interaction_types": [
            INTERACTION_TYPES[index]
            for index in trainable_classes
        ],
        "skipped_interaction_types": skipped_classes,
        "history": history,
        "splits": split_metrics,
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_table(metrics, args.table)
    print(
        json.dumps(
            {
                "metrics": str(metrics_path),
                "table": args.table,
                "best_checkpoint": str(best_checkpoint),
            }
        )
    )


if __name__ == "__main__":
    main()
