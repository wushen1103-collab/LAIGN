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
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
SPLIT_TO_ID = {"train": 0, "internal_val": 1, "external_val": 2, "external_test": 3}
INTERACTION_TYPES = [
    "hydrophobic_contact",
    "hydrogen_bond",
    "water_bridge",
    "salt_bridge",
    "pi_stacking",
    "pi_cation",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a lightweight residue-atom interaction MLP smoke model.")
    parser.add_argument(
        "--features",
        default=str(ROOT / "data/processed/structure_supervision/pair_features_smoke/interaction_pair_features.npz"),
    )
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/interaction_pair_mlp_smoke"))
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints/interaction_pair_mlp_smoke"))
    parser.add_argument("--table", default=str(ROOT / "outputs/tables/table_interaction_pair_mlp_smoke.md"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-mode", choices=["full", "distance_only", "identity_only"], default="full")
    parser.add_argument(
        "--ring-geometry-mode",
        choices=["full", "membership_only", "none"],
        default="full",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def distance_features(distances):
    distances = np.asarray(distances, dtype=np.float32)
    clipped = np.clip(distances, 0.0, 20.0)
    raw = clipped / 20.0
    inv = 1.0 / (1.0 + clipped)
    centers = np.asarray([2.0, 3.5, 5.0, 6.5, 8.0, 10.0, 12.0], dtype=np.float32)
    widths = 1.5
    rbf_min = np.exp(-((clipped[:, 0:1] - centers[None, :]) ** 2) / (2 * widths * widths))
    rbf_centroid = np.exp(-((clipped[:, 1:2] - centers[None, :]) ** 2) / (2 * widths * widths))
    return np.concatenate([raw, inv, rbf_min, rbf_centroid], axis=1).astype(np.float32)


class PairMLP(nn.Module):
    def __init__(self, n_residue, n_element, dist_dim, hidden, dropout, out_dim, feature_mode):
        super().__init__()
        self.feature_mode = feature_mode
        self.use_identity = feature_mode in {"full", "identity_only"}
        self.use_distance = feature_mode in {"full", "distance_only"}
        if self.use_identity:
            self.residue_emb = nn.Embedding(n_residue, 16)
            self.element_emb = nn.Embedding(n_element, 12)
        else:
            self.residue_emb = None
            self.element_emb = None
        in_dim = 0
        if self.use_identity:
            in_dim += 16 + 12
        if self.use_distance:
            in_dim += dist_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, residue_type, ligand_element, dist_feat):
        chunks = []
        if self.use_identity:
            chunks.extend([self.residue_emb(residue_type), self.element_emb(ligand_element)])
        if self.use_distance:
            chunks.append(dist_feat)
        x = torch.cat(chunks, dim=1)
        return self.net(x)


def load_arrays(path, ring_geometry_mode):
    data = np.load(path, allow_pickle=True)
    dist_feat = distance_features(data["distances"])
    ring_geometry_dim = 0
    if "ring_geometry" in data.files and ring_geometry_mode != "none":
        ring_geometry = data["ring_geometry"].astype(np.float32)
        if ring_geometry_mode == "membership_only":
            ring_geometry = ring_geometry[:, :1]
        dist_feat = np.concatenate([dist_feat, ring_geometry], axis=1)
        ring_geometry_dim = int(ring_geometry.shape[1])
    return {
        "residue_type": data["residue_type"].astype(np.int64),
        "ligand_element": data["ligand_element"].astype(np.int64),
        "dist_feat": dist_feat,
        "split": data["split"].astype(np.int64),
        "labels": data["labels"].astype(np.float32),
        "loss_mask": data["loss_mask"].astype(np.float32),
        "ring_geometry_dim": ring_geometry_dim,
    }


def make_dataset(arrays, mask):
    return TensorDataset(
        torch.from_numpy(arrays["residue_type"][mask]),
        torch.from_numpy(arrays["ligand_element"][mask]),
        torch.from_numpy(arrays["dist_feat"][mask]),
        torch.from_numpy(arrays["labels"][mask]),
        torch.from_numpy(arrays["loss_mask"][mask]),
    )


def masked_bce(logits, labels, mask, pos_weight):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
    loss = loss * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def compute_pos_weight(labels, masks, trainable):
    weights = []
    for idx in range(labels.shape[1]):
        active = masks[:, idx] > 0
        if idx not in trainable or not active.any():
            weights.append(1.0)
            continue
        positives = float(labels[active, idx].sum())
        negatives = float(active.sum() - positives)
        if positives <= 0:
            weights.append(1.0)
        else:
            weights.append(min(50.0, negatives / positives))
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def predict(model, loader, device):
    logits = []
    labels = []
    masks = []
    model.eval()
    for residue_type, ligand_element, dist_feat, y, m in loader:
        residue_type = residue_type.to(device)
        ligand_element = ligand_element.to(device)
        dist_feat = dist_feat.to(device)
        logits.append(model(residue_type, ligand_element, dist_feat).cpu())
        labels.append(y)
        masks.append(m)
    logits = torch.cat(logits).numpy()
    labels = torch.cat(labels).numpy()
    masks = torch.cat(masks).numpy()
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
    return probs, labels, masks


def safe_metric(fn, y, p):
    if y.size == 0 or y.min() == y.max():
        return math.nan
    try:
        return float(fn(y, p))
    except Exception:
        return math.nan


def evaluate_probs(probs, labels, masks, trainable_classes):
    class_rows = {}
    micro_y = []
    micro_p = []
    macro_auprc = []
    macro_auroc = []
    for idx, name in enumerate(INTERACTION_TYPES):
        active = (masks[:, idx] > 0) & (idx in trainable_classes)
        y = labels[active, idx].astype(np.int32)
        p = probs[active, idx]
        positives = int(y.sum()) if y.size else 0
        prevalence = float(y.mean()) if y.size else math.nan
        auprc = safe_metric(average_precision_score, y, p)
        auroc = safe_metric(roc_auc_score, y, p)
        pred = (p >= 0.5).astype(np.int32) if y.size else np.asarray([], dtype=np.int32)
        f1 = safe_metric(f1_score, y, pred)
        class_rows[name] = {
            "evaluated": bool(idx in trainable_classes),
            "pairs": int(active.sum()),
            "positives": positives,
            "prevalence": prevalence,
            "auprc": auprc,
            "auroc": auroc,
            "f1_at_0_5": f1,
        }
        if idx in trainable_classes and y.size and y.min() != y.max():
            macro_auprc.append(auprc)
            macro_auroc.append(auroc)
            micro_y.append(y)
            micro_p.append(p)
    if micro_y:
        y_all = np.concatenate(micro_y)
        p_all = np.concatenate(micro_p)
        micro_auprc = safe_metric(average_precision_score, y_all, p_all)
        micro_auroc = safe_metric(roc_auc_score, y_all, p_all)
        micro_prevalence = float(y_all.mean())
    else:
        micro_auprc = micro_auroc = micro_prevalence = math.nan
    return {
        "micro_auprc": micro_auprc,
        "micro_auroc": micro_auroc,
        "micro_prevalence_baseline": micro_prevalence,
        "macro_auprc": float(np.nanmean(macro_auprc)) if macro_auprc else math.nan,
        "macro_auroc": float(np.nanmean(macro_auroc)) if macro_auroc else math.nan,
        "classes": class_rows,
    }


def write_table(metrics, path):
    lines = [
        "# Interaction Pair MLP",
        "",
        f"- features: `{metrics['features']}`",
        f"- feature mode: `{metrics['feature_mode']}`",
        f"- ring geometry mode: `{metrics['ring_geometry_mode']}`",
        f"- ring geometry dimensions: {metrics['ring_geometry_dim']}",
        f"- checkpoint: `{metrics['best_checkpoint']}`",
        f"- trainable interaction types: {', '.join(metrics['trainable_interaction_types'])}",
        f"- skipped interaction types: {', '.join(metrics['skipped_interaction_types']) or 'none'}",
        "",
        "| split | pairs | micro_AUPRC | prevalence_baseline | delta_AUPRC | micro_AUROC | macro_AUPRC | macro_AUROC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name, row in metrics["splits"].items():
        delta = row["micro_auprc"] - row["micro_prevalence_baseline"]
        lines.append(
            f"| {split_name} | {row['pairs']} | {row['micro_auprc']:.4f} | "
            f"{row['micro_prevalence_baseline']:.4f} | {delta:.4f} | {row['micro_auroc']:.4f} | "
            f"{row['macro_auprc']:.4f} | {row['macro_auroc']:.4f} |"
        )
    lines.extend(["", "## Per-Type AUPRC", "", "| split | interaction_type | pairs | positives | AUPRC | AUROC | prevalence |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for split_name, row in metrics["splits"].items():
        for interaction_type, class_row in row["classes"].items():
            if not class_row["evaluated"]:
                continue
            lines.append(
                f"| {split_name} | {interaction_type} | {class_row['pairs']} | {class_row['positives']} | "
                f"{class_row['auprc']:.4f} | {class_row['auroc']:.4f} | {class_row['prevalence']:.4f} |"
            )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    set_seed(args.seed)
    arrays = load_arrays(args.features, args.ring_geometry_mode)
    train_mask = arrays["split"] == SPLIT_TO_ID["train"]
    train_labels = arrays["labels"][train_mask]
    train_loss_mask = arrays["loss_mask"][train_mask]
    trainable_classes = [
        idx for idx in range(train_labels.shape[1])
        if train_loss_mask[:, idx].sum() > 0 and train_labels[:, idx].sum() > 0
    ]
    skipped = [INTERACTION_TYPES[idx] for idx in range(len(INTERACTION_TYPES)) if idx not in trainable_classes]
    if not trainable_classes:
        raise RuntimeError("No trainable interaction classes found.")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_ds = make_dataset(arrays, train_mask)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    eval_loaders = {}
    for split_name, split_id in SPLIT_TO_ID.items():
        mask = arrays["split"] == split_id
        eval_loaders[split_name] = DataLoader(
            make_dataset(arrays, mask),
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=2,
            pin_memory=device.type == "cuda",
        )

    model = PairMLP(
        n_residue=int(arrays["residue_type"].max()) + 1,
        n_element=int(arrays["ligand_element"].max()) + 1,
        dist_dim=arrays["dist_feat"].shape[1],
        hidden=args.hidden,
        dropout=args.dropout,
        out_dim=len(INTERACTION_TYPES),
        feature_mode=args.feature_mode,
    ).to(device)
    pos_weight = compute_pos_weight(train_labels, train_loss_mask, trainable_classes).to(device)
    class_train_mask = torch.zeros(len(INTERACTION_TYPES), dtype=torch.float32, device=device)
    class_train_mask[trainable_classes] = 1.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -math.inf
    history = []
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoint_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_weight = 0.0
        for residue_type, ligand_element, dist_feat, y, m in train_loader:
            residue_type = residue_type.to(device)
            ligand_element = ligand_element.to(device)
            dist_feat = dist_feat.to(device)
            y = y.to(device)
            m = m.to(device) * class_train_mask[None, :]
            optimizer.zero_grad(set_to_none=True)
            logits = model(residue_type, ligand_element, dist_feat)
            loss = masked_bce(logits, y, m, pos_weight)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * float(m.sum().detach().cpu())
            total_weight += float(m.sum().detach().cpu())

        probs, labels, masks = predict(model, eval_loaders["internal_val"], device)
        val_metrics = evaluate_probs(probs, labels, masks, trainable_classes)
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
            f"val_macro_auprc={score:.4f} val_micro_auprc={val_metrics['micro_auprc']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "trainable_classes": trainable_classes,
            "interaction_types": INTERACTION_TYPES,
            "history": history,
            "feature_mode": args.feature_mode,
        },
        best_checkpoint,
            )

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    split_metrics = {}
    for split_name, loader in eval_loaders.items():
        probs, labels, masks = predict(model, loader, device)
        row = evaluate_probs(probs, labels, masks, trainable_classes)
        row["pairs"] = int((arrays["split"] == SPLIT_TO_ID[split_name]).sum())
        split_metrics[split_name] = row

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "features": args.features,
        "feature_mode": args.feature_mode,
        "ring_geometry_mode": args.ring_geometry_mode,
        "ring_geometry_dim": arrays["ring_geometry_dim"],
        "best_checkpoint": str(best_checkpoint),
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "trainable_interaction_types": [INTERACTION_TYPES[idx] for idx in trainable_classes],
        "skipped_interaction_types": skipped,
        "history": history,
        "splits": split_metrics,
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_table(metrics, args.table)
    print(json.dumps({"metrics": str(metrics_path), "table": args.table, "best_checkpoint": str(best_checkpoint)}))


if __name__ == "__main__":
    main()
