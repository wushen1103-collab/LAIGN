#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
SPLIT_TO_CODE = {"train": 0, "val": 1, "test": 2}


def parse_args():
    parser = argparse.ArgumentParser(description="Train an ESM2+Morgan MLP weak pretraining baseline.")
    parser.add_argument("--training-dir", default=str(ROOT / "data/processed/pretrain/training"))
    parser.add_argument("--ligand-fp", default=str(ROOT / "data/processed/pretrain/ligands/morgan_r2_2048/fingerprints_packed_uint8.npy"))
    parser.add_argument("--out-dir", default=str(ROOT / "checkpoints/pretrain_mlp"))
    parser.add_argument("--table", default=str(ROOT / "outputs/tables/table_s4_pretrain_mlp.md"))
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--input-mode", choices=["full", "ligand_only", "target_only"], default="full")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


class PairMLP(nn.Module):
    def __init__(self, ligand_dim=2048, target_dim=1280, hidden=512, dropout=0.15):
        super().__init__()
        self.ligand = nn.Sequential(
            nn.Linear(ligand_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.target = nn.Sequential(
            nn.Linear(target_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, ligand_bits, target_embedding):
        ligand_h = self.ligand(ligand_bits)
        target_h = self.target(target_embedding)
        return self.head(torch.cat([ligand_h, target_h], dim=-1)).squeeze(-1)


class TargetBalancedSampler:
    def __init__(self, pair_target_idx, pair_label, pair_split, seed):
        train_indices = np.flatnonzero(pair_split == SPLIT_TO_CODE["train"])
        targets = pair_target_idx[train_indices]
        labels = pair_label[train_indices]
        order = np.argsort(targets, kind="mergesort")
        train_indices = train_indices[order]
        targets = targets[order]
        labels = labels[order]

        self.pos_by_target = {}
        self.neg_by_target = {}
        start = 0
        while start < len(train_indices):
            end = start + 1
            target = int(targets[start])
            while end < len(train_indices) and int(targets[end]) == target:
                end += 1
            idxs = train_indices[start:end]
            labs = labels[start:end]
            pos = idxs[labs == 1]
            neg = idxs[labs == 0]
            if len(pos):
                self.pos_by_target[target] = pos
            if len(neg):
                self.neg_by_target[target] = neg
            start = end

        self.pos_targets = np.array(sorted(self.pos_by_target), dtype=np.int32)
        self.neg_targets = np.array(sorted(self.neg_by_target), dtype=np.int32)
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size):
        n_pos = batch_size // 2
        n_neg = batch_size - n_pos
        pos_targets = self.rng.choice(self.pos_targets, size=n_pos, replace=True)
        neg_targets = self.rng.choice(self.neg_targets, size=n_neg, replace=True)
        pos_indices = np.fromiter(
            (self.rng.choice(self.pos_by_target[int(t)]) for t in pos_targets),
            dtype=np.int64,
            count=n_pos,
        )
        neg_indices = np.fromiter(
            (self.rng.choice(self.neg_by_target[int(t)]) for t in neg_targets),
            dtype=np.int64,
            count=n_neg,
        )
        batch = np.concatenate([pos_indices, neg_indices])
        self.rng.shuffle(batch)
        return batch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_arrays(training_dir, ligand_fp_path):
    training_dir = Path(training_dir)
    return {
        "target_embeddings": np.load(training_dir / "target_embeddings_fp16.npy", mmap_mode="r"),
        "ligand_fp": np.load(ligand_fp_path, mmap_mode="r"),
        "pair_target_idx": np.load(training_dir / "pair_target_idx.npy", mmap_mode="r"),
        "pair_ligand_idx": np.load(training_dir / "pair_ligand_idx.npy", mmap_mode="r"),
        "pair_label": np.load(training_dir / "pair_label.npy", mmap_mode="r"),
        "pair_split": np.load(training_dir / "pair_split.npy", mmap_mode="r"),
    }


def make_batch(arrays, pair_indices, device, input_mode="full"):
    ligand_rows = arrays["pair_ligand_idx"][pair_indices]
    target_rows = arrays["pair_target_idx"][pair_indices]
    ligand_bits = np.unpackbits(arrays["ligand_fp"][ligand_rows], axis=1).astype(np.float32)
    target_embedding = arrays["target_embeddings"][target_rows].astype(np.float32)
    if input_mode == "ligand_only":
        target_embedding.fill(0.0)
    elif input_mode == "target_only":
        ligand_bits.fill(0.0)
    labels = arrays["pair_label"][pair_indices].astype(np.float32)
    return (
        torch.from_numpy(ligand_bits).to(device, non_blocking=True),
        torch.from_numpy(target_embedding).to(device, non_blocking=True),
        torch.from_numpy(labels).to(device, non_blocking=True),
    )


def expected_calibration_error(y_true, y_prob, bins=15):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not mask.any():
            continue
        ece += mask.mean() * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return ece


def target_level_auc(target_idx, y_true, y_prob):
    scores = []
    grouped = defaultdict(list)
    for i, target in enumerate(target_idx):
        grouped[int(target)].append(i)
    for indices in grouped.values():
        labels = y_true[indices]
        if labels.min() == labels.max():
            continue
        try:
            scores.append(roc_auc_score(labels, y_prob[indices]))
        except ValueError:
            continue
    return {"mean_target_auc": float(np.mean(scores)) if scores else math.nan, "targets_with_auc": len(scores)}


@torch.no_grad()
def evaluate(model, arrays, split_name, device, batch_size, input_mode="full"):
    split_code = SPLIT_TO_CODE[split_name]
    indices = np.flatnonzero(arrays["pair_split"] == split_code)
    probs = []
    labels = []
    targets = []
    model.eval()
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        ligand_bits, target_embedding, y = make_batch(arrays, batch_idx, device, input_mode=input_mode)
        logits = model(ligand_bits, target_embedding)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(prob)
        labels.append(y.detach().cpu().numpy())
        targets.append(np.asarray(arrays["pair_target_idx"][batch_idx]))
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(labels).astype(np.int32)
    target_idx = np.concatenate(targets)
    metrics = {
        "split": split_name,
        "pairs": int(len(y_true)),
        "positives": int(y_true.sum()),
        "weak_negatives": int(len(y_true) - y_true.sum()),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))),
        "ece_15": float(expected_calibration_error(y_true, y_prob, bins=15)),
    }
    metrics.update(target_level_auc(target_idx, y_true, y_prob))
    return metrics


def append_history(path, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_table(path, args, best_step, val_metrics, test_metrics):
    lines = [
        "# S4 Weak Pretrain MLP Baseline",
        "",
        f"- steps: {args.steps}",
        f"- batch size: {args.batch_size}",
        f"- input mode: {args.input_mode}",
        f"- sampler: target-balanced and label-balanced over train split",
        f"- best validation step by AUROC: {best_step}",
        "",
        "| split | pairs | AUROC | AUPRC | mean_target_AUROC | target_count | Brier | ECE-15 | log_loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metrics in [val_metrics, test_metrics]:
        lines.append(
            f"| {metrics['split']} | {metrics['pairs']} | {metrics['auroc']:.4f} | {metrics['auprc']:.4f} | "
            f"{metrics['mean_target_auc']:.4f} | {metrics['targets_with_auc']} | {metrics['brier']:.4f} | "
            f"{metrics['ece_15']:.4f} | {metrics['log_loss']:.4f} |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    arrays = load_arrays(args.training_dir, args.ligand_fp)
    sampler = TargetBalancedSampler(arrays["pair_target_idx"], arrays["pair_label"], arrays["pair_split"], args.seed)

    model = PairMLP(hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history_path = out_dir / "history.csv"
    best_val = -math.inf
    best_step = 0
    best_path = out_dir / "best.pt"
    start = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        batch_indices = sampler.sample(args.batch_size)
        ligand_bits, target_embedding, y = make_batch(arrays, batch_indices, device, input_mode=args.input_mode)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(ligand_bits, target_embedding)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % args.eval_every == 0 or step == args.steps:
            val_metrics = evaluate(model, arrays, "val", device, args.eval_batch_size, input_mode=args.input_mode)
            row = {"step": step, "train_loss": float(loss.item()), **{f"val_{k}": v for k, v in val_metrics.items() if k != "split"}}
            append_history(history_path, row)
            elapsed = time.time() - start
            print(
                f"step={step} loss={loss.item():.4f} val_auroc={val_metrics['auroc']:.4f} "
                f"val_auprc={val_metrics['auprc']:.4f} elapsed_sec={elapsed:.1f}",
                flush=True,
            )
            if val_metrics["auroc"] > best_val:
                best_val = val_metrics["auroc"]
                best_step = step
                torch.save({"model": model.state_dict(), "args": vars(args), "val_metrics": val_metrics}, best_path)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    val_metrics = evaluate(model, arrays, "val", device, args.eval_batch_size, input_mode=args.input_mode)
    test_metrics = evaluate(model, arrays, "test", device, args.eval_batch_size, input_mode=args.input_mode)
    metrics = {"best_step": best_step, "val": val_metrics, "test": test_metrics, "args": vars(args)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_table(args.table, args, best_step, val_metrics, test_metrics)
    print(out_dir / "metrics.json")
    print(args.table)


if __name__ == "__main__":
    main()
