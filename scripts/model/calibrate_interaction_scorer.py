#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn

from scripts.model.train_pair_pretraining import (
    PairMLP,
    SPLIT_TO_CODE,
    expected_calibration_error,
    load_arrays,
    make_batch,
    target_level_auc,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Temperature-scale the weak pretraining MLP baseline.")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints/pretrain_mlp/best.pt"))
    parser.add_argument("--training-dir", default=str(ROOT / "data/processed/pretrain/training"))
    parser.add_argument("--ligand-fp", default=str(ROOT / "data/processed/pretrain/ligands/morgan_r2_2048/fingerprints_packed_uint8.npy"))
    parser.add_argument("--out-dir", default=str(ROOT / "checkpoints/pretrain_mlp"))
    parser.add_argument("--table", default=str(ROOT / "outputs/tables/table_s4_pretrain_mlp_calibration.md"))
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def collect_logits(model, arrays, split_name, device, batch_size, input_mode="full"):
    indices = np.flatnonzero(arrays["pair_split"] == SPLIT_TO_CODE[split_name])
    logits = []
    labels = []
    targets = []
    model.eval()
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        ligand_bits, target_embedding, y = make_batch(arrays, batch_idx, device, input_mode=input_mode)
        batch_logits = model(ligand_bits, target_embedding).detach().cpu().numpy()
        logits.append(batch_logits)
        labels.append(y.detach().cpu().numpy())
        targets.append(np.asarray(arrays["pair_target_idx"][batch_idx]))
    return np.concatenate(logits), np.concatenate(labels).astype(np.int32), np.concatenate(targets)


def metrics_from_logits(split, logits, labels, targets, scale=1.0, bias=0.0):
    transformed = np.clip(logits * scale + bias, -60.0, 60.0)
    probs = 1.0 / (1.0 + np.exp(-transformed))
    out = {
        "split": split,
        "scale": float(scale),
        "bias": float(bias),
        "pairs": int(len(labels)),
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
        "brier": float(brier_score_loss(labels, probs)),
        "log_loss": float(log_loss(labels, np.clip(probs, 1e-6, 1 - 1e-6))),
        "ece_15": float(expected_calibration_error(labels, probs, bins=15)),
    }
    out.update(target_level_auc(targets, labels, probs))
    return out


def fit_temperature(logits, labels, device):
    logits_t = torch.from_numpy(logits.astype(np.float32)).to(device)
    labels_t = torch.from_numpy(labels.astype(np.float32)).to(device)
    log_temperature = torch.nn.Parameter(torch.zeros((), device=device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.25, max_iter=80, line_search_fn="strong_wolfe")
    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature).clamp(0.05, 50.0)
        loss = loss_fn(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_temperature).clamp(0.05, 50.0).detach().cpu())
    return temperature


def fit_platt(logits, labels, device):
    logits_t = torch.from_numpy(logits.astype(np.float32)).to(device)
    labels_t = torch.from_numpy(labels.astype(np.float32)).to(device)
    log_scale = torch.nn.Parameter(torch.zeros((), device=device))
    bias = torch.nn.Parameter(torch.zeros((), device=device))
    optimizer = torch.optim.LBFGS([log_scale, bias], lr=0.25, max_iter=100, line_search_fn="strong_wolfe")
    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad(set_to_none=True)
        scale = torch.exp(log_scale).clamp(0.02, 50.0)
        loss = loss_fn(logits_t * scale + bias, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    scale = float(torch.exp(log_scale).clamp(0.02, 50.0).detach().cpu())
    return scale, float(bias.detach().cpu())


def write_table(path, temperature, platt_scale, platt_bias, rows):
    lines = [
        "# S4 Pretrain MLP Calibration",
        "",
        f"- temperature fitted on validation NLL: {temperature:.4f}",
        f"- Platt scale/bias fitted on validation NLL: {platt_scale:.4f}/{platt_bias:.4f}",
        "",
        "| split | calibration | AUROC | AUPRC | mean_target_AUROC | Brier | ECE-15 | log_loss |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['calibration']} | {row['auroc']:.4f} | {row['auprc']:.4f} | "
            f"{row['mean_target_auc']:.4f} | {row['brier']:.4f} | {row['ece_15']:.4f} | {row['log_loss']:.4f} |"
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    input_mode = ckpt_args.get("input_mode", "full")
    model = PairMLP(hidden=int(ckpt_args.get("hidden", 512)), dropout=float(ckpt_args.get("dropout", 0.15))).to(device)
    model.load_state_dict(checkpoint["model"])
    arrays = load_arrays(args.training_dir, args.ligand_fp)

    val_logits, val_labels, val_targets = collect_logits(model, arrays, "val", device, args.eval_batch_size, input_mode=input_mode)
    test_logits, test_labels, test_targets = collect_logits(model, arrays, "test", device, args.eval_batch_size, input_mode=input_mode)
    temperature = fit_temperature(val_logits, val_labels, device)
    platt_scale, platt_bias = fit_platt(val_logits, val_labels, device)

    rows = []
    for split, logits, labels, targets in [
        ("val", val_logits, val_labels, val_targets),
        ("test", test_logits, test_labels, test_targets),
    ]:
        raw = metrics_from_logits(split, logits, labels, targets, scale=1.0, bias=0.0)
        raw["calibration"] = "raw"
        temperature_row = metrics_from_logits(split, logits, labels, targets, scale=1.0 / temperature, bias=0.0)
        temperature_row["calibration"] = "temperature"
        platt = metrics_from_logits(split, logits, labels, targets, scale=platt_scale, bias=platt_bias)
        platt["calibration"] = "platt"
        rows.extend([raw, temperature_row, platt])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"temperature": temperature, "platt_scale": platt_scale, "platt_bias": platt_bias, "metrics": rows}
    (out_dir / "calibration_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_table(args.table, temperature, platt_scale, platt_bias, rows)
    print(out_dir / "calibration_metrics.json")
    print(args.table)


if __name__ == "__main__":
    main()
