"""Freeze group-disjoint stages before evaluating any new model."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .runtime import ROOT, BASE, load_archive, canonical_id


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/processed/structure_supervision/structure_supervision_split_manifest.csv.gz"
    frame = pd.read_csv(source, low_memory=False).fillna("")
    training = frame[frame.supervision_split.eq("train")]
    group_sizes = training.groupby("split_group_id").size()
    groups = sorted(group_sizes.index, key=lambda g: hashlib.sha256(("IF-20261106:" + str(g)).encode()).hexdigest())
    held_groups, total = set(), 0
    for group in groups:
        held_groups.add(group)
        total += int(group_sizes[group])
        if total >= 5000:
            break
    held = training[training.split_group_id.isin(held_groups)]
    stage1 = training[~training.split_group_id.isin(held_groups)]
    assert not set(held.split_group_id) & set(stage1.split_group_id)
    assert not set(held.receptor_sequence) & set(stage1.receptor_sequence)
    roles = frame.copy()
    roles["if_training_role"] = roles["supervision_split"]
    roles.loc[roles.sample_id.isin(stage1.sample_id), "if_training_role"] = "stage1_train"
    roles.loc[roles.sample_id.isin(held.sample_id), "if_training_role"] = "stage2_train"
    roles.to_csv(BASE / "disjoint_protocol_manifest.csv.gz", index=False)
    config = {
        "seed": 2401, "stage1_seeds": [2401, 2402, 2403],
        "stage2_fit_seeds": [2401, 2402, 2403],
        "stage1_train_ids": sorted(stage1.sample_id.astype(str)),
        "stage2_train_ids": sorted(held.sample_id.astype(str)),
        "stage2_train_groups": sorted(held_groups),
        "stage1_train_graphs_manifest": len(stage1),
        "stage2_train_graphs_manifest": len(held),
        "stage2_train_graph_limit": 5000, "internal_validation_graph_limit": 1000,
        "selection": "Internal BioLiP2 validation only; external labels excluded from every selection decision",
        "stage1_epochs": 10, "stage1_amp": False,
        "group_overlap": 0, "exact_receptor_sequence_overlap": 0,
        "stage1_validation": "Original independent internal_val partition (4653 graphs)",
        "stage2_validation": "1000 graphs from original internal_val partition; never used for parameter fitting",
        "unchanged_external_protocol": "Original PLINDER external_val and external_test; original post-freeze 70",
        "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    (BASE / "disjoint_protocol.json").write_text(json.dumps(config, indent=2))
    summary = {k: v for k, v in config.items() if not k.endswith("ids") and not k.endswith("groups")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
