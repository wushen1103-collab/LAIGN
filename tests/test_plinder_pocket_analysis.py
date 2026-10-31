from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluation"
    / "analyze_plinder_pocket_and_trainable_fusion.py"
)
SPEC = importlib.util.spec_from_file_location("plinder_analysis", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_split(graphs: int, residues_per_graph: int, offset: float):
    sample_ids = np.asarray([f"g{index}" for index in range(graphs)])
    sample_index = np.repeat(np.arange(graphs), residues_per_graph)
    local = np.tile(np.arange(residues_per_graph), graphs)
    labels = (local < 2).astype(np.int8)
    distance_pattern = np.asarray([2.00, 3.00, 2.05, 3.05, 5.00, 6.00])
    distances = np.tile(distance_pattern, graphs) + offset
    interaction = np.where(labels == 1, 0.8, 0.2) + 0.01 * sample_index
    fixed = 0.55 * interaction + 0.45 * np.exp(-distances / 12.0)
    trained = np.where(labels == 1, 0.9, 0.1) + 0.001 * sample_index
    return {
        "labels": labels,
        "sample_index": sample_index.astype(np.int32),
        "sample_ids": sample_ids,
        "score__raw_distance": -distances,
        "score__raw_visnet": interaction,
        "score__fixed_hybrid": fixed,
        "score__trained_stage2": trained,
    }


def test_end_to_end_smoke(tmp_path):
    paths = []
    for seed_offset in (0.0, 0.01, -0.01):
        arrays = {}
        for split, graphs in (
            ("train", 8),
            ("internal_val", 4),
            ("external_val", 2),
            ("external_test", 2),
        ):
            values = make_split(graphs, 6, 0.0)
            values["score__trained_stage2"] = values["score__trained_stage2"] + seed_offset
            for name, value in values.items():
                arrays[f"{split}__{name}"] = value
        path = tmp_path / f"archive_{len(paths)}.npz"
        np.savez_compressed(path, **arrays)
        paths.append(path)

    subset = pd.DataFrame(
        {
            "sample_id": ["g0", "g1", "g0", "g1"],
            "split": ["external_val", "external_val", "external_test", "external_test"],
        }
    )
    subset_path = tmp_path / "subset.csv"
    subset.to_csv(subset_path, index=False)

    archives = [np.load(path, allow_pickle=False) for path in paths]
    MODULE.validate_archives(archives, paths)
    fitted = [
        MODULE.fit_trainable_fusion(archive, [0.1, 1.0], 7 + index)
        for index, archive in enumerate(archives)
    ]
    models = [item[0] for item in fitted]
    trace = fitted[0][1]
    metrics, residues = MODULE.collect_external_rows(
        archives, models, MODULE.load_selected(subset_path)
    )
    summary = MODULE.summarize_metrics(metrics, bootstrap=20, seed=10)
    effects, _ = MODULE.paired_effects(metrics, 20, 40, 11)
    pairs, pair_summary = MODULE.distance_matched_pairs(residues)
    pair_effects, _ = MODULE.distance_matched_effects(pairs, 20, 40, 12)

    assert len(trace) == 2
    assert set(summary["subset"]) == {"All residues", "Distance <= 6 A", "Distance <= 8 A"}
    assert set(summary["method"]) == set(MODULE.METHODS)
    assert residues[["split", "sample_id"]].drop_duplicates().shape[0] == 4
    assert not effects.empty
    assert not pair_summary.empty
    assert not pair_effects.empty
