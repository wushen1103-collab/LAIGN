"""Run the unchanged scorer with only group-disjoint Stage-1 training graphs."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from .runtime import ROOT, BASE, load_archive, canonical_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "8"
    import torch
    torch.set_num_threads(8)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from scripts.model import train_interaction_scorer as original
    original_loader = original.load_graphs
    protocol = json.loads((BASE / "disjoint_protocol.json").read_text())
    allowed = set(protocol["stage1_train_ids"])
    loaded_ids = {}

    def load_disjoint(graph_dir, split):
        graphs = original_loader(graph_dir, split)
        if split == "train":
            graphs = [g for g in graphs if str(g.sample_id) in allowed]
        if a.smoke:
            graphs = graphs[:64 if split == "train" else 16]
        loaded_ids[split] = [str(g.sample_id) for g in graphs]
        return graphs

    original.load_graphs = load_disjoint
    name = ("smoke" if a.smoke else "full") + f"_scorer_seed{a.seed}"
    folder = BASE / name
    folder.mkdir(parents=True, exist_ok=True)
    sys.argv = [original.__file__, "--graph-dir", str(ROOT / "data/processed/structure_supervision/interaction_graphs"),
                "--out-dir", str(folder), "--checkpoint-dir", str(folder / "checkpoints"),
                "--table", str(folder / "metrics.md"), "--seed", str(a.seed),
                "--epochs", "2" if a.smoke else "10", "--batch-size", "8", "--device", "cuda"]
    start = time.time()
    original.main()
    (folder / "actual_graph_ids.json").write_text(json.dumps(loaded_ids))
    (folder / "completion.json").write_text(json.dumps({"status": "complete", "seconds": time.time() - start,
                                                        "seed": a.seed, "smoke": a.smoke,
                                                        "graph_counts": {k: len(v) for k, v in loaded_ids.items()}}))


if __name__ == "__main__":
    main()
