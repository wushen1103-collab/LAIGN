#!/usr/bin/env python3
"""Run LABind on an already prepared PLINDER work directory.

This avoids the official prediction entrypoint's unconditional ESMFold/MoLFormer
downloads. It still uses LABind's official feature and model functions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labind-root",
        default=str(ROOT / ".tools/deep_baselines/src/LABind-main"),
    )
    parser.add_argument(
        "--work-dir",
        default=str(ROOT / "outputs/external_direct_sota_benchmark_plinder200_prep/work/out"),
    )
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument(
        "--gpu-ids",
        default="0",
        help="Visible CUDA ids for LABind DataParallel, e.g. 0 or 0,1 after CUDA_VISIBLE_DEVICES remapping.",
    )
    parser.add_argument("--skip-dssp-msms", action="store_true")
    parser.add_argument("--skip-ankh", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument(
        "--summary",
        default=str(ROOT / "outputs/external_labind_benchmark_plinder200_run_summary.json"),
    )
    return parser.parse_args()


def count_files(path: Path, suffix: str) -> int:
    return sum(1 for _ in path.glob(f"*{suffix}"))


def fasta_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(">"))


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    labind_root = Path(args.labind_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    scripts_dir = labind_root / "scripts"
    fasta = work_dir / "protein.fa"
    pdb_dir = work_dir / "pdb"
    dssp_dir = work_dir / "dssp"
    pos_dir = work_dir / "pos"
    ankh_dir = work_dir / "ankh"

    for required in [scripts_dir, fasta, pdb_dir, work_dir / "ligand.pkl", work_dir / "smiles.txt"]:
        if not required.exists():
            raise FileNotFoundError(required)
    for directory in [dssp_dir, pos_dir, ankh_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    os.chdir(scripts_dir)
    sys.path.insert(0, str(scripts_dir))

    from prediction import getDSSP, getEmbed, getMSMS, prediction as labind_predict

    gpu_ids = [int(value) for value in args.gpu_ids.split(",") if value.strip()]
    if not gpu_ids:
        gpu_ids = [0]
    device = f"cuda:{gpu_ids[0]}"
    n_samples = fasta_count(fasta)
    started = time.time()

    summary = {
        "work_dir": str(work_dir),
        "samples": n_samples,
        "gpu_ids": gpu_ids,
        "started_at_unix": started,
        "stages": {},
    }

    if args.skip_dssp_msms:
        summary["stages"]["dssp_msms"] = "skipped"
    else:
        getDSSP(str(pdb_dir) + "/", dssp_path=str(labind_root / "tools/mkdssp"))
        getMSMS(str(pdb_dir) + "/", msms_path=str(labind_root / "tools/msms"))
        summary["stages"]["dssp_msms"] = {
            "dssp_npy": count_files(dssp_dir, ".npy"),
            "pos_npy": count_files(pos_dir, ".npy"),
        }
        write_summary(Path(args.summary), summary)

    if args.skip_ankh:
        summary["stages"]["ankh"] = "skipped"
    else:
        getEmbed(
            str(fasta),
            embed_path=str(labind_root / "tools/ankh-large"),
            device=device,
            out_path=str(ankh_dir) + "/",
        )
        summary["stages"]["ankh"] = {"ankh_npy": count_files(ankh_dir, ".npy")}
        write_summary(Path(args.summary), summary)

    if args.skip_predict:
        summary["stages"]["prediction"] = "skipped"
    else:
        labind_predict(
            str(fasta),
            batch_size=args.batch,
            device_ids=gpu_ids,
            out_path=str(work_dir),
            model_path=str(labind_root / "model/Unseen") + "/",
        )
        summary["stages"]["prediction"] = {
            "result_csv": str(work_dir / "RESULT.csv"),
            "result_exists": (work_dir / "RESULT.csv").exists(),
        }

    summary["elapsed_sec"] = time.time() - started
    summary["final_counts"] = {
        "dssp_npy": count_files(dssp_dir, ".npy"),
        "pos_npy": count_files(pos_dir, ".npy"),
        "ankh_npy": count_files(ankh_dir, ".npy"),
        "result_exists": (work_dir / "RESULT.csv").exists(),
    }
    write_summary(Path(args.summary), summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
