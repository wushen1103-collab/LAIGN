"""Compute Fig. 3a annotation from the same complex scores as the scatter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def temporal_summary(frame: pd.DataFrame, resamples: int = 10_000, seed: int = 8241) -> dict:
    columns = ['sample_id', 'graph_ap_laign', 'graph_ap_fixed_fusion']
    if not set(columns).issubset(frame.columns):
        raise ValueError('Expected complex identifiers and both paired graph AP columns.')
    if frame.empty or frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError('Each scatter row must identify one unique complex.')
    scores = frame[columns[1:]].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < -1e-12) | (scores > 1 + 1e-12)).any():
        raise ValueError('Graph AP values must be finite and between zero and one.')
    if resamples < 1:
        raise ValueError('The bootstrap replicate count must be positive.')
    difference = scores[:, 0] - scores[:, 1]
    rng = np.random.default_rng(seed)
    draws = []
    # Preserve the published resampling order without allocating an unbounded array.
    for start in range(0, resamples, 250):
        indices = rng.integers(len(frame), size=(min(250, resamples - start), len(frame)))
        draws.append(difference[indices].mean(axis=1))
    low, high = np.quantile(np.concatenate(draws), [0.025, 0.975])
    return dict(n=len(frame), delta=float(difference.mean()), ci95_low=float(low),
                ci95_high=float(high), bootstrap_replicates=resamples, seed=seed,
                estimand='Mean paired per-complex AP difference from probability-ensemble scores',
                resampling_unit='protein-ligand complex')


def annotation_text(summary: dict) -> str:
    return (f"$N={summary['n']}$\n$\\Delta={summary['delta']:+.4f}$\n"
            f"95% CI [{summary['ci95_low']:.4f}, {summary['ci95_high']:.4f}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1]
                        / 'source_data/derived/fig3_temporal_scatter.csv')
    args = parser.parse_args()
    summary = temporal_summary(pd.read_csv(args.source))
    print(json.dumps(dict(summary, annotation=annotation_text(summary)), indent=2))


if __name__ == '__main__':
    main()
