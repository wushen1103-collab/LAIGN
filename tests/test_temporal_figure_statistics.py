from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from figures.temporal_statistics import annotation_text, temporal_summary


ROOT = Path(__file__).resolve().parents[1]


def test_published_scatter_statistics_match_the_corrected_annotation():
    frame = pd.read_csv(ROOT / 'source_data/derived/fig3_temporal_scatter.csv')
    summary = temporal_summary(frame)
    assert summary['n'] == 70
    assert np.isclose(summary['delta'], 0.009714178402794501, rtol=0, atol=1e-12)
    assert np.isclose(summary['ci95_low'], -0.006374299913225312, rtol=0, atol=1e-12)
    assert np.isclose(summary['ci95_high'], 0.025256214840576555, rtol=0, atol=1e-12)
    assert annotation_text(summary) == '$N=70$\n$\\Delta=+0.0097$\n95% CI [-0.0064, 0.0253]'


def test_complex_pairing_is_retained_in_bootstrap_samples():
    frame = pd.DataFrame(dict(sample_id=['a', 'b', 'c'], graph_ap_laign=[0.2, 0.5, 0.8],
                              graph_ap_fixed_fusion=[0.1, 0.4, 0.7]))
    summary = temporal_summary(frame, resamples=251)
    assert np.allclose([summary[key] for key in ['delta', 'ci95_low', 'ci95_high']], 0.1)
    assert summary['bootstrap_replicates'] == 251


def test_duplicate_complexes_are_not_silently_counted_twice():
    frame = pd.DataFrame(dict(sample_id=['a', 'a'], graph_ap_laign=[0.8, 0.9],
                              graph_ap_fixed_fusion=[0.7, 0.6]))
    with pytest.raises(ValueError, match='unique complex'):
        temporal_summary(frame)


def test_invalid_values_are_not_silently_dropped():
    frame = pd.DataFrame(dict(sample_id=['a'], graph_ap_laign=[np.nan],
                              graph_ap_fixed_fusion=[0.7]))
    with pytest.raises(ValueError, match='finite'):
        temporal_summary(frame)


def test_float_roundoff_is_accepted_without_clipping_the_scores():
    value = np.nextafter(1.0, 2.0)
    frame = pd.DataFrame(dict(sample_id=['a'], graph_ap_laign=[value],
                              graph_ap_fixed_fusion=[1.0]))
    assert temporal_summary(frame, resamples=10)['delta'] == value - 1.0


def test_substantively_out_of_range_ap_is_rejected():
    frame = pd.DataFrame(dict(sample_id=['a'], graph_ap_laign=[1.01],
                              graph_ap_fixed_fusion=[0.7]))
    with pytest.raises(ValueError, match='between zero and one'):
        temporal_summary(frame)
