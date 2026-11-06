import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_official_plinder_snapshot_and_cohort():
    folder = ROOT / 'source_data/cohorts'
    release = json.loads((folder / 'plinder_release.json').read_text())
    assert (release['release'], release['iteration']) == ('2024-06', 'v2')
    assert release['split_assignment_mismatches'] == 0
    assert hashlib.sha256((folder / 'plinder_cohorts.csv').read_bytes()).hexdigest() == release['cohort_manifest']['sha256']
    frame = pd.read_csv(folder / 'plinder_cohorts.csv')
    assert frame.sample_id.is_unique
    assert frame.supervision_split.value_counts().to_dict() == {'external_test':785, 'external_val':551}
    assert frame.loc[frame.plinder200].supervision_split.value_counts().to_dict() == {'external_test':100, 'external_val':100}
    expected = frame.supervision_split.map({'external_test':'test','external_val':'val'})
    assert (frame.upstream_split == expected).all()


def test_frozen_disjoint_memberships_have_no_overlap():
    protocol = json.loads((ROOT / 'source_data/cohorts/disjoint_protocol.json').read_text())
    assert not set(protocol['stage1_train_ids']) & set(protocol['stage2_train_ids'])
    assert len(protocol['stage2_train_ids']) == 5000
    frame = pd.read_csv(ROOT / 'source_data/cohorts/structure_splits.csv.gz')
    group = frame.set_index('sample_id').split_group_id
    assert not set(group.loc[protocol['stage1_train_ids']]) & set(group.loc[protocol['stage2_train_ids']])


def test_localizer_training_ids_and_temporal_groups():
    runs = pd.read_csv(ROOT / 'source_data/cohorts/localizer_runs.csv.gz')
    for _, frame in runs.groupby(['cohort', 'run']):
        assert (frame.split == 'train').sum() == 5000
        assert (frame.split == 'internal_val').sum() == 1000
        assert frame.sample_id.is_unique
    temporal = pd.read_csv(ROOT / 'source_data/cohorts/postfreeze_receptors.csv')
    assert len(temporal) == 70 and temporal.receptor_sequence.nunique() == 56
