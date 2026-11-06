"""Executable small-data checks, never used as paper experiment results."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_public_fusion_training_smoke(tmp_path):
    for package in ('lightgbm', 'torch', 'torch_geometric'):
        if importlib.util.find_spec(package) is None:
            pytest.skip('Full fusion smoke requires ' + package)
    from scripts.model.train_residue_localizer import typed_multiscale_features, feature_matrix, DEFAULT_ACTIVE_INTERACTIONS
    from scripts.fusion_validation.feature_data import cache_path
    from scripts.fusion_validation.runtime import canonical_id, load_archive

    rng = np.random.default_rng(914)
    workspace = tmp_path / 'workspace'
    output = workspace / 'outputs/fusion_validation'
    cache = output / 'cache_original'
    cache.mkdir(parents=True)
    arrays = {}
    for split, count in [('train',8), ('internal_val',3), ('external_val',2), ('external_test',2)]:
        ids, labels, distances, scalar, groups = [], [], [], [], []
        for i in range(count):
            sid = f'synthetic:{split}:{i}'
            ids.append(sid)
            y = np.tile([0,1],16).astype(np.int8)
            d = rng.uniform(1,10,32)
            p = np.clip(rng.uniform(0.01,0.8,(96,5)) + np.repeat(y,3)[:,None]*0.1,0,1).astype(np.float32)
            rp = np.repeat(np.arange(32),3)
            pd = np.column_stack([np.repeat(d,3),np.repeat(d+1,3)])
            record = {'labels':y,'residue_for_pair':rp,'pair_distances':pd,'distances':d}
            extra, names = typed_multiscale_features(record,p,DEFAULT_ACTIVE_INTERACTIONS)
            v = p.max(axis=1).reshape(32,3).max(axis=1)
            x, fn = feature_matrix({'distances':d,'visnet':v,'extra_features':extra,'extra_feature_names':names},'stage2','typed_multiscale')
            assert x.shape == (32,73)
            np.savez_compressed(cache_path(cache,sid),sample_id=np.asarray(sid),y=y,distance=d,x=x,
                pair_prob=p,pair_distance=pd,residue_for_pair=rp,feature_names=np.asarray(fn),
                interaction_names=np.asarray(DEFAULT_ACTIVE_INTERACTIONS))
            labels.append(y);distances.append(d);scalar.append(v);groups.append(np.full(32,i))
        arrays[split+'__sample_ids'] = np.asarray(ids)
        arrays[split+'__sample_index'] = np.concatenate(groups)
        arrays[split+'__labels'] = np.concatenate(labels)
        arrays[split+'__score__raw_distance'] = -np.concatenate(distances)
        v = np.concatenate(scalar)
        arrays[split+'__score__raw_visnet'] = v
        arrays[split+'__score__trained_stage2'] = v
        arrays[split+'__score__fixed_hybrid'] = .55*v+.45*np.exp(-np.concatenate(distances)/12)
    reference = workspace / 'outputs/reference/plinder/seed2401'
    reference.mkdir(parents=True)
    np.savez_compressed(reference/'predictions.npz',**arrays)
    with load_archive(reference/'predictions.npz') as archive:
        assert set(dict(archive)) == set(arrays)
        assert np.array_equal(archive['train__labels'], arrays['train__labels'])
    assert canonical_id('biolip2_nr:1cvrA_H37_A_1') == 'biolip2_nr:1cvrA_H37_A_1'
    env = dict(os.environ,LAIGN_WORKSPACE=str(workspace),LAIGN_DEVICE='cpu',
               PYTHONPATH=os.pathsep.join(str(p) for p in sys.path if p))
    commands = [
        ['run_twoinput_fusion','--smoke'],
        ['run_feature_heads','--smoke','--run','0'],
        ['run_deepsets_fusion','--smoke','--run','0'],
    ]
    for command in commands:
        result = subprocess.run([sys.executable,'-m','scripts.fusion_validation.'+command[0],*command[1:]],
            cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    for name in ('smoke_twoinput_plinder','smoke_feature_head_run0','smoke_deepsets_run0'):
        complete = json.loads((output/name/'completion.json').read_text())
        assert complete['status'] == 'complete' and complete['smoke']
        assert complete['external_tuning'] is False
        with np.load(output/name/'external_predictions.npz') as z:
            for key in z.files:
                if '__score__' in key:
                    assert np.isfinite(z[key]).all()
