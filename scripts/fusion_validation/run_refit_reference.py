"""Fit an explicit same-cache full reference for the new channel ablations."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '8')
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .feature_data import load_features
from .fusion_metrics import per_complex

from .runtime import ROOT, BASE, load_archive, canonical_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, required=True)
    parser.add_argument('--cohort', choices=['plinder', 'postfreeze'], default='plinder')
    args = parser.parse_args()
    start = time.time()
    seed = 2401 + args.run
    folder = (ROOT / f'outputs/reference/plinder/seed{seed}' if args.cohort == 'plinder'
              else ROOT / f'outputs/reference/postfreeze/replicate_{"ABC"[args.run]}')
    archive = load_archive(folder / 'predictions.npz')
    cache = BASE / ('cache_original' if args.cohort == 'plinder' else 'cache_postfreeze_' + 'ABC'[args.run])
    suffix = '' if args.cohort == 'plinder' else 'postfreeze_'
    out = BASE / f'refit_{suffix}run{args.run}'
    out.mkdir(parents=True, exist_ok=True)
    train = load_features(cache, archive, 'train')
    validation = load_features(cache, archive, 'internal_val')
    trace, best = [], None
    for c in (.01, .1, 1., 10.):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight='balanced',
                              max_iter=1000, solver='lbfgs', random_state=seed))
        model.fit(train['x'], train['y'])
        score = float(average_precision_score(validation['y'], model.predict_proba(validation['x'])[:, 1]))
        row = dict(C=c, internal_validation_pooled_AP=score,
                   solver_iterations=int(model[-1].n_iter_[0]))
        trace.append(row)
        print(json.dumps(row), flush=True)
        if best is None or score > best[0]:
            best = score, model, row
    joblib.dump(best[1], out / 'reference73.joblib')
    del train, validation
    arrays, checks = {}, []
    for split in ('external_val', 'external_test'):
        if not len(archive[split + '__sample_ids']):
            continue
        data = load_features(cache, archive, split)
        prediction = best[1].predict_proba(data['x'])[:, 1]
        for key, value in [('labels', data['y']), ('sample_index', data['g']), ('sample_ids', data['ids']),
                           ('score__raw_distance', -data['d']), ('score__LAIGN refit (73 features)', prediction)]:
            arrays[split + '__' + key] = value
        frozen = archive[split + '__score__trained_stage2']
        now = per_complex(data['y'], prediction, data['g'], data['ids'], split, 'refit')
        old = per_complex(data['y'], frozen, data['g'], data['ids'], split, 'frozen')
        checks.append(dict(split=split, graph_AP_refit=float(np.nanmean([r['graph_AP'] for r in now])),
                           graph_AP_frozen=float(np.nanmean([r['graph_AP'] for r in old])),
                           score_max_absolute_difference=float(np.max(np.abs(prediction - frozen)))))
    np.savez_compressed(out / 'external_predictions.npz', **arrays)
    (out / 'selection_trace.json').write_text(json.dumps(trace, indent=2))
    (out / 'completion.json').write_text(json.dumps(dict(status='complete', run=args.run,
        cohort=args.cohort, feature_count=73, selected=best[2], external_tuning=False,
        purpose='Same-cache, same-grid reference; original frozen predictions remain separately reported',
        frozen_comparison=checks, seconds=time.time()-start), indent=2))


if __name__ == '__main__':
    main()
