"""Same 73 inputs for nonlinear fusion, plus refitted type-removal heads."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(key,"8")
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .feature_data import load_features

from .runtime import ROOT, BASE, load_archive, canonical_id
TYPES=['hydrophobic_contact','hydrogen_bond','salt_bridge','pi_stacking','pi_cation']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=int,default=0)
    p.add_argument('--cohort',choices=['plinder','postfreeze'],default='plinder')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--skip-ablation',action='store_true')
    a=p.parse_args()
    archive=load_archive(ROOT/f'outputs/reference/plinder/seed{2401+a.run}/predictions.npz'
                    if a.cohort=='plinder' else ROOT/f'outputs/reference/postfreeze/replicate_{"ABC"[a.run]}/predictions.npz')
    cache=BASE/('cache_original' if a.cohort=='plinder' else 'cache_postfreeze_'+'ABC'[a.run])
    suffix='' if a.cohort=='plinder' else 'postfreeze_'
    out=BASE/(('smoke_' if a.smoke else '')+f'feature_head_{suffix}run{a.run}')
    out.mkdir(parents=True, exist_ok=True)
    start=time.time()
    limits={'train':50,'internal_val':30,'external_val':8,'external_test':8} if a.smoke else {}
    datasets={s:load_features(cache,archive,s,limit=limits.get(s)) for s in ('train','internal_val','external_val','external_test') if len(archive[s+'__sample_ids'])}
    arrays={}
    for split in ('external_val','external_test'):
        if split not in datasets:continue
        data=datasets[split]
        arrays[split+'__labels']=data['y']
        arrays[split+'__sample_index']=data['g']
        arrays[split+'__sample_ids']=data['ids']
        arrays[split+'__score__raw_distance']=-data['d']
    trace=[]
    configs=[(15,100,1.),(31,100,1.),(31,300,10.),(63,300,10.)]
    best=None
    for leaves,minimum,l2 in configs[:1] if a.smoke else configs:
        for weight in (None,'balanced'):
            model=LGBMClassifier(n_estimators=20 if a.smoke else 200,learning_rate=.05,num_leaves=leaves,
                                 min_child_samples=minimum,reg_lambda=l2,class_weight=weight,n_jobs=8,
                                 verbosity=-1,random_state=2401+a.run,deterministic=True,force_col_wise=True)
            model.fit(datasets['train']['x'],datasets['train']['y'])
            val=float(average_precision_score(datasets['internal_val']['y'],model.predict_proba(datasets['internal_val']['x'])[:,1]))
            row=dict(method='Full-feature nonlinear fusion',leaves=leaves,min_child_samples=minimum,
                     reg_lambda=l2,class_weight=weight,internal_validation_pooled_AP=val)
            trace.append(row); print(json.dumps(row),flush=True)
            if best is None or val>best[0]:best=(val,model,row)
    joblib.dump(best[1],out/'full_feature_lightgbm.joblib')
    for split in ('external_val','external_test'):
        if split not in datasets:continue
        arrays[split+'__score__Full-feature nonlinear fusion']=best[1].predict_proba(datasets[split]['x'])[:,1]
    del datasets
    if not a.skip_ablation and a.cohort=='plinder':
        for removed in TYPES:
            tr=load_features(cache,archive,'train',removed,limits.get('train'))
            va=load_features(cache,archive,'internal_val',removed,limits.get('internal_val'))
            best=None
            for c in ([.1] if a.smoke else [.01,.1,1.,10.]):
                model=make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight='balanced',max_iter=1000,solver='lbfgs',random_state=2401+a.run))
                model.fit(tr['x'],tr['y'])
                val=float(average_precision_score(va['y'],model.predict_proba(va['x'])[:,1]))
                row=dict(method='Without '+removed,C=c,internal_validation_pooled_AP=val,feature_count=tr['x'].shape[1],
                         removal='Channel-specific features removed and every shared any-channel aggregate recomputed without that channel')
                trace.append(row);print(json.dumps(row),flush=True)
                if best is None or val>best[0]:best=(val,model,row)
            joblib.dump(best[1],out/('without_'+removed+'.joblib'))
            for split in ('external_val','external_test'):
                test=load_features(cache,archive,split,removed,limits.get(split))
                arrays[split+'__score__Without '+removed]=best[1].predict_proba(test['x'])[:,1]
            del tr,va
    np.savez_compressed(out/'external_predictions.npz',**arrays)
    (out/'selection_trace.json').write_text(json.dumps(trace,indent=2))
    (out/'completion.json').write_text(json.dumps(dict(status='complete',run=a.run,smoke=a.smoke,
                                                     seconds=time.time()-start,external_tuning=False,
                                                     selection_endpoint='BioLiP2 internal validation pooled AP'),indent=2))


if __name__=='__main__':main()
