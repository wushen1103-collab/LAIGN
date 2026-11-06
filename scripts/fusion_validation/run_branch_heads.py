"""Refit pure geometric, pure chemical and cross-term-free fusion heads."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'8')
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from .feature_data import load_features,geometry_columns

from .runtime import ROOT, BASE, load_archive, canonical_id


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=int,default=0);a=parser.parse_args()
    archive=load_archive(ROOT/f'outputs/reference/plinder/seed{2401+a.run}/predictions.npz')
    data={s:load_features(BASE/'cache_original',archive,s) for s in ('train','internal_val','external_val','external_test')}
    names=data['train']['names']
    geometry=geometry_columns(names).tolist()
    chemical=[i for i,name in enumerate(names) if i not in geometry and '_weighted_tau_' not in name
              and not name.startswith('visnet_x_range_tau_') and name!='visnet_over_1_plus_distance']
    assert len(geometry)==34 and len(chemical)==21
    branches={'Geometry only (34 features)':geometry,'Chemistry only (21 features)':chemical,
              'Geometry + chemistry without cross terms (55 features)':sorted(geometry+chemical)}
    out=BASE/f'branch_run{a.run}';out.mkdir(parents=True, exist_ok=True)
    arrays={};trace=[]
    for split in ('external_val','external_test'):
        for key,value in [('labels',data[split]['y']),('sample_index',data[split]['g']),('sample_ids',data[split]['ids']),('score__raw_distance',-data[split]['d'])]:arrays[split+'__'+key]=value
    for method,cols in branches.items():
        best=None
        for c in (.01,.1,1.,10.):
            model=make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight='balanced',solver='lbfgs',max_iter=1000,random_state=2401+a.run))
            model.fit(data['train']['x'][:,cols],data['train']['y'])
            value=float(average_precision_score(data['internal_val']['y'],model.predict_proba(data['internal_val']['x'][:,cols])[:,1]))
            row=dict(method=method,C=c,features=[names[i] for i in cols],internal_validation_pooled_AP=value)
            trace.append(row)
            print(json.dumps(dict(method=method,C=c,validation_AP=value)),flush=True)
            if best is None or value>best[0]:best=(value,model,row)
        joblib.dump(best[1],out/(str(len(cols))+'_feature_head.joblib'))
        for split in ('external_val','external_test'):
            arrays[split+'__score__'+method]=best[1].predict_proba(data[split]['x'][:,cols])[:,1]
    np.savez_compressed(out/'external_predictions.npz',**arrays)
    (out/'selection_trace.json').write_text(json.dumps(trace,indent=2))
    (out/'completion.json').write_text(json.dumps(dict(status='complete',run=a.run,external_tuning=False),indent=2))


if __name__=='__main__':main()
