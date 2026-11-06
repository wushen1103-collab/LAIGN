"""Fit Stage 2 only on complexes unseen by the newly trained scorers."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'8')
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .feature_data import load_by_ids
from .fusion_metrics import per_complex,mean_summary,paired_summary

from .runtime import ROOT, BASE, load_archive, canonical_id


def matrix(data,kind):
    if kind=='full':return data['x']
    return np.column_stack([data['x'][:,12],np.exp(-data['d']/12)]).astype(np.float32)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evaluate-only',action='store_true')
    a=p.parse_args()
    start=time.time()
    out=BASE/'disjoint_stage2'
    out.mkdir(parents=True, exist_ok=True)
    cache=BASE/'cache_disjoint'
    index=pd.concat([pd.read_csv(p) for p in cache.glob('shard*of3_index.csv')],ignore_index=True).drop_duplicates('sample_id')
    config=json.loads((BASE/'disjoint_protocol.json').read_text())
    if a.evaluate_only:
        models=joblib.load(out/'models.joblib')
    else:
        tr=load_by_ids(cache,sorted(config['stage2_train_ids']))
        va_ids=sorted(index.loc[index.split.eq('internal_val'),'sample_id'])
        assert len(va_ids)==1000
        va=load_by_ids(cache,va_ids)
        models={};trace=[]
        for method,kind,family in [('LAIGN (disjoint stages)','full','linear'),
                                   ('Nonlinear two-input fusion (disjoint stages)','two','lgbm'),
                                   ('Full-feature nonlinear fusion (disjoint stages)','full','lgbm')]:
            xtr,xva=matrix(tr,kind),matrix(va,kind)
            candidates=([dict(C=c) for c in (.01,.1,1.,10.)] if family=='linear' else
                        [dict(num_leaves=l,min_child_samples=100,reg_lambda=1.,class_weight=w)
                         for l in (7,15,31) for w in (None,'balanced')])
            best=None
            for parameters in candidates:
                if family=='linear':
                    model=make_pipeline(StandardScaler(),LogisticRegression(**parameters,class_weight='balanced',max_iter=1000,solver='lbfgs',random_state=2401))
                else:
                    model=LGBMClassifier(**parameters,n_estimators=200,learning_rate=.05,n_jobs=8,verbosity=-1,random_state=2401,deterministic=True,force_col_wise=True)
                model.fit(xtr,tr['y'])
                value=float(average_precision_score(va['y'],model.predict_proba(xva)[:,1]))
                row=dict(method=method,parameters=parameters,internal_validation_pooled_AP=value)
                trace.append(row);print(json.dumps(row),flush=True)
                if best is None or value>best[0]:best=(value,model,row)
            models[method]=dict(kind=kind,model=best[1],selected=best[2])
        joblib.dump(models,out/'models.joblib')
        (out/'selection_trace.json').write_text(json.dumps(trace,indent=2))
        del tr,va
    selected=set(pd.read_csv(ROOT/'source_data/cohorts/plinder200.csv',usecols=['sample_id']).sample_id)
    cases={split:sorted(index.loc[index.split.eq(split),'sample_id']) for split in ('external_val','external_test')}
    temporal=pd.read_csv(ROOT/'source_data/cohorts/postfreeze_receptors.csv',low_memory=False)
    temporal_ids=sorted(temporal.loc[temporal.supervision_split.eq('external_test'),'sample_id'].astype(str))
    from .feature_data import cache_path
    if all(cache_path(cache,s).exists() for s in temporal_ids):cases['postfreeze']=temporal_ids
    rows=[];arrays={}
    for split,ids in cases.items():
        data=load_by_ids(cache,ids)
        values={name:item['model'].predict_proba(matrix(data,item['kind']))[:,1] for name,item in models.items()}
        values.update({'Distance (disjoint stages)':-data['d'],
                       'Fixed fusion (disjoint stages)':.55*data['x'][:,12]+.45*np.exp(-data['d']/12)})
        for key,value in [('labels',data['y']),('sample_index',data['g']),('sample_ids',data['ids']),('score__raw_distance',-data['d'])]:arrays[split+'__'+key]=value
        cohort='Post-freeze-70 disjoint' if split=='postfreeze' else ('PLINDER-785 test disjoint' if split=='external_test' else 'PLINDER validation disjoint')
        for method,score in values.items():
            arrays[split+'__score__'+method]=score
            result=per_complex(data['y'],score,data['g'],data['ids'],cohort,method)
            rows.extend(result)
            if split!='postfreeze':
                for radius in (None,6.):
                    local=per_complex(data['y'],score,data['g'],data['ids'],'PLINDER-200 disjoint'+(' / 6A' if radius else ''),method,data['d'],radius)
                    rows.extend(r for r in local if r['sample_id'] in selected)
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'per_complex.csv',index=False)
    mean_summary(frame).to_csv(out/'summary.csv',index=False)
    paired_summary(frame,reference='LAIGN (disjoint stages)').to_csv(out/'paired_effects.csv',index=False)
    np.savez_compressed(out/'external_predictions.npz',**arrays)
    (out/'completion.json').write_text(json.dumps(dict(status='complete',seconds=time.time()-start,
                                                       stage1_graphs_manifest=len(config["stage1_train_ids"]),stage2_training_graphs=5000,
                                                       group_overlap=0,scorer_seeds=[2401,2402,2403],
                                                       temporal_evaluated='postfreeze' in cases,external_tuning=False),indent=2))
    print(mean_summary(frame).to_string(index=False),flush=True)


if __name__=='__main__':main()
