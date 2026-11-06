"""Aggregate completed head runs; keep per-run and probability-ensemble results distinct."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .fusion_metrics import per_complex,mean_summary,paired_summary

from .runtime import ROOT, BASE, load_archive, canonical_id


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cohort',choices=['plinder','postfreeze'],default='plinder')
    a=p.parse_args()
    suffix='' if a.cohort=='plinder' else 'postfreeze_'
    all_metrics=[]
    families=[('feature_head_','full_feature_'),('deepsets_','deepsets_'),('refit_','refit_')]
    if a.cohort=='plinder':families.append(('branch_','branch_'))
    for source,destination in families:
        folders=[BASE/f'{source}{suffix}run{r}' for r in range(3)]
        if not all((f/'completion.json').exists() for f in folders):continue
        archives=[load_archive(f/'external_predictions.npz') for f in folders]
        reference=archives[0]
        output={k:reference[k] for k in reference.files if '__score__' not in k or k.endswith('__score__raw_distance')}
        for key in reference.files:
            if '__score__' not in key or key.endswith('__score__raw_distance'):
                assert all(np.array_equal(reference[key],z[key]) for z in archives)
                continue
            output[key]=np.mean([z[key].astype(np.float64) for z in archives],axis=0)
            split,method=key.split('__score__',1)
            for run,z in enumerate(archives):
                rows=per_complex(z[split+'__labels'],z[key],z[split+'__sample_index'],z[split+'__sample_ids'],split,method)
                all_metrics.extend(dict(row,run=run,family=source) for row in rows)
        target=BASE/(destination+('original' if a.cohort=='plinder' else 'postfreeze'))
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target/'external_predictions.npz',**output)
        (target/'completion.json').write_text(json.dumps(dict(status='complete',runs=3,
                                                             aggregation='Arithmetic mean of aligned predicted probabilities',
                                                             cohort=a.cohort),indent=2))
    if all_metrics:
        frame=pd.DataFrame(all_metrics)
        frame.to_csv(BASE/f'{a.cohort}_fusion_per_run_metrics.csv',index=False)
        perrun=frame.groupby(['family','cohort','method','run']).graph_AP.mean().reset_index()
        perrun.to_csv(BASE/f'{a.cohort}_fusion_per_run_graph_AP.csv',index=False)
        perrun.groupby(['family','cohort','method']).graph_AP.agg(['mean','std','count']).reset_index().to_csv(BASE/f'{a.cohort}_fusion_seed_summary.csv',index=False)


if __name__=='__main__':main()
