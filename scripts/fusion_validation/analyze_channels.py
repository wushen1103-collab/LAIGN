"""External type-specific pair quality, restricted to <=6 A candidates."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'8')
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .feature_data import cache_path

from .runtime import ROOT, BASE, load_archive, canonical_id


def main():
    out=BASE/'typed_channel_diagnostics'
    out.mkdir(parents=True, exist_ok=True)
    archive=load_archive(ROOT/'outputs/reference/plinder/seed2401/predictions.npz')
    ids=archive['external_test__sample_ids'].astype(str)
    labels,scores,distances=[],[],[]
    complex_rows=[]
    for sid in ids:
        with load_archive(cache_path(BASE/'cache_original',sid)) as z:
            mask=z['pair_distance'][:,0]<=6.
            y,s,d=z['pair_label'][mask],z['pair_prob'][mask],z['pair_distance'][mask,0]
            names=z['interaction_names'].astype(str)
            labels.append(y);scores.append(s);distances.append(d)
            for j,name in enumerate(names):
                if not y[:,j].sum():continue
                complex_rows.append(dict(sample_id=sid,interaction=name,candidate_pairs=len(y),positive_pairs=int(y[:,j].sum()),
                                         pair_AP=float(average_precision_score(y[:,j],s[:,j])),
                                         distance_AP=float(average_precision_score(y[:,j],-d))))
    y,s,d=np.concatenate(labels),np.concatenate(scores),np.concatenate(distances)
    detail=pd.DataFrame(complex_rows)
    detail.to_csv(out/'per_complex_type.csv',index=False)
    rows=[];rng=np.random.default_rng(8241)
    for j,name in enumerate(names):
        part=detail[detail.interaction.eq(name)]
        delta=(part.pair_AP-part.distance_AP).to_numpy()
        draws=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(axis=1)
        lo,hi=np.quantile(draws,[.025,.975])
        rows.append(dict(interaction=name,complexes_requested=len(ids),complexes_with_positive=len(part),
                         candidate_pairs=len(y),positive_pairs=int(y[:,j].sum()),positive_prevalence=float(y[:,j].mean()),
                         pair_AUPRC=float(average_precision_score(y[:,j],s[:,j])),
                         distance_only_AUPRC=float(average_precision_score(y[:,j],-d)),
                         mean_per_complex_pair_AP=float(part.pair_AP.mean()),
                         mean_per_complex_distance_AP=float(part.distance_AP.mean()),
                         paired_macro_delta=float(delta.mean()),ci95_low=float(lo),ci95_high=float(hi)))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'summary.csv',index=False)
    (out/'completion.json').write_text(json.dumps(dict(status='complete',complexes=785,
                                                       candidate_definition='all residue-ligand atom pairs whose minimum heavy-atom separation is <=6 A',
                                                       reference='PLIP typed pair annotations loaded independently of model scores',
                                                       bootstrap_unit='complex with a positive for the evaluated type',bootstrap_replicates=10000),indent=2))
    print(frame.to_string(index=False),flush=True)


if __name__=='__main__':main()
