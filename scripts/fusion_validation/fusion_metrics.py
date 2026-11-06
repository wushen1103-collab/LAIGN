"""Shared residue-ranking metrics and complex-level paired uncertainty."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def graph_metrics(y, score):
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    if not np.isfinite(score).all() or y.shape != score.shape:
        raise ValueError("Invalid aligned scores")
    positives = int(y.sum())
    if not positives:
        return {"graph_AP": np.nan, "IoU_at_P": np.nan, "recall_at_10": np.nan}
    order = np.argsort(-score, kind="stable")
    tp_p = int(y[order[:positives]].sum())
    return {"graph_AP": float(average_precision_score(y, score)),
            "IoU_at_P": tp_p / (2 * positives - tp_p),
            "recall_at_10": float(y[order[:10]].sum() / positives)}


def per_complex(y, score, groups, ids, cohort, method, distance=None, radius=None):
    rows = []
    boundaries = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, len(groups)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        index = int(groups[start])
        mask = np.ones(stop - start, dtype=bool) if radius is None else distance[start:stop] <= radius
        yy = y[start:stop][mask]
        ss = score[start:stop][mask]
        rows.append({"sample_id": str(ids[index]), "cohort": cohort, "method": method,
                     "residues": len(yy), "positives": int(yy.sum()), **graph_metrics(yy, ss)})
    return rows


def paired_summary(frame, reference="LAIGN", bootstrap=10000, seed=8241):
    rng = np.random.default_rng(seed)
    rows = []
    for cohort, part in frame.groupby("cohort", sort=False):
        for metric in ("graph_AP", "IoU_at_P", "recall_at_10"):
            wide = part.pivot(index="sample_id", columns="method", values=metric)
            if reference not in wide:
                continue
            for method in wide.columns:
                if method == reference:
                    continue
                paired = wide[[reference, method]].dropna()
                diff = (paired[reference] - paired[method]).to_numpy()
                if not len(diff):
                    continue
                means = []
                flips = []
                for start in range(0, bootstrap, 250):
                    size = min(250, bootstrap - start)
                    means.extend(diff[rng.integers(len(diff), size=(size, len(diff)))].mean(axis=1))
                    signs = rng.choice([-1, 1], size=(size, len(diff)))
                    flips.extend((signs * diff).mean(axis=1))
                low, high = np.quantile(means, [0.025, 0.975])
                p = (1 + np.count_nonzero(np.abs(flips) >= abs(diff.mean()) - 1e-15)) / (bootstrap + 1)
                rows.append({"cohort": cohort, "metric": metric, "reference": reference, "comparator": method,
                             "paired_complexes": len(diff), "delta": float(diff.mean()),
                             "ci95_low": float(low), "ci95_high": float(high), "signflip_p": float(p),
                             "bootstrap_replicates": bootstrap})
    return pd.DataFrame(rows)


def mean_summary(frame):
    return frame.groupby(["cohort", "method"], sort=False).agg(
        complexes=("sample_id", "count"),
        evaluated_graph_AP=("graph_AP", "count"),
        graph_AP=("graph_AP", "mean"),
        graph_AP_complex_SD=("graph_AP", "std"),
        IoU_at_P=("IoU_at_P", "mean"), recall_at_10=("recall_at_10", "mean"),
        mean_residues=("residues", "mean"), mean_positives=("positives", "mean"),
    ).reset_index()


def clustered_paired_summary(frame,groups,reference='LAIGN',bootstrap=10000,seed=98241):
    rng=np.random.default_rng(seed)
    rows=[]
    for cohort,part in frame.groupby('cohort',sort=False):
        for metric in ('graph_AP','IoU_at_P','recall_at_10'):
            wide=part.pivot(index='sample_id',columns='method',values=metric)
            for method in wide.columns:
                if method==reference:continue
                paired=wide[[reference,method]].dropna()
                difference=paired[reference]-paired[method]
                mapped=groups.reindex(paired.index)
                if mapped.isna().any():raise ValueError('Missing receptor group')
                table=pd.DataFrame({'difference':difference,'group':mapped})
                clusters=table.groupby('group').difference.agg(['sum','size'])
                sums=clusters['sum'].to_numpy();sizes=clusters['size'].to_numpy();n=len(sums)
                sample=rng.integers(n,size=(bootstrap,n))
                draws=sums[sample].sum(axis=1)/sizes[sample].sum(axis=1)
                signs=rng.choice([-1,1],size=(bootstrap,n))
                null=(signs*sums).sum(axis=1)/sizes.sum()
                lo,hi=np.quantile(draws,[.025,.975])
                rows.append(dict(cohort=cohort,metric=metric,reference=reference,comparator=method,
                                 complexes=len(paired),receptor_sequence_groups=n,delta=float(difference.mean()),
                                 ci95_low=float(lo),ci95_high=float(hi),
                                 cluster_signflip_p=float((1+np.count_nonzero(np.abs(null)>=abs(difference.mean())-1e-15))/(bootstrap+1))))
    return pd.DataFrame(rows)
