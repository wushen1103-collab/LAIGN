"""Load original aligned Stage-2 inputs and type-removal counterfactuals."""
import hashlib
from pathlib import Path
import numpy as np


def cache_path(folder, sid):
    return Path(folder) / (hashlib.sha256(str(sid).encode()).hexdigest()[:24] + ".npz")


def geometry_columns(names):
    return np.asarray([i for i, name in enumerate(names) if name in ("distance", "log1p_distance")
                       or name.startswith("range_tau_") or name.startswith("typed_multiscale__ligand_atom_pair_count")
                       or name.startswith("typed_multiscale__centroid_distance")
                       or name.startswith("typed_multiscale__contact_")
                       or name.startswith("typed_multiscale__distance_soft_")])


def drop_type_matrix(z, removed):
    names = z["feature_names"].astype(str).tolist()
    types = z["interaction_names"].astype(str).tolist()
    keep_types = [i for i, t in enumerate(types) if t != removed]
    nr = len(z["y"])
    counts = np.bincount(z["residue_for_pair"], minlength=nr)
    if not np.all(counts == counts[0]):
        raise ValueError("Expected Cartesian residue-ligand atom pairs")
    nl = int(counts[0])
    pairmax = z["pair_prob"][:,keep_types].max(axis=1).reshape(nr,nl).astype(np.float64)
    pd = z["pair_distance"][:,0].reshape(nr,nl).astype(np.float64)
    scalar = pairmax.max(axis=1).astype(np.float64)
    distance = z["distance"]
    changes = {"visnet": scalar,
               "visnet_logit": np.log(np.clip(scalar,1e-5,1-1e-5)/np.clip(1-scalar,1e-5,1-1e-5)),
               "visnet_over_1_plus_distance": scalar/(1+distance),
               "typed_multiscale__any_interaction_pair_max": scalar,
               "typed_multiscale__any_interaction_pair_mean": pairmax.mean(axis=1),
               "typed_multiscale__any_interaction_top3_mean": np.sort(pairmax,axis=1)[:,-min(3,nl):].mean(axis=1)}
    for tau in (1,2,3,4,6,8,10,12,16,20):
        changes[f"visnet_x_range_tau_{tau}"]=scalar*np.exp(-distance/tau)
    for tau in (4,8):
        changes[f"typed_multiscale__any_interaction_weighted_tau_{tau}"]=(pairmax*np.exp(-pd/tau)).mean(axis=1)
    for t in (.3,.5,.7):
        count=(pairmax>=t).sum(axis=1)
        changes[f"typed_multiscale__any_interaction_count_ge_{t:g}"]=count
        changes[f"typed_multiscale__any_interaction_frac_ge_{t:g}"]=count/nl
    x=z["x"].copy()
    for i,name in enumerate(names):
        if name in changes:
            x[:,i]=changes[name]
    keep=[i for i,name in enumerate(names) if not name.startswith("typed_multiscale__"+removed+"_")]
    assert len(keep)==70
    return x[:,keep]


def load_features(cache, archive, split, removed=None, limit=None):
    ids=archive[split+"__sample_ids"].astype(str)
    if limit:
        ids=ids[:limit]
    matrices,ys,distances,groups=[],[],[],[]
    names=None
    for i,sid in enumerate(ids):
        with np.load(cache_path(cache,sid)) as z:
            matrices.append(z["x"] if removed is None else drop_type_matrix(z,removed))
            ys.append(z["y"])
            distances.append(z["distance"])
            groups.append(np.full(len(z["y"]),i,dtype=np.int32))
            names=z["feature_names"].astype(str).tolist()
    x,y,d,g=np.concatenate(matrices),np.concatenate(ys),np.concatenate(distances),np.concatenate(groups)
    if limit is None:
        assert np.array_equal(y,archive[split+"__labels"])
        assert np.array_equal(g,archive[split+"__sample_index"])
        assert np.max(np.abs(d+archive[split+"__score__raw_distance"]))<1e-8
    return dict(x=x,y=y,d=d,g=g,ids=ids,names=names)


def load_by_ids(cache,ids):
    matrices,ys,distances,groups=[],[],[],[]
    names=None
    for i,sid in enumerate(ids):
        with np.load(cache_path(cache,sid)) as z:
            matrices.append(z['x']);ys.append(z['y']);distances.append(z['distance'])
            groups.append(np.full(len(z['y']),i,dtype=np.int32))
            names=z['feature_names'].astype(str).tolist()
    return dict(x=np.concatenate(matrices),y=np.concatenate(ys),d=np.concatenate(distances),
                g=np.concatenate(groups),ids=np.asarray(ids,dtype=str),names=names)
