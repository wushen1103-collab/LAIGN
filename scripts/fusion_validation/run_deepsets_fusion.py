"""Typed pair-set DeepSets with mean/max pooling and matched geometry."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'8')
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score
from .feature_data import cache_path,geometry_columns

from .runtime import ROOT, BASE, load_archive, canonical_id


class PairSetData:
    def __init__(self,cache,ids):
        ys,geometries,pairs,counts,distances,groups=[],[],[],[],[],[]
        self.ids=np.asarray(ids,dtype=str)
        for i,sid in enumerate(ids):
            with load_archive(cache_path(cache,sid)) as z:
                cols=geometry_columns(z['feature_names'].astype(str))
                assert len(cols)==34
                ys.append(z['y'])
                geometries.append(z['x'][:,cols])
                distances.append(z['distance'])
                pairs.append(np.column_stack([z['pair_prob'],z['pair_distance'][:,0]/12]).astype(np.float32))
                counts.append(np.bincount(z['residue_for_pair'],minlength=len(z['y'])))
                assert np.array_equal(z['residue_for_pair'],np.repeat(np.arange(len(z['y'])),counts[-1]))
                groups.append(np.full(len(z['y']),i,dtype=np.int32))
        self.y=np.concatenate(ys).astype(np.float32)
        self.geometry=np.concatenate(geometries)
        self.pairs=np.concatenate(pairs)
        self.count=np.concatenate(counts)
        self.ptr=np.r_[0,np.cumsum(self.count)]
        self.distance=np.concatenate(distances)
        self.group=np.concatenate(groups)

    def batch(self,indices,mean,std,device):
        count=self.count[indices]
        group=np.repeat(np.arange(len(indices)),count)
        local=np.arange(int(count.sum()))-np.repeat(np.r_[0,np.cumsum(count)[:-1]],count)
        pair_index=np.repeat(self.ptr[indices],count)+local
        return (torch.from_numpy(self.pairs[pair_index]).to(device),
                torch.from_numpy(group).to(device),
                torch.from_numpy(count.astype(np.float32)).to(device),
                torch.from_numpy((self.geometry[indices]-mean)/std).to(device),
                torch.from_numpy(self.y[indices]).to(device))


class DeepSets(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.phi=nn.Sequential(nn.Linear(6,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.rho=nn.Sequential(nn.Linear(2*width+34,64),nn.SiLU(),nn.Dropout(.1),nn.Linear(64,1))

    def forward(self,pairs,groups,counts,geometry):
        embedded=self.phi(pairs)
        n=geometry.shape[0]
        expand=groups[:,None].expand(-1,embedded.shape[1])
        summed=torch.zeros(n,embedded.shape[1],device=embedded.device,dtype=embedded.dtype)
        summed.scatter_add_(0,expand,embedded)
        maximum=torch.full_like(summed,-torch.inf)
        maximum.scatter_reduce_(0,expand,embedded,reduce='amax',include_self=True)
        pooled=torch.cat([summed/counts[:,None],maximum,geometry],dim=1)
        return self.rho(pooled).squeeze(-1)


@torch.no_grad()
def predict(model,data,mean,std,device,batch_size=2048):
    model.eval()
    values=[]
    for start in range(0,len(data.y),batch_size):
        batch=data.batch(np.arange(start,min(start+batch_size,len(data.y))),mean,std,device)
        values.append(torch.sigmoid(model(*batch[:4])).cpu().numpy())
    return np.concatenate(values)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=int,default=0)
    parser.add_argument('--cohort',choices=['plinder','postfreeze'],default='plinder')
    parser.add_argument('--smoke',action='store_true')
    a=parser.parse_args()
    torch.set_num_threads(8)
    seed=(2401 if a.cohort=='plinder' else 3401)+a.run
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    archive=load_archive(ROOT/f'outputs/reference/plinder/seed{seed}/predictions.npz'
                    if a.cohort=='plinder' else ROOT/f'outputs/reference/postfreeze/replicate_{"ABC"[a.run]}/predictions.npz')
    cache=BASE/('cache_original' if a.cohort=='plinder' else 'cache_postfreeze_'+'ABC'[a.run])
    suffix='' if a.cohort=='plinder' else 'postfreeze_'
    out=BASE/(('smoke_' if a.smoke else '')+f'deepsets_{suffix}run{a.run}')
    out.mkdir(parents=True, exist_ok=True)
    start=time.time()
    data={}
    for split in ('train','internal_val','external_val','external_test'):
        ids=archive[split+'__sample_ids'].astype(str)
        if not len(ids):continue
        if a.smoke:ids=ids[:40 if split=='train' else 12]
        data[split]=PairSetData(cache,ids)
        if not a.smoke:
            assert np.array_equal(data[split].y,archive[split+'__labels'])
        print(json.dumps(dict(split=split,residues=len(data[split].y),pairs=len(data[split].pairs))),flush=True)
    mean=data['train'].geometry.mean(axis=0,dtype=np.float64).astype(np.float32)
    std=data['train'].geometry.std(axis=0,dtype=np.float64).astype(np.float32)
    std=np.maximum(std,1e-6)
    device=torch.device(os.environ.get("LAIGN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    trace=[]
    best=None
    configurations=[(32,.001),(64,.0003)]
    for width,lr in configurations[:1] if a.smoke else configurations:
        model=DeepSets(width).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
        positive=float(data['train'].y.sum())
        pos_weight=torch.tensor((len(data['train'].y)-positive)/positive,device=device)
        loss_fn=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        rng=np.random.default_rng(seed)
        for epoch in range(2 if a.smoke else 10):
            model.train()
            order=rng.permutation(len(data['train'].y))
            losses=[]
            for first in range(0,len(order),2048):
                batch=data['train'].batch(order[first:first+2048],mean,std,device)
                opt.zero_grad(set_to_none=True)
                loss=loss_fn(model(*batch[:4]),batch[4])
                if not torch.isfinite(loss):raise FloatingPointError('DeepSets loss')
                loss.backward();opt.step();losses.append(float(loss.detach()))
            score=predict(model,data['internal_val'],mean,std,device)
            val=float(average_precision_score(data['internal_val'].y,score))
            row=dict(width=width,lr=lr,epoch=epoch+1,loss=float(np.mean(losses)),internal_validation_pooled_AP=val)
            trace.append(row);print(json.dumps(row),flush=True)
            if best is None or val>best[0]:
                best=(val,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},row.copy())
    model=DeepSets(best[2]['width']).to(device)
    model.load_state_dict(best[1])
    arrays={}
    for split in ('external_val','external_test'):
        if split not in data:continue
        ds=data[split]
        arrays[split+'__labels']=ds.y.astype(np.int8)
        arrays[split+'__sample_index']=ds.group
        arrays[split+'__sample_ids']=ds.ids
        arrays[split+'__score__raw_distance']=-ds.distance
        arrays[split+'__score__Typed pair-set DeepSets']=predict(model,ds,mean,std,device)
    np.savez_compressed(out/'external_predictions.npz',**arrays)
    torch.save(dict(state_dict=best[1],selected=best[2],geometry_mean=mean,geometry_std=std,seed=seed),out/'model.pt')
    (out/'selection_trace.json').write_text(json.dumps(trace,indent=2))
    (out/'completion.json').write_text(json.dumps(dict(status='complete',run=a.run,smoke=a.smoke,
                                                      seconds=time.time()-start,parameters=sum(p.numel() for p in model.parameters()),
                                                      selection=best[2],external_tuning=False),indent=2))


if __name__=='__main__':main()
