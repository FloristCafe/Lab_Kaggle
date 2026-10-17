"""Unit-sampled common protocol: same compute budget for all controlled arms."""
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from agents.strategic.data import MASKS, VALUES, history_at
from .labels import replan_targets


def unpack_review(z):
    if int(z['schema_version']) != 5:
        raise ValueError('require clean review dataset schema 5')
    result={key:z[key] for key in z.files if key not in ('feature_masks','feature_values')}
    f=np.zeros((len(z['steps']),21,24,24),np.float32)
    f[:,MASKS],f[:,VALUES]=z['feature_masks'],z['feature_values']
    result['features']=f
    return result


def corrupt_context(units, seed, probability=1.):
    """Perturb ONLY existing causal memory. No target/segment information used."""
    result=units.copy()
    rng=np.random.default_rng(seed)
    for u in range(len(units)):
        if units[u,3]<.5 or rng.random()>=probability:
            continue
        choice=int(rng.integers(5))
        if choice==0:
            result[u,10]=1  # stale memory; never makes fresh features stale
        elif choice in (1,2):
            shift=rng.integers(-3,4,2) if choice==1 else rng.integers(-1,2,2)
            result[u,8:10]=np.clip(np.rint(units[u,8:10]*23)+shift,0,23)/23
        elif choice==3:
            result[u,8:17]=0
        else:
            result[u,12:17]=0;result[u,12+int(rng.integers(5))]=1
    return result.astype(np.float16).astype(np.float32)


class ReviewDataset(Dataset):
    def __init__(self, paths, config, training=False):
        self.paths=list(map(Path,paths));self.config=config;self.training=training
        self.cache=OrderedDict();self.epoch=0;rows=[];modes=[]
        self.view=config['study']['label_view'] if training else 'clean'
        prefix='dirty_' if self.view=='dirty' else ''
        for shard,path in enumerate(self.paths):
            with np.load(path,allow_pickle=False) as z:
                valid=z[prefix+'label_valid'].copy()
                if config['study']['scope']=='review_epochs':
                    valid|=z['replan_valid']
                    valid&=z['review_frames'][:,None]
                else:
                    valid[np.arange(len(valid))%config['study']['frame_stride']!=0]=False
                t,u=np.where(valid)
                rows.extend(zip(np.full(len(t),shard),t,u))
                modes.extend(z[prefix+'mode'][t,u].tolist())
        self.rows=np.asarray(rows,np.int32).reshape(-1,3)
        self.modes=np.asarray(modes,np.int8)
        if len(self.modes) and ((self.modes<0)|(self.modes>=5)).any():
            raise ValueError('sampled unit has no reliable semantic mode')
        self.mode_counts=np.bincount(self.modes,minlength=5)

    def __len__(self):return len(self.rows)

    def load(self,shard):
        if shard not in self.cache:
            with np.load(self.paths[shard],allow_pickle=False) as z:self.cache[shard]=unpack_review(z)
            while len(self.cache)>1:self.cache.popitem(last=False)
        self.cache.move_to_end(shard)
        return self.cache[shard]

    def __getitem__(self,index):
        shard,t,u=map(int,self.rows[index]);d=self.load(shard)
        prefix='dirty_' if self.view=='dirty' else ''
        units=d['unit_vectors'][t].astype(np.float32)
        context=self.config['study']['context']
        if context=='corrupted':
            token=f'{self.config["study"]["seed"]}:{self.epoch}:{self.paths[shard].name}:{t}'
            seed=int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],'little')
            units=corrupt_context(units,seed,self.config['study']['corruption_probability'])
        elif context!='scripted':
            raise ValueError('policy contexts are evaluated sequentially; never substitute expert goals')
        regions=np.zeros((16,24,24),bool)
        for uid in np.flatnonzero(d[prefix+'label_valid'][t] | (d['semantic_valid'][t] if not prefix else False)):
            sid=d[prefix+'segment_id'][t,uid]
            regions[uid]=d[prefix+'goal_regions'][sid].astype(bool)
        gv=d[prefix+'label_valid'][t].copy()
        r,rv,compatible=replan_targets(d['mode'][t],regions,d['semantic_valid'][t],
            d['boundary_replan'][t],d['bridge_continue'][t],units,d['label_valid'][t])
        sampled=np.zeros(16,bool);sampled[u]=True
        return {'history':torch.from_numpy(history_at(d['features'],d['steps'],d['match_steps'],t)),
                'positions':torch.from_numpy(d['positions'][t].astype(np.int64)), 'units':torch.from_numpy(units),
                'alive':torch.from_numpy(d['active'][t]),'mode':torch.from_numpy(d[prefix+'mode'][t].astype(np.int64)),
                'regions':torch.from_numpy(regions),'valid':torch.from_numpy(gv & sampled),
                'confidence':torch.from_numpy(d[prefix+'label_confidence'][t].astype(np.float32)),
                'replan':torch.from_numpy(r.astype(np.float32)),'replan_valid':torch.from_numpy(rv & sampled),
                'sampled':torch.from_numpy(sampled),'compatible':torch.from_numpy(compatible),
                'shard':torch.tensor(shard),'step':torch.tensor(int(d['steps'][t])),
                'match_step':torch.tensor(int(d['match_steps'][t])),
                'segment_id':torch.from_numpy(d['segment_id'][t].astype(np.int64))}


class UnitEpochSampler(Sampler):
    def __init__(self,dataset,seed,budget,balanced=False):
        self.dataset,self.seed,self.budget,self.balanced=dataset,seed,budget,balanced
        self.epoch=0

    def __len__(self):return self.budget

    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch)
        p=None
        if self.balanced:
            p=1/np.maximum(self.dataset.mode_counts[self.dataset.modes],1);p=p/p.sum()
        selected=rng.choice(len(self.dataset),self.budget,replace=self.balanced or self.budget>len(self.dataset),p=p)
        # Group disk access for BOTH samplers. Per-unit probabilities remain the
        # sampled epoch histogram; order within each shard is randomized.
        shards=np.unique(self.dataset.rows[selected,0]);rng.shuffle(shards)
        for shard in shards:
            indices=selected[self.dataset.rows[selected,0]==shard];rng.shuffle(indices)
            yield from map(int,indices)


def class_weights(counts,cap):
    counts=np.asarray(counts,float)
    return np.minimum(counts.sum()/(5*np.maximum(counts,1)),cap).astype(np.float32)


def paths_for_split(data_dir, split):
    manifest=json.loads((Path(data_dir)/'manifest.json').read_text())
    return [Path(data_dir)/name for name,row in manifest['files'].items() if row['split']==split]
