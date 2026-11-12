"""Compact dataset_v2_goal, causal histories, segment region targets and queries."""
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from agents.v3.dataset import CAUSAL_CURRENT_CHANNELS

MASKS = (0,1,3,4,5,6,7,8,10,12,17,18,19,20)
VALUES = tuple(i for i in range(21) if i not in MASKS)


def history_at(features,steps,match_steps,t):
    result = np.zeros((3,18,24,24),np.float32)
    for lag in range(3):
        past = t-lag
        if past<0 or (lag and (not (np.diff(steps[past:t+1])==1).all()
                              or not (np.diff(match_steps[past:t+1])==1).all())):
            break
        result[lag] = features[past,list(CAUSAL_CURRENT_CHANNELS)]
    return result


def unpack(npz):
    if int(npz['schema_version'])!=4:
        raise ValueError('require dataset_v2_goal schema4')
    features = np.zeros((len(npz['steps']),21,24,24),np.float32)
    features[:,MASKS] = npz['feature_masks']
    features[:,VALUES] = npz['feature_values']
    result = {k:npz[k] for k in npz.files if k not in ('feature_masks','feature_values')}
    result['features'] = features
    return result


class GoalDataset(Dataset):
    def __init__(self, paths, frame_stride=1, cache_size=1):
        self.paths = sorted(map(Path,paths))
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.indices = []
        self.valid_units = 0
        self.frame_ids = []
        sizes = []
        for shard,path in enumerate(self.paths):
            with np.load(path,allow_pickle=False) as d:
                valid = d['label_valid']
                ids = np.flatnonzero(valid.any(1))[::frame_stride]
                self.frame_ids.append(ids)
                self.indices.extend((shard,int(t)) for t in ids)
                self.valid_units += int(valid[ids].sum())
                sizes.append(len(ids))
        self.offsets = np.r_[0,np.cumsum(sizes)]

    def __len__(self):
        return len(self.indices)

    def _load(self,shard):
        if shard not in self.cache:
            with np.load(self.paths[shard],allow_pickle=False) as npz:
                self.cache[shard] = unpack(npz)
            while len(self.cache)>self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(shard)
        return self.cache[shard]

    def __getitem__(self,index):
        shard,t = self.indices[index]
        d = self._load(shard)
        regions = np.zeros((16,24,24),np.uint8)
        for uid in np.flatnonzero(d['label_valid'][t]):
            regions[uid] = d['goal_regions'][d['segment_id'][t,uid]]
        return dict(history=torch.from_numpy(history_at(d['features'],d['steps'],d['match_steps'],t)),
                    positions=torch.from_numpy(d['positions'][t].astype(np.int64)),
                    units=torch.from_numpy(d['unit_vectors'][t].astype(np.float32)),
                    alive=torch.from_numpy(d['active'][t]),
                    mode=torch.from_numpy(d['mode'][t].astype(np.int64)),
                    regions=torch.from_numpy(regions.astype(bool)),
                    valid=torch.from_numpy(d['label_valid'][t]),
                    confidence=torch.from_numpy(d['label_confidence'][t].astype(np.float32)),
                    segment_id=torch.from_numpy(d['segment_id'][t].astype(np.int64)),
                    shard=torch.tensor(shard),step=torch.tensor(int(d['steps'][t])))


def collate_goal(rows):
    return {key:torch.stack([row[key] for row in rows]) for key in rows[0]}
