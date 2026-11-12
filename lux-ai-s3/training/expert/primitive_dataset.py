"""Causal expert action auxiliary data; no inferred macro labels."""
import argparse
import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from torch.utils.data import Dataset

from agents.mappo.shared.features import CausalObservationBuilder
from agents.mappo.shared.manager import TeamMemory
from agents.mappo.shared.physics import PhysicsState
from training.contracts import config,json_write
from .replays import observation,actions,sha


def build(manifest_path,teacher,output,diagnostic=False,submission_id=None):
    manifest=json.loads(manifest_path.read_text())
    sources=[r for r in manifest["sources"] if teacher in r["teachers"]]
    if not sources:raise ValueError("requested primitive teacher absent")
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    shards=[]
    for source in sources[:1] if diagnostic else sources:
        path=Path(source["path"])
        if sha(path)!=source["raw_sha256"]:raise RuntimeError("expert replay changed")
        replay=json.loads(path.read_text());cfg=source["environment"]
        for side,name in enumerate(source["teachers"]):
            if name!=teacher:continue
            version=(source.get("versions") or {}).get(str(side))
            if version is None:version=(source.get("versions") or {}).get(side)
            if submission_id is not None and (version is None or version["submission_id"]!=submission_id):
                continue
            builder=CausalObservationBuilder(side,cfg)
            memory=TeamMemory(config("a")["review"],cfg)
            physics=PhysicsState(side,cfg)
            state=SimpleNamespace(side=side,manager=memory,env_cfg=cfg)
            rows=[];labels=[];times=[]
            for t in range(505):
                obs=observation(replay,t,side)
                frame=builder.observe(obs)
                physics.observe(obs,frame,builder.reconstructor)
                state.obs=obs;state.frame=frame;state.steps=t
                if t%4 or not frame["alive"].any():continue
                public=physics.critic_view(state)
                # Previous teacher intent is unknown. Only known local physical values are retained.
                physical_units=public["units"][:,:3].copy()
                code=actions(replay,t,side)[:,0]
                if side:
                    code=np.asarray([0,3,4,1,2,5])[code]
                rows.append({"grid":public["grid"],"position_energy":physical_units,
                             "alive":public["alive"],"team":public["team"]})
                labels.append(code);times.append(t)
                if diagnostic and len(rows)>=4:break
            key=f"{source['replay_id']}_p{side}"
            file=output/f"{key}.npz"
            np.savez_compressed(file,**{k:np.stack([r[k] for r in rows]) for k in rows[0]},
                                expert_action=np.stack(labels),timestep=np.asarray(times,np.int16))
            shards.append({"file":file.name,"sha256":sha(file),"rows":len(rows),"root":source["root"],
                           "side":side,"replay":source["replay_id"],"split":source["split"],
                           "source_sha256":source["raw_sha256"],"submission_id":version["submission_id"] if version else None})
            print("PRIMITIVE",len(shards),len(sources),key,len(rows),flush=True)
    result={"schema":"expert_primitive_aux_v1","teacher":teacher,"submission_id":submission_id,"training_ready":not diagnostic and submission_id is not None,
            "stride":4,"shards":shards,"input":"public encoder maps, own position/energy, public team time only",
            "macro_R_Mode_Goal":"unavailable","waypoint_labels":False,
            "loss":"six-way move/wait/SAP cross entropy; no SAP target supervision"}
    json_write(output/"manifest.json",result)
    return result


class PrimitiveDataset(Dataset):
    def __init__(self,root,split):
        self.root=Path(root)
        self.manifest=json.loads((self.root/"manifest.json").read_text())
        self.shards=[s for s in self.manifest["shards"] if s["split"]==split]
        self.rows=[(i,j) for i,s in enumerate(self.shards) for j in range(s["rows"])]
        self.cache=OrderedDict()

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        shard,row=self.rows[index];spec=self.shards[shard]
        if shard not in self.cache:
            path=self.root/spec["file"]
            if sha(path)!=spec["sha256"]:raise RuntimeError("primitive auxiliary shard changed")
            with np.load(path,allow_pickle=False) as z:self.cache[shard]={k:z[k] for k in z.files}
            while len(self.cache)>1:self.cache.popitem(last=False)
        self.cache.move_to_end(shard)
        arrays=self.cache[shard]
        state={k:arrays[k][row].copy() for k in ("grid","position_energy","alive","team")}
        return state,arrays["expert_action"][row].copy()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--teacher",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--diagnostic",action="store_true")
    p.add_argument("--submission-id",type=int)
    a=p.parse_args();build(a.manifest,a.teacher,a.output,a.diagnostic,a.submission_id)


if __name__=="__main__":main()
