"""Reconstruct current features from an existing explicit macro teacher's public trace."""
import argparse
import gzip
import json
import pickle
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from torch.utils.data import Dataset

from agents.mappo.shared.features import CausalObservationBuilder
from agents.mappo.shared.manager import TeamMemory
from agents.mappo.shared.physics import PhysicsState
from agents.mappo.shared.policy import pack_policy
from training.contracts import config, json_write
from .replays import sha


def cases(root):
    result=[]
    for directory in sorted(root.glob("case_*")):
        if directory.is_symlink():raise RuntimeError("unexpected case symlink")
        pointer=json.loads((directory/"committed.json").read_text())
        request=json.loads((directory/f'{pointer["attempt"]}.request.json').read_text())["case"]
        if request["model"]!="S2" or request["strategy"]!="FULL":continue
        attempt=directory/pointer["attempt"]
        if sha(attempt/"complete.json")!=pointer["complete_sha256"]:
            raise RuntimeError("historical complete receipt drift")
        receipt=json.loads((attempt/"complete.json").read_text())
        for n in ["trace.pkl.gz","trace.pkl.hashes.jsonl","summary.json"]:
            if sha(attempt/n)!=receipt["files"][n]:raise RuntimeError(f"historical source changed {attempt/n}")
        result.append({"case":request,"attempt":str(attempt),"sha256":receipt["files"]["trace.pkl.gz"]})
    if len(result)!=48 or len({(x["case"]["root"],x["case"]["side"]) for x in result})!=48:
        raise RuntimeError("expected 48 complete S2 FULL source games")
    return sorted(result,key=lambda x:(x["case"]["map_rank"],x["case"]["side"]))


def build_case(source,path,limit=None):
    c=source["case"];side=c["side"]
    states=[];commands=[];steps=[];outcomes=[];rawobs=[]
    passive=None;pending=None;seen_steps=[];decisions=0
    with gzip.open(Path(source["attempt"])/"trace.pkl.gz","rb") as f:
        while True:
            try:event=pickle.load(f)
            except EOFError:break
            kind=event["kind"]
            if kind=="public_IPC":
                obs=event["response"]["obs"]
                if passive is None:
                    env_cfg=event["response"]["env_cfg"]
                    builder=CausalObservationBuilder(side,env_cfg)
                    passive=SimpleNamespace(side=side,env_cfg=env_cfg,obs=obs,steps=0,
                                            builder=builder,physics=PhysicsState(side,env_cfg),
                                            manager=TeamMemory(config("a")["review"],env_cfg))
                passive.obs=obs;passive.steps=int(obs["steps"])
                passive.frame=passive.builder.observe(obs)
                passive.physics.observe(obs,passive.frame,passive.builder.reconstructor)
                seen_steps.append(passive.steps)
            elif kind=="decision":
                if event["step"]!=passive.steps or pending is not None:
                    raise RuntimeError("decision clock or pending-label mismatch")
                p=event["packet"];action=event["action"]
                previous=np.asarray(p["previous_intent"])[0]
                manager=passive.manager
                manager.modes=previous[:,0].astype(np.int8).copy()
                manager.goals=previous[:,1:3].astype(np.int16).copy()
                manager.ages=previous[:,3].astype(np.int16).copy()
                manager.valid=previous[:,4].astype(bool).copy()
                manager.blocked=previous[:,5].astype(np.int16).copy()
                manager.reached=previous[:,6].astype(bool).copy()
                manager.review_mask=np.asarray(p["review"])[0].astype(bool).copy()
                state=pack_policy(passive,True)
                old_support=np.asarray(p["legal_goals"])[0]
                if side:old_support=old_support[::-1]
                if not np.array_equal(state["legal"],old_support):
                    raise RuntimeError("current and logged physical goal support differ")
                if not np.array_equal(state["alive"],np.asarray(p["alive"])[0]):
                    raise RuntimeError("logged and reconstructed live UID support differ")
                cmd=np.column_stack([np.asarray(action[k])[0] for k in ("r","mode","goal")]).astype(np.int64)
                if side:cmd[:,2]=575-cmd[:,2]
                pending=(state,cmd,event["record"]["rows"])
            elif kind=="actual_primitive" and pending is not None:
                state,cmd,rows=pending
                if event["step"]!=passive.steps:raise RuntimeError("Manager outcome clock mismatch")
                for uid in np.flatnonzero(state["review"] & cmd[:,0].astype(bool)):
                    actual_goal=int(np.asarray(event["goals"])[uid]@[24,1])
                    teacher_goal=int(cmd[uid,2] if side==0 else 575-cmd[uid,2])
                    if not event["valid"][uid] or actual_goal!=teacher_goal or int(event["modes"][uid])!=cmd[uid,1]:
                        raise RuntimeError("teacher label was not accepted by actual Manager")
                if state["review"].any():
                    states.append(state);commands.append(cmd);steps.append(passive.steps)
                    rawobs.append(passive.obs)
                    outcomes.append([{"uid":u,"accepted_v2":row.get("accepted",False),
                                      "outcome":row.get("outcome"),"reason":row.get("reason")}
                                     for u,row in enumerate(rows)])
                    decisions+=1
                    if limit and decisions>=limit:
                        break
                pending=None
    if not limit and seen_steps!=list(range(506)):
        raise RuntimeError("missing or duplicate public observations")
    if not states:raise RuntimeError("no supervised review")
    arrays={k:np.stack([s[k] for s in states]) for k in states[0]}
    arrays["commands"]=np.stack(commands);arrays["timestep"]=np.asarray(steps,np.int16)
    np.savez_compressed(path,**arrays)
    meta={"source":source,"rows":len(states),"split":"train" if c["map_rank"]<16 else
           "calibration" if c["map_rank"]<20 else "exploration",
          "teacher":"G06_FULL_S2_B0_RMode_V2_adapted_Goal",
          "top_ranked_primitive_teacher":False,"future_inputs":False,
          "outcomes":outcomes,"public_observations":rawobs}
    json_write(path.with_suffix(".json"),meta)
    return {"file":path.name,"sha256":sha(path),"meta_sha256":sha(path.with_suffix(".json")),
            "case_id":c["index"],"map":c["root"],"side":side,"split":meta["split"],"rows":len(states)}


def build(root,output,diagnostic=False):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    sources=cases(root)
    shards=[]
    selected=sources[:1] if diagnostic else sources
    for i,source in enumerate(selected):
        shards.append(build_case(source,output/f"case_{source['case']['index']:03d}.npz",4 if diagnostic else None))
        print("BUILD",i+1,len(selected),shards[-1]["rows"],flush=True)
    manifest={"schema":"current_actor_macro_teacher_v1","sources":sources,"shards":shards,
              "training_ready":not diagnostic,"teacher":"G06_FULL_S2_B0_RMode_V2_adapted_Goal",
              "top_expert_goal_supervision_ready":False,
              "scope":"Explicit historical macro teacher only. Top-agent primitive replays are audited separately.",
              "split":"original map_rank 0–15 train, 16–19 calibration, 20–23 exploration",
              "features":"current frozen causal physics; previous intent from pre-decision record",
              "label":"actual Manager-accepted R/Mode/Goal; fallback and Continue retained",
              "old400_seeds":"historically missing; no reuse of old Waypoint labels or weights"}
    json_write(output/"manifest.json",manifest)
    return manifest


class BCDataset(Dataset):
    def __init__(self,root,split_name):
        self.root=Path(root);self.manifest=json.loads((self.root/"manifest.json").read_text())
        self.shards=[s for s in self.manifest["shards"] if s["split"]==split_name]
        self.rows=[(i,j) for i,s in enumerate(self.shards) for j in range(s["rows"])]
        self.cache=OrderedDict()

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        shard,offset=self.rows[index];spec=self.shards[shard]
        if shard not in self.cache:
            p=self.root/spec["file"]
            if sha(p)!=spec["sha256"] or sha(p.with_suffix(".json"))!=spec["meta_sha256"]:
                raise RuntimeError("BC shard content changed")
            with np.load(p,allow_pickle=False) as arrays:self.cache[shard]={k:arrays[k] for k in arrays.files}
            while len(self.cache)>1:self.cache.popitem(last=False)
        self.cache.move_to_end(shard)
        arrays=self.cache[shard]
        state={k:v[offset].copy() for k,v in arrays.items() if k not in ("commands","timestep")}
        return state,arrays["commands"][offset].copy()


def verify(root):
    manifest=json.loads((root/"manifest.json").read_text())
    seen=set();maps={}
    counts={}
    for split_name in ("train","calibration","exploration"):
        data=BCDataset(root,split_name)
        counts[split_name]=len(data)
        for i in range(len(data)):
            state,commands=data[i]
            if state["units"].shape!=(16,22) or state["grid"].shape!=(19,24,24):
                raise RuntimeError("BC shape contract")
            claims=np.where(state["valid"],state["previous"],-1)
            for u in range(16):
                if not state["review"][u]:continue
                claims[u]=-1
                if commands[u,0]:
                    m,g=commands[u,1:]
                    if not state["legal"][g] or (m==0 and state["false_score"][g]) or g in claims:
                        raise RuntimeError("BC label outside deployment support")
                    claims[u]=g
                else:claims[u]=state["previous"][u]
        for shard in data.shards:
            key=shard["map"],shard["side"]
            if key in seen:raise RuntimeError("duplicate BC game")
            seen.add(key)
            if shard["map"] in maps and maps[shard["map"]]!=split_name:
                raise RuntimeError("map leakage")
            maps[shard["map"]]=split_name
    if manifest["training_ready"] and not all(counts.values()):
        raise RuntimeError("empty BC calibration or exploration partition")
    return {"status":"PASS","rows":counts,"maps":len(maps),
            "teacher":manifest["teacher"],"top_expert_goal_supervision_ready":False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--diagnostic",action="store_true")
    p.add_argument("--verify",action="store_true")
    a=p.parse_args()
    if a.verify:print(json.dumps(verify(a.output),indent=2))
    else:
        if not a.source:p.error("--source is required")
        build(a.source,a.output,a.diagnostic)


if __name__=="__main__":main()
