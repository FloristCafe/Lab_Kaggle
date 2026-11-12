"""Read-only independent verification of the expert preparation handoff."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.contracts import read,sha,source_lock
from training.opponents.pool import registry


def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b,strict=True))
    return a==b


def verify(artifacts,engineering):
    os.environ["LUX_OPPONENT_REGISTRY"]=str(artifacts/"admitted_opponents.json")
    spec=registry()
    if read(ROOT/"release_source_lock.json")!=source_lock():raise RuntimeError("release changed")
    frozen=read(artifacts/"frozen_agent_hashes.json")
    for file,digest in frozen.items():
        path=Path(file) if Path(file).is_absolute() else ROOT/file
        if sha(path)!=digest:raise RuntimeError("frozen Agent changed: "+file)
    games=0
    for name,agent in spec["agents"].items():
        if not agent["enabled"]:continue
        sides=set()
        for item in agent["full_game_acceptance"]:
            path=Path(item["summary"])
            if sha(path)!=item["summary_sha256"]:raise RuntimeError("admission summary changed")
            summary=read(path);trajectory=read(path.parent/"trajectory.json")
            if summary["agent"]!=name or not summary["complete"] or summary["steps"]!=505:
                raise RuntimeError("incomplete expert admission")
            if [r["step"] for r in trajectory]!=list(range(1,506)):raise RuntimeError("noncontiguous expert game")
            if [r["step"] for r in trajectory if r["terminal"]]!=[505]:raise RuntimeError("expert terminal mismatch")
            if sum(trajectory[-1]["wins"])!=5:raise RuntimeError("official counters invalid")
            sides.add(summary["side"]);games+=1
        if sides!={0,1}:raise RuntimeError("missing expert side")
    results={}
    for model in "abc":
        left=engineering/f"continuous_{model}"
        right=engineering/f"resumed_{model}"
        lp=left/"batch_000001/checkpoint.pt";rp=right/"batch_000001/checkpoint.pt"
        for checkpoint in (lp,rp):
            receipt=read(checkpoint.parent/"commit.json")
            if any(sha(checkpoint.parent/name)!=value for name,value in receipt.items()):
                raise RuntimeError("resume evidence receipt changed")
        a=torch.load(lp,map_location="cpu",weights_only=False)
        b=torch.load(rp,map_location="cpu",weights_only=False)
        keys=("actor","critic","actor_opt","critic_opt","rng","opponent_pool","games","steps","updates")
        if not all(same(a[k],b[k]) for k in keys):raise RuntimeError("recovery mismatch "+model)
        for i in range(2):
            file=f"batch_{i:06d}/game_{i:06d}.json"
            if sha(left/file)!=sha(right/file):raise RuntimeError("resumed trajectory differs")
        selection=read(engineering/f"bc_{model}/selected.json")
        if sha(selection["checkpoint"])!=selection["sha256"]:raise RuntimeError("BC smoke selected hash differs")
        bc=torch.load(selection["checkpoint"],map_location="cpu",weights_only=False)
        if not bc["identity"]["smoke"] or bc["updates"]!=1:raise RuntimeError("unexpected BC smoke budget")
        results[model]={"recovery":"all tensors/AdamW/RNG/pool/trajectories equal",
                        "bc_sha256":selection["sha256"],"BC_updates":bc["updates"],
                        "RL_completed_steps":a["steps"]}
    report=read(engineering/"verification.json")
    if report["status"]!="PASS" or report["formal_BC_epochs"] or report["formal_RL_games"]:
        raise RuntimeError("engineering report inconsistent")
    raw=read(artifacts/"raw_teacher_versions02.json")
    rebuilt=read(artifacts/"raw_teacher_versions_rebuilt01.json")
    if raw["sources"]!=rebuilt["sources"]:raise RuntimeError("independently rebuilt source identities differ")
    for path,digest in raw["index_sources_sha256"].items():
        if sha(path)!=digest:raise RuntimeError("source metadata changed")
    data=read(artifacts/"independent_data_verification01.json")
    for directory,key in (("macro_dataset01","macro_manifest_sha256"),("primitive_flat02","primitive_manifest_sha256")):
        if sha(artifacts/directory/"manifest.json")!=data[key]:raise RuntimeError("dataset manifest changed")
    print(json.dumps({"status":"PASS","native_opponent_complete_games":games,
                      "native_engineering_steps":505*games,"models":results,
                      "frozen_agent_files":len(frozen),"source_file_count":len(source_lock()),
                      "data_verification_sha256":sha(artifacts/"independent_data_verification01.json"),
                      "formal_training_started":False,"environment_starts_in_this_verifier":0},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts",type=Path,required=True)
    p.add_argument("--engineering",type=Path,required=True)
    a=p.parse_args();verify(a.artifacts,a.engineering)
