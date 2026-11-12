"""Small engineering-only BC -> RL and exact recovery checks. Never formal training."""
import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.contracts import committed,json_write,restore_rng,rng_state,setup,sha
from training.core import grad_step,models
from training.expert.macro_dataset import BCDataset
from training.expert.primitive_dataset import PrimitiveDataset
from training.pretrain import loss,primitive_loss


def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b,strict=True))
    return a==b


def run(artifacts,output):
    output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,LUX_ARTIFACTS=str(artifacts),
             LUX_OPPONENT_REGISTRY=str(artifacts/"admitted_opponents.json"),
             PYTHONHASHSEED="0",CUBLAS_WORKSPACE_CONFIG=":4096:8",
             OMP_NUM_THREADS="2",MKL_NUM_THREADS="2")
    commands=[]
    def execute(name,args):
        cmd=[sys.executable,"-B",*args]
        commands.append({"name":name,"argv":cmd})
        with (output/f"{name}.log").open("x") as f:
            completed=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=False)
        if completed.returncode:
            json_write(output/"failure.json",{"name":name,"returncode":completed.returncode,"commands":commands})
            raise RuntimeError(f"engineering command failed: {name}")
        print(name,"PASS",flush=True)
    execute("unit",["-m","pytest","-q","tests"])
    execute("static",["-m","ruff","check","--isolated","--select","E9,F","agents","training","scripts","tools","tests"])
    results={}
    for model in "abc":
        bc=output/f"bc_{model}"
        base=["-m","training.pretrain","--model",model,
              "--dataset",str(artifacts/"macro_diagnostic01"),
              "--teacher","G06_FULL_S2_B0_RMode_V2_adapted_Goal",
              "--primitive-dataset",str(artifacts/"primitive_flat02"),
              "--run-dir",str(bc),"--smoke"]
        execute(f"bc_{model}",base)
        execute(f"bc_resume_{model}",base+["--resume"])
        execute(f"bc_check_{model}",["-m","training.bc_check","--model",model,
                "--checkpoint",str(bc/"selected.json"),"--allow-smoke"])
        common=["-m","training.train","--model",model,"--seed","1313969260",
                "--smoke","--steps","24","--bc-checkpoint",str(bc/"selected.json")]
        continuous=output/f"continuous_{model}"
        resumed=output/f"resumed_{model}"
        execute(f"continuous_{model}",common+["--games","2","--run-dir",str(continuous)])
        execute(f"split_{model}",common+["--games","1","--run-dir",str(resumed)])
        execute(f"resume_{model}",common+["--games","2","--run-dir",str(resumed),"--resume","latest"])
        x=torch.load(committed(continuous)/"checkpoint.pt",map_location="cpu",weights_only=False)
        y=torch.load(committed(resumed)/"checkpoint.pt",map_location="cpu",weights_only=False)
        fields=("actor","critic","actor_opt","critic_opt","rng","opponent_pool","games","steps","updates")
        checks={key:equal(x[key],y[key]) for key in fields}
        for batch in range(2):
            file=f"batch_{batch:06d}/game_{batch:06d}.json"
            checks[f"trajectory_{batch}"]=sha(continuous/file)==sha(resumed/file)
        for d in (continuous,resumed):
            for f in d.glob("attempt_*/bc_load.json"):
                checks[str(f.relative_to(d))]=json.loads(f.read_text())["rl_optimizers_empty"]
        if not all(checks.values()):
            json_write(output/f"failure_{model}.json",checks)
            raise RuntimeError("resumed RL differs from uninterrupted RL")
        results[model]={"checks":checks,"games":x["games"],"steps":x["steps"],
                        "opponent_ids":x["opponent_pool"]["ids"]}
        # BC resume uses the same model/AdamW/RNG; independently replay one further offline update.
        state=torch.load(committed(bc)/"checkpoint.pt",map_location="cpu",weights_only=False)
        data=BCDataset(artifacts/"macro_diagnostic01","train")
        primitive=PrimitiveDataset(artifacts/"primitive_flat02","train")
        def next_update(payload,model=model,data=data,primitive=primitive):
            setup("cuda",999)
            actor,_=models(model,"cuda")
            aux=torch.nn.Linear(37,6).to("cuda")
            actor.load_state_dict(payload["actor"]);aux.load_state_dict(payload["auxiliary"])
            params=list(actor.parameters())+list(aux.parameters())
            opt=torch.optim.AdamW(params,lr=.001,weight_decay=0)
            opt.load_state_dict(payload["optimizer"])
            restore_rng(payload["rng"])
            value,_=loss(actor,data,[0,1],"cuda")
            objective=value+.1*primitive_loss(actor,aux,primitive,[0,1],"cuda")
            grad_step(objective,opt,1.,params)
            return {"actor":copy.deepcopy(actor.state_dict()),"auxiliary":copy.deepcopy(aux.state_dict()),
                    "optimizer":copy.deepcopy(opt.state_dict()),"rng":rng_state()}
        left=next_update(state)
        disk=torch.load(committed(bc)/"checkpoint.pt",map_location="cpu",weights_only=False)
        right=next_update(disk)
        results[model]["bc_optimizer_rng_next_update_equal"]=equal(left,right)
        if not results[model]["bc_optimizer_rng_next_update_equal"]:
            raise RuntimeError("BC optimizer/RNG continuation changed")
    json_write(output/"commands.json",commands)
    report={"status":"PASS","engineering_only":True,"formal_BC_epochs":0,"formal_RL_games":0,
            "engineering_RL_primitive_steps":288,"models":results}
    json_write(output/"verification.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();run(a.artifacts,a.output)
