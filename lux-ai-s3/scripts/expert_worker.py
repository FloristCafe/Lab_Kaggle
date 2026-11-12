"""Seeded CPU inference; external projects remain outside the main repository."""
import argparse
import hashlib
import contextlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["JAX_PLATFORMS"]="cpu"

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"vendor/lux-design-s3/src"))

import numpy as np
import torch


def from_json(value):
    if isinstance(value,list):return np.array(value)
    if isinstance(value,dict):return {k:from_json(v) for k,v in value.items()}
    return value


def create(spec,side,config):
    root=Path(spec["root"])
    for filename,expected in spec["sha256"].items():
        with Path(filename).open("rb") as stream:
            if hashlib.file_digest(stream,"sha256").hexdigest()!=expected:
                raise RuntimeError("external source or weight changed before game")
    sys.path.insert(0,str(root))
    player=f"player_{side}"
    if spec["kind"]=="adg4b":
        from agent import Agent
        return Agent(player,config,str(root/"imitation_learning/weights"))
    if spec["kind"]=="flat":
        import agent
        native=agent.Agent(player,config)
        # Upstream catches load errors and can fall back to a weak policy. Admission must fail instead.
        if not hasattr(native,"actor_model") or not hasattr(native,"env"):
            raise RuntimeError("Flat Neurons model failed to load")
        return native
    if spec["kind"]=="gregor":
        original=torch.jit.load
        def relocate(path,*args,**kwargs):
            name=Path(path).name
            if name not in spec["weight_names"]:raise RuntimeError("unregistered model load")
            return original(str(root/name),*args,**kwargs)
        torch.jit.load=relocate
        try:
            module_spec=importlib.util.spec_from_file_location("gregor_main",root/"main.py")
            module=importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
        finally:torch.jit.load=original
        class Wrapped:
            def act(self,step,obs,remainingOverageTime=60):
                from argparse import Namespace
                return module.agent_fn(Namespace(step=step,obs=obs,player=player,
                                                 remainingOverageTime=remainingOverageTime),
                                       {"env_cfg":config})["action"]
        return Wrapped()
    raise ValueError("unknown native opponent kind")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--registry",type=Path,required=True)
    p.add_argument("--agent",required=True)
    p.add_argument("--side",type=int,choices=(0,1),required=True)
    p.add_argument("--seed",type=int,required=True)
    a=p.parse_args()
    spec=json.loads(a.registry.read_text())["agents"][a.agent]
    torch.set_num_threads(2)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    native=None
    for line in sys.stdin:
        request=json.loads(line)
        try:
            with contextlib.redirect_stdout(sys.stderr),torch.no_grad():
                if request["op"]=="initialize":
                    native=create(spec,a.side,request["config"])
                    result={"initialized":True}
                elif request["op"]=="act":
                    action=native.act(request["step"],from_json(request["obs"]),60)
                    result={"action":np.asarray(action,dtype=np.int32).tolist()}
                else:raise ValueError("unknown request")
            print(json.dumps(result),flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"error":f"{type(exc).__name__}: {exc}"}),flush=True)
            raise


if __name__=="__main__":main()
