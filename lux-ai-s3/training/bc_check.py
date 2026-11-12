"""Read-only matching-architecture BC checkpoint verification."""
import argparse
import json
from pathlib import Path

import torch

from training.contracts import committed,read,setup,sha
from training.core import models
from training.expert.initialize import load_bc


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices="abc",required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--allow-smoke",action="store_true")
    a=p.parse_args()
    setup("cpu",0)
    actor,_=models(a.model,"cpu")
    meta=load_bc(actor,a.model,a.checkpoint,a.allow_smoke)
    selection=read(a.checkpoint) if a.checkpoint.suffix==".json" else None
    path=Path(selection["checkpoint"]) if selection else a.checkpoint
    state=torch.load(path,map_location="cpu",weights_only=False)
    latest=committed(path.parent.parent)
    history=torch.load(latest/"checkpoint.pt",map_location="cpu",weights_only=False)["history"]
    best=min(history,key=lambda m:(m["calibration_nll"],m["epoch"]))
    if selection and best["epoch"]!=selection["selected_epoch"]:
        raise RuntimeError("calibration selection differs from stored history")
    print(json.dumps({"status":"PASS","selected_epoch":state["epoch"],"parameters":sum(x.numel() for x in actor.parameters()),
                      "sha256":sha(path),**meta},indent=2))


if __name__=="__main__":main()
