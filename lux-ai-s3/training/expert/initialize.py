"""BC-to-RL transition. BC optimizer is not reused as an RL optimizer."""
from pathlib import Path

import torch

from training.contracts import read,sha,source_lock


def load_bc(actor,model,path,smoke=False):
    path=Path(path)
    if path.name=="selected.json":
        selected=read(path);checkpoint=Path(selected["checkpoint"])
        if sha(checkpoint)!=selected["sha256"]:raise RuntimeError("selected BC checkpoint hash mismatch")
    else:checkpoint=path
    receipt=read(checkpoint.parent/"commit.json")
    if receipt.get("checkpoint.pt")!=sha(checkpoint):raise RuntimeError("BC checkpoint commit mismatch")
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if state.get("schema")!="macro_bc_v1" or state["identity"]["model"]!=model:
        raise RuntimeError("BC architecture/schema mismatch")
    if state["identity"]["smoke"] and not smoke:raise RuntimeError("engineering BC cannot initialize formal RL")
    current=source_lock()
    if any(state["identity"]["sources"].get(k)!=h for k,h in current.items() if k.startswith("agents/")):
        raise RuntimeError("BC actor feature or deployment implementation changed")
    if not smoke:
        selection=read(checkpoint.parent.parent/"selected.json")
        if selection["sha256"]!=sha(checkpoint) or selection["criterion"]!="calibration_nll_then_earliest_epoch":
            raise RuntimeError("formal RL requires the calibration-selected BC checkpoint")
    actor.load_state_dict(state["actor"],strict=True)
    if not all(torch.equal(value.cpu(),state["actor"][name].cpu()) for name,value in actor.state_dict().items()):
        raise RuntimeError("BC parameter load was not exact")
    return {"checkpoint_sha256":sha(checkpoint),"teacher":state["identity"]["teacher"],
            "dataset_sha256":state["identity"]["dataset_sha256"],"model":model,
            "engineering":state["identity"]["smoke"],"all_tensors_equal":True}
