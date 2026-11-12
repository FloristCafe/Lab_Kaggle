"""Offline macro BC for A/B/C. Explicit teacher identity is mandatory."""
import argparse
import fcntl
import os
# MLflow telemetry uses global random in a background thread. Keep tracking local and synchronous.
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
os.environ["MLFLOW_ENABLE_ASYNC_LOGGING"] = "false"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import time
from pathlib import Path

import mlflow
import numpy as np
import torch

from agents.mappo.shared.policy import decide
from training.contracts import ARTIFACTS,commit,committed,json_write,read,restore_rng,rng_state,setup,sha,source_lock
from training.core import batch,grad_step,models
from training.expert.macro_dataset import BCDataset,verify
from training.expert.primitive_dataset import PrimitiveDataset


def loss(actor,data,indices,device):
    samples=[data[i] for i in indices]
    state=batch([s for s,_ in samples],device)
    commands=torch.from_numpy(np.stack([c for _,c in samples])).to(device)
    _,logp,active,_,_=decide(actor,state,commands)
    per_head=[]
    for k in range(3):
        mask=active[...,k]
        if mask.any():per_head.append(-logp[...,k][mask].mean())
    if not per_head:raise RuntimeError("BC batch has no supervised decisions")
    return torch.stack(per_head).sum(),int(active[...,2].sum())


@torch.no_grad()
def evaluate(actor,data,device,batch_size):
    actor.eval()
    total=weight=0
    for first in range(0,len(data),batch_size):
        ids=list(range(first,min(first+batch_size,len(data))))
        value,_=loss(actor,data,ids,device)
        total+=float(value)*len(ids);weight+=len(ids)
    actor.train()
    if weight==0:raise RuntimeError("empty BC evaluation")
    return total/weight


def primitive_loss(actor, auxiliary, data, indices, device):
    rows=[data[i] for i in indices]
    state=batch([s for s,_ in rows],device)
    encoded=actor.encoder(state["grid"])
    physical=state["position_energy"].float()
    xy=(physical[...,:2]*23).round().long()
    cells=xy[...,0]*24+xy[...,1]
    local=encoded.gather(1,cells[...,None].expand(-1,-1,encoded.shape[-1]))
    clock=state["team"][:,None,4:6].float().expand(-1,16,-1)
    prediction=auxiliary(torch.cat((local,physical,clock),-1))
    # Stored replay classes may be int32 or int64; CrossEntropy requires int64 class indices.
    target=torch.from_numpy(np.stack([a for _,a in rows])).to(device=device,dtype=torch.long)
    active=state["alive"]
    return torch.nn.functional.cross_entropy(prediction[active],target[active])


@torch.no_grad()
def evaluate_primitive(actor,auxiliary,data,device,batch_size):
    if not len(data):raise RuntimeError("empty expert calibration data")
    total=count=0
    for start in range(0,len(data),batch_size):
        ids=list(range(start,min(start+batch_size,len(data))))
        total+=float(primitive_loss(actor,auxiliary,data,ids,device))*len(ids)
        count+=len(ids)
    return total/count


def shuffled_game_order(data):
    """Shuffle games then rows within each game; keep one decompressed shard hot."""
    groups={}
    for index,(shard,_) in enumerate(data.rows):
        groups.setdefault(shard,[]).append(index)
    keys=list(groups)
    result=[]
    for position in torch.randperm(len(keys)).tolist():
        values=groups[keys[position]]
        result.extend(values[i] for i in torch.randperm(len(values)).tolist())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices="abc",required=True)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--teacher",required=True)
    p.add_argument("--primitive-dataset",type=Path,required=True)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--seed",type=int,default=3101)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    verify(a.dataset)
    manifest=read(a.dataset/"manifest.json")
    if a.teacher!=manifest["teacher"]:raise ValueError("explicit BC teacher does not match manifest")
    if not a.smoke and not manifest["training_ready"]:
        raise RuntimeError("diagnostic dataset cannot produce a formal BC checkpoint")
    cfg=read(Path(__file__).parent/"configs/pretrain"/f"model_{a.model}.json")
    epochs=1 if a.smoke else cfg["epochs"]
    train=BCDataset(a.dataset,"train")
    cal=train if a.smoke else BCDataset(a.dataset,"calibration")
    primitive_manifest=read(a.primitive_dataset/"manifest.json")
    if not a.smoke and not primitive_manifest["training_ready"]:
        raise RuntimeError("diagnostic primitive dataset is not admitted for formal BC")
    primitive_train=PrimitiveDataset(a.primitive_dataset,"train")
    primitive_cal=(torch.utils.data.Subset(primitive_train,list(range(min(4,len(primitive_train)))))
                   if a.smoke else PrimitiveDataset(a.primitive_dataset,"calibration"))
    if not len(primitive_train) or not len(primitive_cal):raise RuntimeError("missing expert split")
    identity={"primitive_dataset_sha256":sha(a.primitive_dataset/"manifest.json"),
              "primitive_teacher":primitive_manifest["teacher"],"model":a.model,"seed":a.seed,"teacher":a.teacher,"config":cfg,"smoke":a.smoke,
              "dataset_sha256":sha(a.dataset/"manifest.json"),"sources":source_lock()}
    if a.run_dir.exists()!=a.resume:raise RuntimeError("BC fresh/resume mismatch")
    a.run_dir.mkdir(parents=True,exist_ok=a.resume)
    if a.resume:
        if read(a.run_dir/"identity.json")!=identity:raise RuntimeError("BC resume identity mismatch")
    else:json_write(a.run_dir/"identity.json",identity)
    with (a.run_dir/".lock").open("a") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempt=a.run_dir/f"attempt_{time.time_ns()}";attempt.mkdir()
        mlflow.set_tracking_uri(f"sqlite:///{ARTIFACTS}/mlflow.db")
        mlflow.set_experiment("lux-s3-expert-bc")
        with mlflow.start_run(run_name=f"BC-{a.model}-{a.seed}{'-smoke' if a.smoke else ''}"):
            mlflow.log_params({**cfg,"model":a.model,"seed":a.seed,"teacher":a.teacher,
                               "dataset_sha256":identity["dataset_sha256"],"smoke":a.smoke})
            setup(a.device,a.seed)
            actor,_=models(a.model,a.device)
            auxiliary=torch.nn.Linear(37,6).to(a.device)
            parameters=list(actor.parameters())+list(auxiliary.parameters())
            optimizer=torch.optim.AdamW(parameters,lr=cfg["lr"],weight_decay=cfg["weight_decay"])
            first=0;history=[];updates=0
            if a.resume:
                path=committed(a.run_dir)
                if path is None:raise RuntimeError("no committed BC state")
                state=torch.load(path/"checkpoint.pt",map_location="cpu",weights_only=False)
                actor.load_state_dict(state["actor"]);auxiliary.load_state_dict(state["auxiliary"])
                optimizer.load_state_dict(state["optimizer"])
                first=state["epoch"];history=state["history"];updates=state["updates"]
                restore_rng(state["rng"])
            try:
                for epoch in range(first,epochs):
                    start=time.monotonic()
                    if a.device=="cuda":torch.cuda.reset_peak_memory_stats()
                    order=shuffled_game_order(train)
                    if a.smoke:order=order[:min(4,len(order))]
                    primitive_order=shuffled_game_order(primitive_train)
                    train_loss=[];grad=[];primitive_losses=[]
                    for start_i in range(0,len(order),cfg["batch_size"]):
                        indices=order[start_i:start_i+cfg["batch_size"]]
                        value,_=loss(actor,train,indices,a.device)
                        aux_ids=[primitive_order[(start_i+j)%len(primitive_order)] for j in range(len(indices))]
                        extra=primitive_loss(actor,auxiliary,primitive_train,aux_ids,a.device)
                        objective=value+cfg["primitive_aux_weight"]*extra
                        norm=grad_step(objective,optimizer,cfg["grad_clip"],parameters)
                        primitive_losses.append(float(extra.detach()))
                        updates+=1;train_loss.append(float(value.detach()));grad.append(norm)
                    macro_calibration=evaluate(actor,cal,a.device,cfg["batch_size"])
                    primitive_calibration=evaluate_primitive(actor,auxiliary,primitive_cal,a.device,cfg["batch_size"])
                    calibration=macro_calibration+cfg["primitive_aux_weight"]*primitive_calibration
                    metrics={"epoch":epoch+1,"train_nll":float(np.mean(train_loss)),"calibration_nll":calibration,
                             "macro_calibration_nll":macro_calibration,"primitive_calibration_nll":primitive_calibration,
                             "primitive_train_nll":float(np.mean(primitive_losses)),
                             "grad_norm":float(np.mean(grad)),"updates":updates,"seconds":time.monotonic()-start,
                             "gpu_peak_mib":torch.cuda.max_memory_reserved()/2**20 if a.device=="cuda" else 0}
                    history.append(metrics)
                    state={"schema":"macro_bc_v1","identity":identity,"actor":actor.state_dict(),
                           "auxiliary":auxiliary.state_dict(),"optimizer":optimizer.state_dict(),"rng":rng_state(),"epoch":epoch+1,
                           "updates":updates,"history":history}
                    stage=attempt/f"epoch_{epoch:04d}";stage.mkdir()
                    commit(a.run_dir,stage,epoch,state,metrics)
                    mlflow.log_metrics(metrics,step=epoch+1)
                    restore_rng(state["rng"])
                    print("BC_COMMITTED",metrics,flush=True)
                selected=min(history,key=lambda x:(x["calibration_nll"],x["epoch"]))
                checkpoint=a.run_dir/f"batch_{selected['epoch']-1:06d}/checkpoint.pt"
                selected_path=a.run_dir/"selected.json"
                record={"checkpoint":str(checkpoint.absolute()),"sha256":sha(checkpoint),
                        "selected_epoch":selected["epoch"],"criterion":"calibration_nll_then_earliest_epoch","calibration_objective":"macro_conditional_nll_plus_fixed_primitive_aux",
                        "model":a.model,"teacher":a.teacher,"smoke":a.smoke,
                        "dataset_sha256":identity["dataset_sha256"]}
                if selected_path.exists():
                    if read(selected_path)!=record:raise RuntimeError("BC selection is immutable")
                else:json_write(selected_path,record)
                if not a.smoke and not (a.run_dir/"exploration.json").exists():
                    frozen=torch.load(checkpoint,map_location="cpu",weights_only=False)
                    actor.load_state_dict(frozen["actor"])
                    auxiliary.load_state_dict(frozen["auxiliary"])
                    exploration=evaluate(actor,BCDataset(a.dataset,"exploration"),a.device,cfg["batch_size"])
                    json_write(a.run_dir/"exploration.json",{"selected_sha256":sha(selected_path),
                                                           "macro_nll":exploration,
                                                           "primitive_nll":evaluate_primitive(actor,auxiliary,PrimitiveDataset(a.primitive_dataset,"exploration"),a.device,cfg["batch_size"]),
                                                           "selection_already_frozen":True})
            except BaseException as exc:
                json_write(attempt/"failure.json",{"error":type(exc).__name__,"message":str(exc)})
                raise


if __name__=="__main__":main()
