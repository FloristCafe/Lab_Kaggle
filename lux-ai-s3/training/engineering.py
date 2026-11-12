"""Bounded engineering checks. Never uses a registered training or development map."""
import argparse
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from agents.mappo.shared.environment import TeamEnv
from training.contracts import ARTIFACTS, config, json_write, optimizer_pair, setup, sha
from training.core import learn, models, play
from training.evaluate import verify_case


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list,tuple)):
        return len(a) == len(b) and all(equal(x,y) for x,y in zip(a,b,strict=True))
    return a == b


def resume_check(out, model, device):
    continuous, split = out/"continuous", out/"split"
    base = [sys.executable, "-B", "-m", "training.train", "--model", model, "--seed", "1313969260",
            "--device", device, "--smoke", "--steps", "24"]
    for name, dest, extra in (("continuous", continuous, ["--games","2"]),
                             ("split_first", split, []),
                             ("split_resume", split, ["--games","2","--resume","latest"])):
        with (out/f"{name}.log").open("x") as stream:
            subprocess.run([*base, "--run-dir", str(dest), *extra], stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
    steps = ["batch_000000", "batch_000001"]
    compared = {}
    for pos, p in enumerate(steps):
        a = torch.load(continuous/p/"checkpoint.pt", map_location="cpu", weights_only=False)
        b = torch.load(split/p/"checkpoint.pt", map_location="cpu", weights_only=False)
        compared[p] = {k: equal(v,b[k]) for k,v in a.items()}
        compared[p]["game_log_sha256"] = (sha(continuous/p/f"game_{pos:06d}.json") ==
                                            sha(split/p/f"game_{pos:06d}.json"))
        if not all(compared[p].values()):
            json_write(out/"difference.json", compared)
            raise RuntimeError("continuous versus resumed mismatch")
    return compared


def terminal_check(out, model, device, side):
    setup(device, 1313969260)
    cfg = config(model)
    actor, critic = models(model,device)
    aopt, copt = optimizer_pair(actor, critic, cfg["ppo"])
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{ARTIFACTS}/mlflow.db")
    mlflow.set_experiment("lux-s3-mappo-architecture")
    start = time.monotonic()
    env = TeamEnv(cfg, out/"worker.log")
    try:
        result = play(env,actor,15001,side,device,greedy=False,audit_actions=True)
    finally:
        env.close()
    case = [0,side,model,1313969260,"ENGINEERING_RANDOM_INITIALIZATION"]
    json_write(out/"case.json", {"identity":case,"summary":result.summary,
                                  "trajectory":result.trajectory,"options":result.options})
    json_write(out/"sha256.json", {"case.json":sha(out/"case.json")})
    value = verify_case(out,case,15001)
    if not np.isfinite(value):
        raise RuntimeError("invalid official return")
    with mlflow.start_run(run_name=f"{model}-terminal-engineering"):
        metrics = learn([result], actor, critic, aopt, copt, cfg["ppo"], device)
        payload = {"steps":len(result.rewards), "map":15001, "side":side, "terminal_margin":value,
                   "throughput_inference":505/result.seconds, "goal_lifetimes":len(result.options),
                   "engine_resets":result.summary["metrics"]["match_resets"],
                   "gpu_peak_mib":torch.cuda.max_memory_reserved()/2**20 if device == "cuda" else 0,
                   "throughput_with_update":505/(time.monotonic()-start), **metrics}
        mlflow.log_params({"model":model, "public_engineering_map":15001, "side":side})
        mlflow.log_metrics(payload)
        return dict(payload, mlflow_run=mlflow.active_run().info.run_id)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices="abc",required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--side",type=int,choices=(0,1),default=0)
    p.add_argument("--output",type=Path,required=True)
    options=p.add_mutually_exclusive_group(required=True)
    options.add_argument("--resume-check",action="store_true")
    options.add_argument("--terminal-check",action="store_true")
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    try:
        summary=(resume_check(a.output,a.model,a.device) if a.resume_check else
                 terminal_check(a.output,a.model,a.device,a.side))
        result={"status":"PASS","model":a.model,"seconds":time.monotonic()-start,"details":summary}
        json_write(a.output/"result.json",result)
        print(result)
    except BaseException as exc:
        json_write(a.output/"failure.json",{"type":type(exc).__name__,"message":str(exc)})
        raise


if __name__=="__main__":
    main()
