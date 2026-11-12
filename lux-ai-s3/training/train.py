"""Foreground serial collection and complete-game checkpoint recovery."""
import argparse
import fcntl
import os
# MLflow telemetry uses global random in a background thread. Keep tracking local and synchronous.
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
os.environ["MLFLOW_ENABLE_ASYNC_LOGGING"] = "false"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import resource
import sys
import time
from pathlib import Path

import mlflow
import torch

from agents.mappo.shared.environment import TeamEnv
from training.contracts import (ARTIFACTS, CONFIG, ROOT, commit, committed, config,
                                engine_contract, git_identity, json_write, map_pool,
                                optimizer_pair, read, restore_rng, rng_state, setup, sha,
                                source_lock, training_slot)
from training.core import learn, models, play
from training.expert.initialize import load_bc
from training.opponents.pool import Pool,registry


def train(args):
    cfg = config(args.model)
    if args.opponent_registry is not None:
        os.environ["LUX_OPPONENT_REGISTRY"] = str(args.opponent_registry.absolute())
    if not args.smoke and args.bc_checkpoint is None and args.resume is None:
        raise ValueError("formal RL requires the matching BC checkpoint")
    opt_cfg = cfg["ppo"]
    if not args.smoke and args.seed not in cfg["fit_seeds"]:
        raise ValueError("formal seed must be frozen in the config")
    if args.smoke and (args.steps < 20 or args.steps > 130):
        raise ValueError("smoke steps must be 20–130")
    if not args.smoke and args.steps != 112:
        raise ValueError("--steps is exclusively for smoke")
    if not args.smoke and args.games is not None and (not 1 <= args.games <= cfg["games"]
                                                      or args.games % opt_cfg["batch_games"]):
        raise ValueError("--games must be on a complete update boundary inside the frozen budget")
    if args.smoke and args.games not in (None, 1, 2):
        raise ValueError("smoke permits one or two short episodes")
    run = Path(args.run_dir).expanduser()
    if args.resume is None:
        run.mkdir(parents=True, exist_ok=False)
    elif not run.is_dir():
        raise FileNotFoundError(run)
    with (run / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args, run, cfg, opt_cfg)


def execute(args, run, cfg, opt_cfg):
    if args.bc_checkpoint is None and args.resume is not None:
        args.bc_checkpoint = Path(read(run/"identity.json")["bc_init"]["selection"])
    if args.bc_checkpoint is not None:
        raw = read(args.bc_checkpoint) if args.bc_checkpoint.suffix == ".json" else None
        selected = Path(raw["checkpoint"]) if raw else args.bc_checkpoint
        bc_identity = {"selection":str(args.bc_checkpoint.absolute()),"sha256":sha(selected)}
    else:
        bc_identity = None
    roster = registry() if os.environ.get("LUX_OPPONENT_REGISTRY") else None
    if not args.smoke and roster is None:
        raise RuntimeError("formal RL requires an admitted multi-agent registry")
    source = source_lock()
    if not args.smoke and read(ROOT/"release_source_lock.json") != source:
        raise RuntimeError("formal release source lock mismatch")
    pool = map_pool()
    runtime = {"device": args.device, "torch": torch.__version__,
               "python": sys.version.split()[0], "engine": engine_contract(),
               "cuda": torch.version.cuda}
    if args.device == "cuda":
        runtime["gpu"] = torch.cuda.get_device_name()
    contract = {"model": args.model, "seed": args.seed, "config": cfg, "sources": source,
                "maps_sha256": sha(CONFIG/"maps.json"), "runtime": runtime,
                "smoke": args.smoke, "smoke_steps": args.steps if args.smoke else None,
                "bc_init":bc_identity, "opponent_registry_sha256":sha(os.environ["LUX_OPPONENT_REGISTRY"]) if roster else None}
    identity = run / "identity.json"
    target = args.games or (1 if args.smoke else cfg["games"])
    if args.resume is None:
        json_write(identity, contract)
    elif read(identity) != contract:
        raise RuntimeError("resume identity mismatch")
    artifact = ARTIFACTS
    artifact.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{artifact/'mlflow.db'}")
    mlflow.set_experiment("lux-s3-mappo-architecture")
    attempt = run / f"attempt_{time.time_ns()}"
    attempt.mkdir()
    env = None
    try:
        with mlflow.start_run(run_name=f"{args.model}-{args.seed}{'-smoke' if args.smoke else ''}"):
            mlflow.log_params({"model": args.model, "seed": args.seed, "smoke": args.smoke,
                               "maps_sha256": contract["maps_sha256"], **git_identity(),
                               "python_executable": sys.executable, **runtime,
                               "identity_sha256": sha(identity), **opt_cfg})
            json_write(attempt/"tracking.json", {"run_id": mlflow.active_run().info.run_id,
                                                "tracking_uri": mlflow.get_tracking_uri()})
            setup(args.device, args.seed)
            actor, critic = models(args.model, args.device)
            if args.bc_checkpoint is not None:
                loaded_bc = load_bc(actor,args.model,args.bc_checkpoint,args.smoke)
                if loaded_bc["checkpoint_sha256"] != bc_identity["sha256"]:
                    raise RuntimeError("BC identity changed during loading")
            aopt, copt = optimizer_pair(actor, critic, opt_cfg)
            if aopt.state or copt.state:
                raise RuntimeError("fresh MAPPO optimizer unexpectedly contains state")
            if args.bc_checkpoint is not None:
                json_write(attempt/"bc_load.json",dict(loaded_bc,rl_optimizers_empty=True))
            sampler = Pool(args.seed,roster) if roster else None
            done = steps = updates = 0
            saved = committed(run)
            if args.resume is not None:
                expected = saved / "checkpoint.pt" if saved is not None else None
                if expected is None or args.resume != "latest" and Path(args.resume).absolute() != expected.absolute():
                    raise RuntimeError("only the latest committed checkpoint is resumable in place")
                ckpt = torch.load(expected, map_location="cpu", weights_only=False)
                if ckpt["contract"] != contract or ckpt["schedule"] != "fixed":
                    raise RuntimeError("checkpoint configuration mismatch")
                actor.load_state_dict(ckpt["actor"])
                critic.load_state_dict(ckpt["critic"])
                aopt.load_state_dict(ckpt["actor_opt"])
                copt.load_state_dict(ckpt["critic_opt"])
                done, steps, updates = (ckpt[k] for k in ("games", "steps", "updates"))
                if updates != int(saved.name.split("_")[1]) + 1:
                    raise RuntimeError("checkpoint update sequence mismatch")
                if sampler is not None:sampler.load_state_dict(ckpt["opponent_pool"])
                restore_rng(ckpt["rng"])
            if done > target:
                raise RuntimeError("requested budget precedes committed progress")
            if done < target:
                env = TeamEnv(cfg, attempt/"worker.log")
            while done < target:
                begin = time.monotonic()
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                take = 1 if args.smoke else min(opt_cfg["batch_games"], target - done)
                rolls = []
                staged = attempt / f"staging_{updates:06d}"
                staged.mkdir()
                for i in range(take):
                    root, side = (15001, 0) if args.smoke else training_slot(done+i, pool["training"])
                    opponent = sampler.sample() if sampler else "v2"
                    roll = play(env, actor, root, side, args.device, opponent=opponent, limit=args.steps if args.smoke else None,
                                audit_actions=args.smoke)
                    if not args.smoke and not roll.summary["complete"]:
                        raise RuntimeError("short trajectory outside smoke")
                    if sampler is not None and roll.summary["complete"]:
                        sampler.observe(opponent,roll.summary["game_margin"])
                    rolls.append(roll)
                    record = {"summary": roll.summary, "options": roll.options}
                    if args.smoke:
                        record["trajectory"] = roll.trajectory
                    json_write(staged/f"game_{done+i:06d}.json", record)
                metrics = learn(rolls, actor, critic, aopt, copt, opt_cfg, args.device)
                advanced = sum(len(r.rewards) for r in rolls)
                steps += advanced
                done += take
                seconds = time.monotonic() - begin
                metrics.update(games=done, primitive_steps=steps, seconds=seconds,
                               primitive_per_sec=advanced/seconds,
                               rollout_seconds=sum(r.seconds for r in rolls),
                               cpu_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                               gpu_peak_mib=torch.cuda.max_memory_reserved()/2**20 if args.device == "cuda" else 0,
                               actor_params=sum(p.numel() for p in actor.parameters()),
                               critic_params=sum(p.numel() for p in critic.parameters()),
                               legal_goal_fraction=1.0,
                               official_mean_margin=(None if args.smoke else
                                   sum(r.summary["game_margin"] for r in rolls)/take),
                               lr=opt_cfg["lr"],
                               mode_counts=[sum(r.summary["mode_counts"][k] for r in rolls) for k in range(5)],
                               mean_option_duration=sum(o["duration"] for r in rolls for o in r.options) /
                                                    max(1, sum(len(r.options) for r in rolls)))
                if sampler is not None:
                    for opponent_name, (played, won) in sampler.results.items():
                        metrics[f"opponent_games/{opponent_name}"] = played
                        metrics[f"opponent_win_credit/{opponent_name}"] = won
                snapshot = {"contract": contract, "actor": actor.state_dict(), "critic": critic.state_dict(),
                            "actor_opt": aopt.state_dict(), "critic_opt": copt.state_dict(), "rng": rng_state(),
                            "games": done, "steps": steps, "updates": updates+1, "schedule": "fixed",
                            "opponent_pool":sampler.state_dict() if sampler else None}
                path = commit(run, staged, updates, snapshot, metrics)
                updates += 1
                if not args.smoke or all(r.summary["complete"] for r in rolls):
                    tracked = metrics
                else:
                    tracked = {k: v for k, v in metrics.items() if k != "official_mean_margin"}
                try:
                    mlflow.log_metrics({k: v for k, v in tracked.items() if isinstance(v, (int, float))}, step=steps)
                finally:
                    restore_rng(snapshot["rng"])
                print(f"COMMITTED {path} {metrics}", flush=True)
            mlflow.log_artifact(str(identity))
    except BaseException as exc:
        json_write(attempt/"failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices="abc", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--games", type=int, help="stop at this completed-game count, at most the frozen total")
    parser.add_argument("--resume", metavar="PATH_OR_LATEST")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--bc-checkpoint", type=Path)
    parser.add_argument("--opponent-registry",type=Path)
    parser.add_argument("--steps", type=int, default=112, help="primitive budget for --smoke only")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
