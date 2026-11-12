"""Paired public Dev24 evaluation with an independent aggregate command."""
import argparse
import fcntl
import itertools
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import time
from pathlib import Path

import numpy as np
import torch

from agents.mappo.shared.environment import TeamEnv
from training.contracts import (CONFIG, config, engine_contract, json_write, map_pool, read,
                                setup, sha, source_lock, sync)
from training.core import models, play


def checkpoints(paths):
    specs = []
    current = source_lock()
    engine = engine_contract()
    for path in paths:
        path = Path(path)
        digest = sha(path)
        receipt = read(path.parent/"commit.json")
        if receipt.get(path.name) != digest:
            raise RuntimeError("checkpoint digest mismatch")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        contract = ckpt["contract"]
        model = contract["model"]
        if (contract["smoke"] or ckpt["games"] <= 0 or contract["sources"] != current
            or contract["maps_sha256"] != sha(CONFIG/"maps.json")
            or contract["runtime"]["engine"] != engine or contract["config"] != config(model)):
            raise RuntimeError("incompatible checkpoint")
        specs.append((model, contract["seed"], digest, ckpt))
    if len({(a, b) for a, b, _, _ in specs}) != len(specs):
        raise ValueError("one checkpoint per model and seed")
    if len({ckpt["steps"] for _, _, _, ckpt in specs}) != 1:
        raise ValueError("paired checkpoints must have equal primitive-step budgets")
    return sorted(specs, key=lambda x: x[:2])


def evaluate(args):
    if args.opponent_registry:
        os.environ["LUX_OPPONENT_REGISTRY"]=str(args.opponent_registry.absolute())
    if args.opponent!="v2":
        from training.opponents.pool import registry
        opponents=registry()
        if not opponents["agents"].get(args.opponent,{}).get("enabled"):
            raise RuntimeError("evaluation opponent is not admitted")
    specs = checkpoints(args.checkpoint)
    if args.output.exists() != args.resume:
        raise RuntimeError("use a fresh output path or --resume")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    pool = map_pool()["development"]
    cases = [(i, side, m, s, h) for i in range(len(pool)) for side in range(2) for m, s, h, _ in specs]
    identity = {"maps": pool, "maps_sha256": sha(CONFIG/"maps.json"), "cases": cases,
                "source": source_lock(), "opponent": args.opponent, "device": args.device,
                "opponent_registry_sha256":sha(os.environ["LUX_OPPONENT_REGISTRY"]) if args.opponent!="v2" else None,
                "greedy": "native joint Mode×Goal, conditional sequential UID"}
    # Use the same JSON types before identity comparison.
    identity["cases"] = [list(x) for x in cases]
    dest = args.output/"identity.json"
    with (args.output/".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            if read(dest) != identity:
                raise RuntimeError("evaluation identity mismatch")
        else:
            json_write(dest, identity)
        attempt = args.output/f"attempt_{time.time_ns()}"
        attempt.mkdir()
        setup(args.device, 0)
        # Keep model copies on CPU. Move one at a time to CUDA.
        actors = {}
        for m, s, _, ckpt in specs:
            actor, _ = models(m, "cpu")
            actor.load_state_dict(ckpt["actor"])
            actors[(m,s)] = actor.eval()
        worker = TeamEnv(config(specs[0][0]), attempt/"worker.log")
        try:
            for pos, (i, side, m, s, h) in enumerate(cases):
                path = args.output/f"case_{pos:04d}"
                if path.exists():
                    verify_case(path, identity["cases"][pos], pool[i],args.opponent)
                    continue
                actor = actors[m,s].to(args.device)
                result = play(worker, actor, pool[i], side, args.device, opponent=args.opponent, greedy=True, audit_actions=True)
                actor.to("cpu")
                if not result.summary["complete"]:
                    raise RuntimeError("evaluation did not reach official termination")
                stage = attempt/f"staging_{pos:04d}"
                stage.mkdir()
                json_write(stage/"case.json", {"identity": list(cases[pos]), "summary": result.summary,
                                                "trajectory": result.trajectory, "options": result.options})
                json_write(stage/"sha256.json", {"case.json": sha(stage/"case.json")})
                sync(stage)
                stage.rename(path)
                sync(args.output)
                print(pos+1, "/", len(cases), m, s, i, side, result.summary["game_margin"], flush=True)
        except BaseException as exc:
            json_write(attempt/"failure.json", {"error": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            worker.close()
    print_summary(args.output)


def verify_case(path, case, root,expected_opponent="v2"):
    raw = path/"case.json"
    if read(path/"sha256.json") != {"case.json": sha(raw)}:
        raise RuntimeError("case SHA256 mismatch")
    data = read(raw)
    trajectory = data["trajectory"]
    if data["identity"] != case or [x["t"] for x in trajectory] != list(range(1, 506)):
        raise RuntimeError("case identity or step sequence mismatch")
    if [x["t"] for x in trajectory if x["reset"]] != [101, 202, 303, 404, 505]:
        raise RuntimeError("invalid submatch boundary")
    if any(x["terminal"] for x in trajectory[:-1]) or not trajectory[-1]["terminal"]:
        raise RuntimeError("invalid terminal boundary")
    summary = data["summary"]
    if (summary["seed"] != root or summary["side"] != case[1] or summary["opponent"] != expected_opponent
        or summary["goal_conflicts"] or not summary["complete"]):
        raise RuntimeError("invalid committed game metadata")
    wins = trajectory[-1]["team_wins"]
    side = case[1]
    margin = wins[side] - wins[1-side]
    if sum(wins) != 5 or wins != summary["official_wins"] or margin != summary["game_margin"]:
        raise RuntimeError("official margin mismatch")
    return margin


def summarize(out):
    identity = read(out/"identity.json")
    maps = identity["maps"]
    groups = {}
    expected = {f"case_{i:04d}" for i in range(len(identity["cases"]))}
    actual = {p.name for p in out.glob("case_*")}
    if actual != expected:
        raise RuntimeError(f"missing or unexpected cases: {len(expected-actual)} / {len(actual-expected)}")
    for pos, case in enumerate(identity["cases"]):
        i, side, model, seed, _ = case
        key = model, seed
        if key not in groups:
            groups[key] = np.full((len(maps), 2), np.nan)
        if np.isfinite(groups[key][i,side]):
            raise RuntimeError("duplicate map-side result")
        groups[key][i,side] = verify_case(out/f"case_{pos:04d}", case, maps[i],identity.get("opponent","v2"))
    if any(not np.isfinite(v).all() for v in groups.values()):
        raise RuntimeError("incomplete grid")
    cfg = config("a")
    rng = np.random.default_rng(cfg["eval_bootstrap_seed"])
    draw = rng.integers(0, len(maps), (cfg["eval_bootstrap_draws"], len(maps)))
    def stats(row):
        by_map = row.mean(1)
        ci = np.quantile(by_map[draw].mean(1), [.025,.975])
        return {"mean": float(row.mean()), "mean_by_side": row.mean(0).tolist(),
                "maps": [{"map": root, "mean": float(v), "by_side": list(s)}
                         for root, v, s in zip(maps, by_map, row, strict=True)],
                "map_bootstrap95": ci.tolist()}
    fits = {f"{m}-{s}": dict(stats(v), wins=int((v>0).sum()), ties=int((v==0).sum()), losses=int((v<0).sum()))
            for (m,s),v in groups.items()}
    model_rows = {m: np.stack([v for (model, _),v in groups.items() if m==model]).mean(0)
                  for m in sorted({m for m, _ in groups})}
    comparisons = {}
    for a,b in itertools.combinations(model_rows,2):
        sa = {s for m,s in groups if m==a}
        sb = {s for m,s in groups if m==b}
        if sa == sb:
            comparisons[f"{b}-{a}"] = stats(model_rows[b]-model_rows[a])
    return {"games": len(expected), "steps": 505*len(expected), "fits": fits,
            "model_equal_seed": {m: stats(v) for m,v in model_rows.items()},
            "comparisons": comparisons, "limits": "Map clusters, fixed fitted seeds, previously used Dev24."}


def print_summary(path):
    import json
    print(json.dumps(summarize(path), indent=2, allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--opponent",default="v2")
    p.add_argument("--opponent-registry",type=Path)
    p.add_argument("--summarize", action="store_true")
    a = p.parse_args()
    if a.summarize:
        if a.checkpoint or a.resume:
            p.error("--summarize reads committed evaluation cases")
        print_summary(a.output)
    else:
        if not a.checkpoint:
            p.error("--checkpoint is required")
        evaluate(a)


if __name__ == "__main__":
    main()
