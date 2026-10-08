"""Paired v2 development schedule with immutable fingerprints and MLflow runs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import mlflow

from collect_results import summarize


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / ".venv" / "bin" / "luxai-s3"
ENV_BIN = ROOT / ".venv" / "bin"
TRACKING_DB = Path("/home/issue/ml-workspace/artifacts/mlflow.db")
ARTIFACT_ROOT = Path("/home/issue/ml-workspace/artifacts/mlruns")


def git_value(root, *args):
    result = subprocess.run(["git", *args], cwd=root, text=True,
                            capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def agent_hash(name):
    digest = hashlib.sha256()
    for source in sorted((ROOT / "agents" / name).rglob("*.py")):
        digest.update(str(source.relative_to(ROOT)).encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()


def trace_summary(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    directions = {0: (0, 0), 1: (0, -1), 2: (1, 0),
                  3: (0, 1), 4: (-1, 0)}
    planned_conflicts = 0
    for row in rows:
        starts = {unit_id: tuple(tile) for unit_id, tile in row["positions"].items()}
        next_tiles = {}
        for unit_id, (x, y) in starts.items():
            dx, dy = directions[row["actions"][unit_id]]
            next_tiles[unit_id] = (x + dx, y + dy)
        if any(next_tiles[left] == next_tiles[right]
               and starts[left] != starts[right]
               for index, left in enumerate(starts)
               for right in list(starts)[index + 1:]):
            planned_conflicts += 1
    return {
        "frames": len(rows),
        "contradictions": max((row["contradictions"] for row in rows), default=0),
        "static_rank_final": rows[-1]["rank"] if rows else 0,
        "dynamic_frames": sum(row["parser_reason"] == "dynamic_only" for row in rows),
        "redundant_frames": sum(row["independent"] is False for row in rows),
        "yield_decisions": sum(row["yielded_unit"] is not None for row in rows),
        "planned_conflict_frames": planned_conflicts,
        "agent_decision_seconds": sum(row["decision_seconds"] for row in rows),
        "max_decision_seconds": max((row["decision_seconds"] for row in rows), default=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="v2_epoch_router_dev")
    parser.add_argument("--seed-start", type=int, default=102)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.seed_count < 1 or args.timeout < 1 or Path(args.suite).name != args.suite:
        parser.error("invalid schedule arguments")
    if not RUNNER.exists():
        parser.error(f"missing runner: {RUNNER}")

    replay_dir = ROOT / "replays" / args.suite
    output_dir = ROOT / "outputs" / args.suite
    trace_dir = ROOT / "artifacts" / "v2_traces" / args.suite
    for directory in (replay_dir, output_dir, trace_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "opponent": "v0",
        "sides": [0, 1],
        "agent_hashes": {name: agent_hash(name) for name in ("v2", "v0")},
        "environment_revision": git_value(ROOT / "vendor" / "lux-design-s3", "rev-parse", "HEAD"),
        "schedule_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_revision": git_value(ROOT, "rev-parse", "HEAD"),
        "git_dirty": bool(git_value(ROOT, "status", "--short")),
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            parser.error("suite fingerprint differs; choose a new suite")
    elif (output_dir / "results.jsonl").exists() or any(replay_dir.glob("*.json")):
        parser.error("suite has results without a manifest")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
    experiment_name = "lux-s3-v2-epoch-router"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=ARTIFACT_ROOT.as_uri())
    mlflow.set_experiment(experiment_name)
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        for side in (0, 1):
            name = f"v2_v0_{seed}_p{side}"
            replay = replay_dir / f"{name}.json"
            if replay.exists():
                print(f"skip {name}", flush=True)
                continue
            agents = ["v2", "v0"] if side == 0 else ["v0", "v2"]
            command = [str(RUNNER), *(str(ROOT / "agents" / agent / "main.py")
                                      for agent in agents), "--seed", str(seed),
                       "--output", str(replay)]
            environment = os.environ.copy()
            environment["PATH"] = f"{ENV_BIN}:{environment.get('PATH', '')}"
            environment["LUX_V2_TRACE_DIR"] = str(trace_dir)
            environment["LUX_V2_TRACE_TAG"] = name
            started = time.monotonic()
            print(f"running {name}", flush=True)
            with mlflow.start_run(run_name=name) as run:
                mlflow.log_params({
                    "suite": args.suite, "seed": seed, "v2_side": side,
                    "opponent": "v0", "agent_sha256": manifest["agent_hashes"]["v2"],
                    "opponent_sha256": manifest["agent_hashes"]["v0"],
                    "environment_revision": manifest["environment_revision"],
                    "git_revision": manifest["git_revision"],
                    "python_runner": str(RUNNER),
                })
                mlflow.set_tag("git_dirty", str(manifest["git_dirty"]))
                try:
                    result = subprocess.run(command, cwd=ROOT, env=environment,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True,
                                            timeout=args.timeout, check=False)
                    return_code, output = result.returncode, result.stdout
                except subprocess.TimeoutExpired as error:
                    return_code, output = 124, str(error)
                duration = time.monotonic() - started
                log_path = output_dir / f"{name}.log"
                log_path.write_text(output, encoding="utf-8")
                trace = trace_dir / f"{name}_player_{side}.jsonl"
                diagnostics = trace_summary(trace) if trace.exists() else {}
                invalid_actions = len(re.findall(r"invalid action", output, flags=re.I))
                metrics = {"duration_seconds": duration,
                           "return_code": return_code,
                           "invalid_action_logs": invalid_actions,
                           **diagnostics}
                row = {
                    "seed": seed, "side": side, "opponent": "v0",
                    "return_code": return_code, "duration_seconds": duration,
                    "replay": str(replay), "trace": str(trace) if trace.exists() else None,
                    "log": str(log_path), "mlflow_run_id": run.info.run_id,
                    "metrics": metrics,
                }
                if return_code == 0 and replay.exists():
                    row["result"] = summarize(replay, "v2", f"player_{side}")
                    metrics.update({
                        "match_wins": row["result"]["agent_match_wins"],
                        "point_margin": row["result"]["agent_points"]
                        - row["result"]["opponent_points"],
                    })
                mlflow.log_metrics(metrics)
                mlflow.set_tags({"replay_path": str(replay),
                                 "trace_path": str(trace), "log_path": str(log_path)})
                with (output_dir / "results.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row) + "\n")
                print(f"{name}: exit={return_code}, seconds={duration:.1f}", flush=True)


if __name__ == "__main__":
    main()
