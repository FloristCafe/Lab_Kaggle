"""User-run paired development schedule for the v1 rule teacher."""

import argparse
import hashlib
import json
import os
from pathlib import Path
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
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def fixtures(seed, opponents, self_play):
    for opponent in opponents:
        for side in (0, 1):
            yield seed, opponent, side
    if self_play:
        yield seed, "v1", 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--opponents", nargs="+", choices=("v0", "starter"), default=["v0"])
    parser.add_argument("--self-play", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--suite", default="v1_dev")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.seed_count < 1 or args.timeout < 1:
        parser.error("seed-count and timeout must be positive")
    if args.suite in ("", ".", "..") or Path(args.suite).name != args.suite:
        parser.error("suite must be one directory name")
    if not RUNNER.exists():
        parser.error(f"missing official runner: {RUNNER}")

    replay_dir = ROOT / "replays" / args.suite
    trace_dir = ROOT / "artifacts" / "v1_traces" / args.suite
    output_dir = ROOT / "outputs" / args.suite
    for folder in (replay_dir, output_dir):
        folder.mkdir(parents=True, exist_ok=True)
    if args.trace:
        trace_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB}")
    experiment_name = "lux-s3-v1-rule-dispatcher"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=ARTIFACT_ROOT.as_uri())
    mlflow.set_experiment(experiment_name)
    git_revision = git_value(ROOT, "rev-parse", "HEAD")
    git_status = git_value(ROOT, "status", "--short")
    env_revision = git_value(ROOT / "vendor" / "lux-design-s3", "rev-parse", "HEAD")
    agent_hashes = {}
    for agent_name in ("v1", "v0", "starter"):
        digest = hashlib.sha256()
        for source in sorted((ROOT / "agents" / agent_name).rglob("*.py")):
            digest.update(str(source.relative_to(ROOT)).encode())
            digest.update(source.read_bytes())
        agent_hashes[agent_name] = digest.hexdigest()
    manifest = {
        "agent_hashes": agent_hashes,
        "environment_revision": env_revision,
        "schedule_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trace": args.trace,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            parser.error("suite contains a different agent, environment, or trace setting; use a new --suite")
    elif any(replay_dir.glob("*.json")) or (output_dir / "results.jsonl").exists():
        parser.error("suite has prior games without a manifest; use a new --suite")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        for _, opponent, side in fixtures(seed, args.opponents, args.self_play):
            name = f"v1_{opponent}_{seed}_p{side}"
            replay = replay_dir / f"{name}.json"
            if replay.exists():
                print(f"skip existing {replay}", flush=True)
                continue
            agents = ["v1", opponent] if side == 0 else [opponent, "v1"]
            command = [
                str(RUNNER), *(str(ROOT / "agents" / agent / "main.py") for agent in agents),
                "--seed", str(seed), "--output", str(replay),
            ]
            environment = os.environ.copy()
            environment["PATH"] = f"{ENV_BIN}:{environment.get('PATH', '')}"
            if args.trace:
                environment["LUX_V1_TRACE_DIR"] = str(trace_dir)
                environment["LUX_V1_TRACE_TAG"] = name
            else:
                environment.pop("LUX_V1_TRACE_DIR", None)
                environment.pop("LUX_V1_TRACE_TAG", None)
            environment.pop("LUX_V1_FIXED_TARGET", None)
            print(f"running {name}", flush=True)
            with mlflow.start_run(run_name=name) as run:
                mlflow.log_params({
                    "suite": args.suite, "seed": seed, "opponent": opponent,
                    "v1_side": side, "trace": args.trace,
                    "git_revision": git_revision,
                    "code_sha256": agent_hashes["v1"],
                    "opponent_sha256": agent_hashes[opponent],
                    "environment_revision": env_revision,
                    "python_runner": str(RUNNER),
                })
                mlflow.set_tags({
                    "git_dirty": str(bool(git_status)),
                    "replay_path": str(replay),
                    "agent_revision": "v1_rule_dispatcher",
                })
                started = time.monotonic()
                try:
                    result = subprocess.run(
                        command, cwd=ROOT, env=environment, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        timeout=args.timeout, check=False,
                    )
                    return_code = result.returncode
                    output = result.stdout
                except subprocess.TimeoutExpired as error:
                    return_code = 124
                    output = str(error)
                duration = time.monotonic() - started
                log_path = output_dir / f"{name}.log"
                log_path.write_text(output, encoding="utf-8")
                mlflow.log_metrics({"duration_seconds": duration, "return_code": return_code})
                trace_paths = sorted(trace_dir.glob(f"{name}_*.frames")) if args.trace else []
                mlflow.set_tag("trace_paths", json.dumps([str(path) for path in trace_paths]))
                mlflow.set_tag("log_path", str(log_path))
                row = {
                    "seed": seed, "opponent": opponent, "side": side,
                    "return_code": return_code, "duration_seconds": duration,
                    "replay": str(replay), "traces": [str(path) for path in trace_paths],
                    "mlflow_run_id": run.info.run_id,
                }
                if return_code == 0 and replay.exists():
                    sides = ("player_0", "player_1") if opponent == "v1" else (f"player_{side}",)
                    results = [summarize(replay, "v1", player_side) for player_side in sides]
                    row["results"] = results
                    for index, summary in enumerate(results):
                        suffix = f"p{index}" if opponent == "v1" else "v1"
                        mlflow.log_metrics({
                            f"{suffix}_match_wins": summary["agent_match_wins"],
                            f"{suffix}_points": summary["agent_points"],
                            f"{suffix}_point_margin": summary["agent_points"] - summary["opponent_points"],
                        })
                with (output_dir / "results.jsonl").open("a", encoding="utf-8") as summary_file:
                    summary_file.write(json.dumps(row) + "\n")
                print(f"{name}: exit={return_code}, seconds={duration:.1f}", flush=True)


if __name__ == "__main__":
    main()
