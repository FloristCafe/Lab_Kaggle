"""Convert observation-only v1 frames into one compressed NPZ per agent run."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess

import numpy as np


FEATURE_NAMES = (
    "visible", "never_seen", "observation_age", "visible_empty",
    "visible_nebula", "visible_asteroid", "last_empty", "last_nebula",
    "last_asteroid", "visible_energy", "known_relic_center",
    "relic_support", "score_probability", "score_confidence", "own_count",
    "own_energy", "enemy_count", "enemy_energy", "enemy_proximity",
    "own_proximity", "previous_target_count", "previous_occupied",
)


def read_frames(path):
    frames = []
    with path.open("rb") as source:
        while size := source.read(4):
            if len(size) != 4:
                raise ValueError(f"{path}: incomplete frame header")
            length = struct.unpack("<I", size)[0]
            payload = source.read(length)
            if len(payload) != length:
                raise ValueError(f"{path}: incomplete frame payload")
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                frames.append({key: data[key] for key in data.files})
    if not frames:
        raise ValueError(f"{path}: no decisions")
    return frames


def git_value(*args):
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v1_sl"))
    parser.add_argument("--allow-probe", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.traces:
        frames = read_frames(path)
        keys = frames[0].keys()
        if any(frame.keys() != keys for frame in frames):
            raise ValueError(f"{path}: inconsistent frame fields")
        data = {key: np.stack([frame[key] for frame in frames]) for key in keys}
        if bool(np.any(data["probe_mode"])) and not args.allow_probe:
            raise ValueError(f"{path}: fixed-target probe is not an SL teacher trace")
        if data["global_features"].shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"{path}: feature schema mismatch")
        steps = data["step"]
        if not np.array_equal(steps, np.arange(steps[0], steps[0] + len(steps))):
            raise ValueError(f"{path}: missing or repeated steps")
        metadata = {
            "source_trace": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "feature_names": FEATURE_NAMES,
            "git_revision": git_value("rev-parse", "HEAD"),
            "git_status": git_value("status", "--short"),
            "environment_revision": "c6d1e665d12a467b299fb586b3f876e47847746b",
            "label_policy": "v1_rule_dispatcher_observation_only",
        }
        output = args.output_dir / f"{path.stem}.npz"
        np.savez_compressed(output, **data, metadata=json.dumps(metadata))
        print(f"{output}: {len(frames)} decisions")


if __name__ == "__main__":
    main()
