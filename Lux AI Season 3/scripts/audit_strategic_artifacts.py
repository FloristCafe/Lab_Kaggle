"""Post-run checks and derived statistics; never retrains or changes data."""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.strategic.data import history_at, unpack
from agents.strategic.experiment import digest, save_json, setup
from agents.strategic.features import CausalObservationBuilder, MODES
from agents.strategic.model import CentralizedCritic, StrategicActor
from scripts.replay_parser import _as_obs
from scripts.build_strategic_dataset import parse_raw


def hist_stats(hist):
    hist = np.asarray(hist)
    cumulative = hist.cumsum()
    return {
        "mean": float(hist @ np.arange(len(hist)) / hist.sum()),
        "quantiles": {
            str(q): int(np.searchsorted(cumulative, q * hist.sum()))
            for q in (0.25, 0.5, 0.75, 0.9, 0.95)
        },
        "above12_fraction": float(hist[13:].sum() / hist.sum()),
    }


def aggregate_arm(summary):
    ev = summary["evaluations"]
    metrics = {
        key: sum(e["metrics"].get(key, 0) for e in ev) for key in ev[0]["metrics"]
    }
    states = {
        key: sum(e["tactical_states"][key] for e in ev)
        for key in ev[0]["tactical_states"]
    }
    tactical = {
        key: sum(e["tactical_diagnostics"][key] for e in ev)
        for key in ev[0]["tactical_diagnostics"]
    }
    modes = np.sum([e["mode_distribution"] for e in ev], 0)
    result = {
        "win_rate": summary["win_rate"],
        "mean_score": summary["mean_score"],
        "mean_opponent_score": summary["mean_opponent_score"],
        "aggregate_metrics": metrics,
        "tactical_states": states,
        "tactical_diagnostics": tactical,
        "mode_counts": modes.tolist(),
        "mode_fractions": (modes / modes.sum()).tolist(),
        "match_wins": sum(e["match_wins"] for e in ev),
        "opponent_match_wins": sum(e["opponent_match_wins"] for e in ev),
        "sap_efficiency": metrics["sap_effective_damage"]
        / max(1, metrics["sap_energy_spent"]),
        "actor_changed": summary["initial_actor_digest"]
        != summary["final_actor_digest"],
        "primitive_training_steps": summary["primitive_training_steps"],
        "optimizer_updates": summary["optimizer_updates"],
        "wall_seconds": summary["wall_seconds"],
        "max_initial_logprob_error": max(
            (u["first_update_logprob_error"] for u in summary["updates"]), default=0
        ),
    }
    for key in (
        "goal_entropy",
        "joint_entropy",
        "survival",
        "death_rate_per_100_steps",
        "goal_persistence",
        "replanning_frequency",
        "goal_change_frequency",
        "mean_goal_age",
        "mean_tau",
    ):
        result["mean_" + key] = float(np.mean([e[key] for e in ev]))
    return result


def main():
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'experiments/strategic_audit_v2')
    parser.add_argument('--matrix-dir', type=Path, default=ROOT/'experiments/CDE_goal_mappo_v2')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(exist_ok=True)
    data_dir = ROOT / "data/dataset_v2_goal"
    manifest = json.loads((data_dir / "manifest.json").read_text())
    summary = json.loads((data_dir / "summary.json").read_text())
    il_manifest = json.loads(
        (ROOT / "experiments/B_strategic_goal_il/manifest.json").read_text()
    )
    arms = json.loads((args.matrix_dir / "summary.json").read_text())
    # Preserve the EXACT pre-style-cleanup sources used to generate data and runs.
    source = out / "executed_source"
    paths = list((ROOT / "agents/strategic").glob("*.py")) + [
        ROOT / "scripts" / name
        for name in (
            "build_strategic_dataset.py",
            "train_strategic_il.py",
            "train_goal_mappo.py",
            "strategic_env_worker.py",
            "reproduce_waypoint_il.py",
            "check_goal_env.py",
        )
    ]
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert digest(path) == digest(destination), (
                "executed sources changed; use original snapshot"
            )
        else:
            shutil.copyfile(path, destination)
        hashes[str(relative)] = digest(path)
    save_json(out / "executed_source_sha256.json", hashes)
    assert all(hashes[k] == v for k, v in manifest["pipeline_sha256"].items())
    assert digest(data_dir / "manifest.json") == il_manifest["dataset_manifest_sha256"]
    train = set(il_manifest["train_files"])
    val = set(il_manifest["validation_files"])
    assert not train & val and len(train) == 320 and len(val) == 80
    seeds = lambda names: {manifest["files"][name]["source_seed"] for name in names}
    raw_dir = ROOT / "data/raw_replays"
    dtype_counts = {}
    examples = {name: [] for name in MODES}
    invalid_sap = total_sap = invalid_anchor_segments = invalid_anchor_labels = (
        sap_anchor_segments
    ) = 0
    frame_count = 0
    for name, row in manifest["files"].items():
        path = data_dir / name
        assert digest(path) == row["npz_sha256"]
        with np.load(path, allow_pickle=False) as d:
            assert (
                d["feature_masks"].dtype == np.uint8
                and d["feature_values"].dtype == np.float16
            )
            assert (
                d["goal_regions"].dtype == np.uint8
                and d["label_valid"].dtype == np.bool_
            )
            assert ((d["feature_masks"] == 0) | (d["feature_masks"] == 1)).all()
            frame_count += len(d["steps"])
            for key in (
                "feature_masks",
                "feature_values",
                "positions",
                "energies",
                "actions",
                "label_valid",
                "goal_regions",
            ):
                dtype_counts[key] = str(d[key].dtype)
            meta = json.loads(str(d["metadata"]))
            cfg = meta["env_cfg"]
            sap = (d["actions"][:, :, 0] == 5) & d["active"]
            target = d["positions"] + d["actions"][:, :, 1:]
            legal = (
                (d["energies"] >= cfg["unit_sap_cost"])
                & (np.abs(d["actions"][:, :, 1:]).max(-1) <= cfg["unit_sap_range"])
                & (target >= 0).all(-1)
                & (target < 24).all(-1)
            )
            total_sap += int(sap.sum())
            invalid_sap += int((sap & ~legal).sum())
            segments = json.loads(path.with_suffix(".segments.json").read_text())
            for s in segments:
                t, uid = s["anchor_frame"], s["unit_id"]
                if s["valid"] and s["mode"] == "INTERCEPT" and sap[t, uid]:
                    sap_anchor_segments += 1
                    if not legal[t, uid]:
                        invalid_anchor_segments += 1
                        invalid_anchor_labels += t - s["start"] + 1
            for m, mode in enumerate(MODES):
                if len(examples[mode]) >= 3:
                    continue
                valid = d["label_valid"] & (d["mode"] == m)
                ids = np.argwhere(valid)
                if len(ids):
                    t, uid = map(int, ids[0])
                    sid = int(d["segment_id"][t, uid])
                    examples[mode].append(
                        {
                            "file": name,
                            "timestep": t,
                            "unit_slot": uid,
                            "position": d["positions"][t, uid].tolist(),
                            "goal": d["goal"][t, uid].tolist(),
                            "confidence": float(d["label_confidence"][t, uid]),
                            "segment": segments[sid],
                        }
                    )
    # Actual replay prefix vs stored online-ready inputs, both physical sides.
    first = next(iter(manifest["files"]))
    raw = json.loads((raw_dir / (first.split("_p")[0] + ".json")).read_text())
    prefix_checks = []
    config = manifest["config"]
    for side in (0, 1):
        # Full 506-frame parser vs physically truncated JSON: future observations
        # and all future action records are absent from the second input file.
        full = parse_raw(raw_dir / (first.split("_p")[0] + ".json"), side, config)
        truncated = {**raw, "steps": raw["steps"][:110]}
        with tempfile.TemporaryDirectory(dir=out) as temporary:
            prefix_path = Path(temporary) / "prefix.json"
            prefix_path.write_text(json.dumps(truncated))
            prefix = parse_raw(prefix_path, side, config)
        keys = ("features", "unit_vectors", "positions", "energies", "active", "rank")
        for key in keys:
            np.testing.assert_array_equal(full[key][:110], prefix[key])
        prefix_checks.append(
            {
                "side": side,
                "frames": 110,
                "full_frames": len(full["steps"]),
                "compared_arrays": keys,
                "passed": True,
            }
        )
    with np.load(data_dir / first) as z:
        d = unpack(z)
        side = int(d["team_id"])
        builder = CausalObservationBuilder(side, raw["configuration"]["env_cfg"])
        for t, records in enumerate(raw["steps"][:110]):
            f = builder.observe(_as_obs(records[side]))
            np.testing.assert_array_equal(
                f["history"], history_at(d["features"], d["steps"], d["match_steps"], t)
            )
            np.testing.assert_array_equal(f["positions"], d["positions"][t])
    run_ids = [
        summary["mlflow_run_id"],
        json.loads(
            (ROOT / "experiments/A_waypoint_reproduction/summary.json").read_text()
        )["mlflow_run_id"],
        json.loads((ROOT / "experiments/B_strategic_goal_il/summary.json").read_text())[
            "mlflow_run_id"
        ],
    ] + [s["mlflow_run_id"] for s in arms.values()]
    statuses = {run_id: mlflow.get_run(run_id).info.status for run_id in run_ids}
    assert all(s == "FINISHED" for s in statuses.values())
    result = {
        "frames": frame_count,
        "dtype_schema": dtype_counts,
        "all_400_npz_hashes_verified": True,
        "all_400_mask_values_binary": True,
        "split_disjoint": True,
        "train_unique_seeds": len(seeds(train)),
        "validation_unique_seeds": len(seeds(val)),
        "train_validation_seed_overlap": list(seeds(train) & seeds(val)),
        "missing_seed_files": sum(
            row["source_seed"] is None for row in manifest["files"].values()
        ),
        "old_corpus_overlap": manifest["old_corpus_overlap"],
        "goal_distance": hist_stats(summary["goal_distance_hist"]),
        "valid_segment_duration": hist_stats(summary["valid_segment_duration_hist"]),
        "mode_fractions": {
            k: v / summary["valid_samples"] for k, v in summary["mode_counts"].items()
        },
        "invalid_sap_attempts": invalid_sap,
        "all_sap_attempts": total_sap,
        "valid_sap_anchor_segments": sap_anchor_segments,
        "invalid_sap_anchor_segments": invalid_anchor_segments,
        "labels_with_invalid_sap_anchor": invalid_anchor_labels,
        "raw_json_count_preserved": len(list(raw_dir.glob("*.json"))),
        "npz_bytes": sum(p.stat().st_size for p in data_dir.glob("*.npz")),
        "dataset_directory_bytes": sum(
            p.stat().st_size for p in data_dir.rglob("*") if p.is_file()
        ),
        "actor_parameters": sum(p.numel() for p in StrategicActor().parameters()),
        "critic_parameters": sum(p.numel() for p in CentralizedCritic().parameters()),
        "prefix_checks": prefix_checks,
        "disk_online_history_exact_match_frames": 110,
        "mlflow_status": statuses,
        "arms": {k: aggregate_arm(v) for k, v in arms.items()},
    }
    save_json(out / "summary.json", result)
    save_json(out / "label_examples.json", examples)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
