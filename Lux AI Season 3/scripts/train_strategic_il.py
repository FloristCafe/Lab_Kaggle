"""Train/evaluate factorized semantic-goal IL on isolated replay groups."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.strategic.data import GoalDataset, collate_goal
from agents.strategic.experiment import digest, provenance, save_json, setup
from agents.strategic.model import (
    StrategicActor,
    actor_input,
    choose_actions,
    supervised_loss,
)
from agents.v3.dataset import ShardFrameSampler
from scripts.train_full_bc import grouped_replay_split


def on_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def evaluate(model, loader, device):
    model.eval()
    counts = dict(
        units=0,
        mode_correct=0,
        joint_mode_correct=0,
        goal_correct=0,
        joint_goal_correct=0,
        joint_correct=0,
        top5=0,
        top10=0,
        anchor=0,
        within3=0,
        pairs=0,
        unchanged=0,
    )
    sums = dict(
        distance=0.0,
        entropy=0.0,
        peak=0.0,
        mode_loss=0.0,
        goal_loss=0.0,
        weights=0.0,
        joint_entropy=0.0,
        joint_peak=0.0,
    )
    confusion = np.zeros((5, 5), np.int64)
    previous = {}
    gx, gy = torch.meshgrid(
        torch.arange(24, device=device), torch.arange(24, device=device), indexing="ij"
    )
    with torch.no_grad():
        for cpu_batch in loader:
            b = on_device(cpu_batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ml, gl = model(*actor_input(b))
            _, losses = supervised_loss(ml, gl, b)
            valid = b["valid"] & b["alive"]
            if not valid.any():
                continue
            true = b["mode"][valid]
            all_goal = gl[valid].float().flatten(-2)
            expert_goals = all_goal[torch.arange(len(true), device=device), true]
            goal = expert_goals.argmax(1)
            region = b["regions"][valid].flatten(1)
            predicted_mode = ml[valid].argmax(1)
            jm, jg, _ = choose_actions(ml, gl, greedy=True)
            jmode, jgoal = jm[valid], jg[valid]
            correct = region.gather(1, goal[:, None]).squeeze(1)
            jcorrect = region.gather(1, jgoal[:, None]).squeeze(1)
            ranked = expert_goals.topk(10, 1).indices
            probability = expert_goals.softmax(1)
            logprob = expert_goals.log_softmax(1)
            distance = (
                (gx.flatten()[None] - goal[:, None] // 24).abs()
                + (gy.flatten()[None] - goal[:, None] % 24).abs()
            ).float()
            distance = distance.masked_fill(~region, torch.inf).min(1).values
            anchor = b["positions"][valid][:, 0] * 24 + b["positions"][valid][:, 1]
            counts["units"] += len(true)
            for key, val in dict(
                mode_correct=(predicted_mode == true).sum(),
                joint_mode_correct=(jmode == true).sum(),
                goal_correct=correct.sum(),
                joint_goal_correct=jcorrect.sum(),
                joint_correct=(jcorrect & (jmode == true)).sum(),
                top5=region.gather(1, ranked[:, :5]).any(1).sum(),
                top10=region.gather(1, ranked).any(1).sum(),
                anchor=(goal == anchor).sum(),
                within3=(distance <= 3).sum(),
            ).items():
                counts[key] += int(val)
            weight = float(b["confidence"][valid].sum())
            sums["mode_loss"] += losses["mode_loss"] * weight
            sums["goal_loss"] += losses["goal_loss"] * weight
            sums["weights"] += weight
            sums["distance"] += float(distance.sum())
            sums["entropy"] += float(-(probability * logprob).sum())
            sums["peak"] += float(probability.max(1).values.sum())
            log_modes = ml[valid].float().log_softmax(1)
            log_goals = all_goal.log_softmax(-1)
            joint = log_modes[:, :, None] + log_goals
            sums["joint_entropy"] += float(-(joint.exp() * joint).sum())
            sums["joint_peak"] += float(joint.flatten(1).exp().max(1).values.sum())
            np.add.at(confusion, (true.cpu().numpy(), predicted_mode.cpu().numpy()), 1)
            # Raw prediction stability, only within the same expert segment.
            pred_cpu = gl.float().flatten(-2).argmax(-1)
            for row in range(len(cpu_batch["history"])):
                shard, step = int(cpu_batch["shard"][row]), int(cpu_batch["step"][row])
                for uid in range(16):
                    if not cpu_batch["valid"][row, uid]:
                        previous.pop(uid, None)
                        continue
                    mode = int(cpu_batch["mode"][row, uid])
                    goal_pred = int(pred_cpu[row, uid, mode])
                    segment = int(cpu_batch["segment_id"][row, uid])
                    prev = previous.get(uid)
                    if (
                        prev
                        and prev[:2] == (shard, segment)
                        and 0 < step - prev[2] <= 8
                    ):
                        counts["pairs"] += 1
                        counts["unchanged"] += goal_pred == prev[3]
                    previous[uid] = (shard, segment, step, goal_pred)
    n = max(1, counts["units"])
    precision = confusion.diagonal() / np.maximum(confusion.sum(0), 1)
    recall = confusion.diagonal() / np.maximum(confusion.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    w = max(sums["weights"], 1e-9)
    return dict(
        mode_accuracy=counts["mode_correct"] / n,
        joint_mode_accuracy=counts["joint_mode_correct"] / n,
        mode_macro_f1=float(f1.mean()),
        mode_balanced_accuracy=float(recall.mean()),
        goal_accuracy_expert_mode=counts["goal_correct"] / n,
        joint_goal_accuracy=counts["joint_goal_correct"] / n,
        joint_mode_goal_accuracy=counts["joint_correct"] / n,
        goal_top5=counts["top5"] / n,
        goal_top10=counts["top10"] / n,
        goal_manhattan=sums["distance"] / n,
        within3=counts["within3"] / n,
        goal_entropy=sums["entropy"] / n,
        peak_probability=sums["peak"] / n,
        anchor_rate=counts["anchor"] / n,
        joint_entropy=sums["joint_entropy"] / n,
        joint_peak_probability=sums["joint_peak"] / n,
        raw_goal_persistence=counts["unchanged"] / max(1, counts["pairs"]),
        persistence_pairs=counts["pairs"],
        loss=(sums["mode_loss"] + sums["goal_loss"]) / w,
        mode_loss=sums["mode_loss"] / w,
        goal_loss=sums["goal_loss"] / w,
        valid_units=counts["units"],
        mode_confusion=confusion.tolist(),
        mode_f1=f1.tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/dataset_v2_goal")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "experiments/B_strategic_goal_il"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/strategic_baseline.yaml"
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--min-games", type=int, default=400)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    config["il"]["epochs"] = args.epochs
    setup(args.seed)
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    paths = sorted(args.data_dir / name for name in manifest["files"])
    if len(paths) != args.min_games:
        raise ValueError("dataset not complete; do not train partial corpus")
    for p in paths:
        assert digest(p) == manifest["files"][p.name]["npz_sha256"]
    train_paths, val_paths = grouped_replay_split(paths, args.fold)
    train = GoalDataset(train_paths, frame_stride=config["il"]["frame_stride"])
    validation = GoalDataset(val_paths, frame_stride=config["il"]["frame_stride"])
    if not len(train) or not len(validation):
        raise ValueError("insufficient strategic-label coverage")
    sampler = ShardFrameSampler(train, args.seed)
    kwargs = dict(
        batch_size=config["il"]["batch_frames"],
        collate_fn=collate_goal,
        pin_memory=True,
        num_workers=0,
    )
    loader = DataLoader(train, sampler=sampler, **kwargs)
    vloader = DataLoader(validation, **kwargs)
    device = torch.device("cuda")
    if torch.cuda.mem_get_info()[0] < 2 * 1024**3:
        raise RuntimeError("insufficient GPU free memory")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("use a fresh experiment directory")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    from ml_workspace.resource_guard import cuda_memory_guard, cuda_memory_stats
    from ml_workspace.tracking import tracked_run
    from dataclasses import asdict
    import mlflow

    started = time.perf_counter()
    with tracked_run(
        args.output_dir.name,
        experiment_name="lux-s3-strategic-baseline",
        params={
            "seed": args.seed,
            "epochs": args.epochs,
            "frame_stride": config["il"]["frame_stride"],
            "dataset_version": "dataset_v2_goal",
            "fold": args.fold,
        },
    ) as run:
        model = StrategicActor().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["il"]["lr"],
            weight_decay=config["il"]["weight_decay"],
        )
        metadata = dict(
            provenance=provenance(),
            dataset_manifest_sha256=digest(args.data_dir / "manifest.json"),
            train_files=[p.name for p in train_paths],
            validation_files=[p.name for p in val_paths],
            parameters=sum(p.numel() for p in model.parameters()),
            train_sampled_units=train.valid_units,
            validation_sampled_units=validation.valid_units,
            mode_labels="heuristic confidence-filtered event evidence; not human truth",
            previous_goal_input="causal scripted context, never future expert segment targets",
            optimizer="Adam",
            seed=args.seed,
            mlflow_run_id=run.info.run_id,
        )
        save_json(args.output_dir / "manifest.json", metadata)
        with cuda_memory_guard("strategic-il-preflight"):
            b = on_device(next(iter(loader)), device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ml, gl = model(*actor_input(b))
            loss, _ = supervised_loss(ml, gl, b)
            loss.backward()
        optimizer.zero_grad(set_to_none=True)
        del b, ml, gl, loss
        best, stale, best_epoch = float("inf"), 0, 0
        rows = []
        for epoch in range(1, args.epochs + 1):
            sampler.set_epoch(epoch)
            model.train()
            train_loss, total = 0.0, 0
            for cpu_batch in loader:
                b = on_device(cpu_batch, device)
                if config["il"]["previous_context_dropout"]:
                    drop = (
                        torch.rand(len(b["units"]), device=device)
                        < config["il"]["previous_context_dropout"]
                    )
                    b["units"][drop, :, 8:17] = 0
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ml, gl = model(*actor_input(b))
                loss, _ = supervised_loss(ml, gl, b)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite strategic IL loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                n = int(b["valid"].sum())
                train_loss += loss.item() * n
                total += n
            metrics = evaluate(model, vloader, device)
            row = dict(
                epoch=epoch,
                train_loss=train_loss / max(total, 1),
                **{
                    f"val_{k}": v
                    for k, v in metrics.items()
                    if isinstance(v, (int, float))
                },
                elapsed_seconds=time.perf_counter() - started,
            )
            rows.append(row)
            with (args.output_dir / "train_log.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            save_json(args.output_dir / "validation_log.json", metrics)
            mlflow.log_metrics(
                {k: v for k, v in row.items() if k != "epoch"}, step=epoch
            )
            print(json.dumps(row), flush=True)
            if metrics["loss"] < best:
                best, stale, best_epoch = metrics["loss"], 0, epoch
                torch.save(
                    dict(
                        actor_state_dict=model.state_dict(),
                        config=config,
                        epoch=epoch,
                        feature_version="causal_grouped_v2",
                        metrics=metrics,
                    ),
                    args.output_dir / "best_model.pth",
                )
            else:
                stale += 1
            if stale >= config["il"]["patience"]:
                break
        checkpoint = torch.load(
            args.output_dir / "best_model.pth", weights_only=False, map_location=device
        )
        model.load_state_dict(checkpoint["actor_state_dict"])
        metrics = evaluate(model, vloader, device)
        # A classifier predicting the training-majority mode is a required control.
        tc, vc = np.zeros(5, np.int64), np.zeros(5, np.int64)
        for data, counts in ((train, tc), (validation, vc)):
            for shard, p in enumerate(data.paths):
                with np.load(p, allow_pickle=False) as d:
                    ids = data.frame_ids[shard]
                    valid = d["label_valid"][ids]
                    counts += np.bincount(d["mode"][ids][valid], minlength=5)
        summary = dict(
            best_epoch=best_epoch,
            stopped_epoch=rows[-1]["epoch"],
            metrics=metrics,
            majority_mode=int(tc.argmax()),
            majority_mode_accuracy=float(vc[tc.argmax()] / vc.sum()),
            train_mode_counts=tc.tolist(),
            validation_mode_counts=vc.tolist(),
            train_games=len(train_paths),
            validation_games=len(val_paths),
            wall_seconds=time.perf_counter() - started,
            cuda_memory=asdict(cuda_memory_stats()),
            validation_sampling="every fourth eligible frame, same replay-level holdout",
            mlflow_run_id=run.info.run_id,
        )
        save_json(args.output_dir / "summary.json", summary)
        mlflow.log_artifacts(str(args.output_dir))
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
