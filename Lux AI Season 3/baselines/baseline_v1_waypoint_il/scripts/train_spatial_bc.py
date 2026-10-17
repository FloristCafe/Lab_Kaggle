"""Full grouped causal-history normalized Gaussian KL training; raw logits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import mlflow
import numpy as np
import sklearn
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.v3.dataset import (
    CAUSAL_ANCHOR_CHANNEL,
    CAUSAL_CURRENT_CHANNELS,
    HISTORY_CHANNELS,
    ShardFrameSampler,
    UnitFrameDataset,
    collate_spatial_frames,
    gaussian_heatmaps,
    normalize_spatial_targets,
)
from agents.v3.losses import (
    causal_destination_mask,
    masked_spatial_kl,
    spatial_kl_per_unit,
)
from agents.v3.network import Network
from scripts.replay_parser import BASE_CHANNEL_NAMES
from scripts.train_bc import set_seed
from scripts.train_full_bc import grouped_replay_split


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_manifest_replays(data_dir, required_games=400):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest.get("schema_version") != 3 or manifest.get("intent_horizon") != 12:
        raise ValueError("expected compact schema 3 and 12-step intent horizon")
    paths = sorted({data_dir / Path(row["output"]).name for row in manifest["replays"]})
    games = {path.stem.rsplit("_p", 1)[0] for path in paths}
    if len(games) != required_games:
        raise ValueError(f"require exactly {required_games} manifest games, found {len(games)}")
    expected = {name: digest for entry in manifest.get("episodes", {}).values()
                for name, digest in entry.get("output_sha256", {}).items()}
    for path in paths:
        if not path.is_file() or (path.name in expected and sha256(path) != expected[path.name]):
            raise ValueError(f"missing or altered manifest shard: {path}")
    return paths


def validate_reference_split(paths, train_paths, val_paths, reference):
    hashes = {path.name: sha256(path) for path in paths}
    if hashes != reference["npz_sha256"]:
        raise ValueError("dataset files differ from the archived 400-game reference")
    if ([p.name for p in train_paths] != reference["train_files"]
            or [p.name for p in val_paths] != reference["validation_files"]):
        raise ValueError("group split differs from the archived reference")
    train_games = {p.stem.rsplit("_p", 1)[0] for p in train_paths}
    val_games = {p.stem.rsplit("_p", 1)[0] for p in val_paths}
    if len(train_games) != 320 or len(val_games) != 80 or train_games & val_games:
        raise ValueError("require disjoint 320/80 replay groups")
    return hashes


def evaluate_distributions(predict_logits, loader, device, unit_batch=32):
    """Raw KL and spatial metrics; physical mask is decoding-only diagnostic."""
    names = ("kl", "soft_ce", "target_entropy", "accuracy", "mean_manhattan",
             "mean_euclidean", "within_1", "within_2", "within_3", "anchor_prediction_rate",
             "stay_correct", "stay_units", "moving_correct", "moving_units", "masked_accuracy",
             "masked_moving_correct", "masked_mean_manhattan", "mean_peak_probability",
             "prediction_entropy", "valid_units")
    totals = torch.zeros(len(names), device=device, dtype=torch.float64)
    with torch.no_grad():
        for features, target, valid in loader:
            for start in range(0, len(target), unit_batch):
                x = features[start:start + unit_batch].to(device, non_blocking=True)
                y = target[start:start + unit_batch].to(device, non_blocking=True)
                v = valid[start:start + unit_batch].to(device, non_blocking=True)
                x, y = x[v], y[v]
                if not len(y):
                    continue
                logits = predict_logits(x).float()
                if logits.shape != (len(y), 1, 24, 24) or not torch.isfinite(logits).all():
                    raise ValueError("nonfinite or incorrectly shaped logits")
                flat_y = y.flatten(1)
                log_probability = torch.nn.functional.log_softmax(logits.flatten(1), dim=1)
                probability = log_probability.exp()
                labels = flat_y.argmax(1)  # Evaluation only, never hard-label optimization.
                pred = logits.flatten(1).argmax(1)
                anchor = x[:, CAUSAL_ANCHOR_CHANNEL].flatten(1).argmax(1)
                allowed = causal_destination_mask(x)
                if not allowed.flatten(1).gather(1, labels[:, None]).all():
                    raise ValueError("decoding mask excludes a supervised center")
                masked_pred = logits.masked_fill(~allowed, -1e9).flatten(1).argmax(1)
                dx, dy = (pred // 24 - labels // 24).abs(), (pred % 24 - labels % 24).abs()
                distance = dx + dy
                masked_distance = ((masked_pred // 24 - labels // 24).abs()
                                   + (masked_pred % 24 - labels % 24).abs())
                correct, stay = pred == labels, labels == anchor
                entropy = -torch.special.xlogy(flat_y, flat_y).sum(1)
                values = (
                    spatial_kl_per_unit(logits, y).sum(), -(flat_y * log_probability).sum(),
                    entropy.sum(), correct.sum(), distance.sum(),
                    (dx.float().square() + dy.float().square()).sqrt().sum(),
                    (distance <= 1).sum(), (distance <= 2).sum(), (distance <= 3).sum(),
                    (pred == anchor).sum(), (correct & stay).sum(), stay.sum(),
                    (correct & ~stay).sum(), (~stay).sum(), (masked_pred == labels).sum(),
                    ((masked_pred == labels) & ~stay).sum(), masked_distance.sum(),
                    probability.max(1).values.sum(), -(probability * log_probability).sum(),
                    logits.new_tensor(len(y)),
                )
                totals += torch.stack(values).double()
    values = dict(zip(names, totals.cpu().tolist(), strict=True))
    n = int(values["valid_units"])
    if not n or not np.isfinite(list(values.values())).all():
        raise ValueError("empty validation or nonfinite distribution metrics")
    result = {key: value / n for key, value in values.items()
              if key not in ("stay_correct", "stay_units", "moving_correct", "moving_units",
                             "masked_moving_correct", "valid_units")}
    result.update(valid_units=n, stay_units=int(values["stay_units"]),
                  moving_units=int(values["moving_units"]),
                  stay_accuracy=values["stay_correct"] / max(1, values["stay_units"]),
                  moving_accuracy=values["moving_correct"] / max(1, values["moving_units"]),
                  masked_moving_accuracy=values["masked_moving_correct"] / max(1, values["moving_units"]))
    return result


def train_epoch(model, loader, optimizer, device, unit_batch):
    model.train()
    total_loss, units, updates = 0.0, 0, 0
    for features, target, valid in loader:
        optimizer.zero_grad(set_to_none=True)
        frame_units = int(valid.sum())
        for start in range(0, len(target), unit_batch):
            x = features[start:start + unit_batch].to(device, non_blocking=True)
            y = target[start:start + unit_batch].to(device, non_blocking=True)
            v = valid[start:start + unit_batch].to(device, non_blocking=True)
            count = int(v.sum())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = masked_spatial_kl(logits, y, v)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite KL")
            (loss * (count / max(1, frame_units))).backward()
            total_loss += loss.detach().item() * count
            units += count
        optimizer.step()
        updates += 1
    return {"train_kl": total_loss / max(1, units), "train_valid_units": units,
            "optimizer_updates": updates}


def plot_curves(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    epochs = [row["epoch"] for row in rows]
    axes[0].plot(epochs, [row["train_kl"] for row in rows], label="train")
    axes[0].plot(epochs, [row["val_kl"] for row in rows], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Per-unit KL(target || prediction)", yscale="log")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_mean_manhattan"] for row in rows])
    axes[1].set(xlabel="Epoch", ylabel="Raw top-1 Manhattan error")
    axes[2].plot(epochs, [row["val_accuracy"] for row in rows], label="exact")
    axes[2].plot(epochs, [row["val_within_3"] for row in rows], label="distance <=3")
    axes[2].plot(epochs, [row["val_anchor_prediction_rate"] for row in rows], label="anchor prediction")
    axes[2].set(xlabel="Epoch", ylabel="Fraction", ylim=(0, 1))
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output / "validation_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed_replays_unitcentric25")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/bc_causal_gaussian_kl400_fold0_seed42")
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--frames-per-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--min-games", type=int, default=400)
    parser.add_argument("--reference-manifest", type=Path,
                        default=ROOT / "outputs/bc_unitcentric_400_fold0_seed42/manifest.json")
    parser.add_argument("--preflight-only", action="store_true",
                        help="data/split and forward/backward checks only; no optimizer updates")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    if (not 1.5 <= args.sigma <= 2 or not 1 <= args.max_epochs <= 60
            or args.patience != 5 or not 1 <= args.batch_size <= 32 or args.frames_per_batch < 1
            or args.min_games != 400 or not np.isfinite(args.lr) or args.lr <= 0
            or args.weight_decay != 1e-4):
        parser.error("require 400 games, sigma 1.5..2, <=60 epochs, patience=5, <=32 units, weight_decay=1e-4")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be new/empty")
    paths = select_manifest_replays(args.data_dir, args.min_games)
    train_paths, val_paths = grouped_replay_split(paths, args.fold)
    reference = json.loads(args.reference_manifest.read_text())
    dataset_hashes = validate_reference_split(paths, train_paths, val_paths, reference)
    train_data, val_data = UnitFrameDataset(train_paths, args.sigma), UnitFrameDataset(val_paths, args.sigma)
    set_seed(args.seed)
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")
    if torch.cuda.mem_get_info()[0] < 2 * 1024**3:
        raise RuntimeError("require at least 2 GiB CUDA free")
    device = torch.device("cuda")
    sampler = ShardFrameSampler(train_data, args.seed)
    train_loader = DataLoader(train_data, batch_size=args.frames_per_batch, sampler=sampler,
                              collate_fn=collate_spatial_frames, num_workers=0, pin_memory=True)
    mean_loader = DataLoader(train_data, batch_size=16, collate_fn=collate_spatial_frames, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.frames_per_batch,
                            collate_fn=collate_spatial_frames, num_workers=0, pin_memory=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:////home/issue/ml-workspace/artifacts/mlflow.db")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    from ml_workspace.resource_guard import cuda_memory_guard, cuda_memory_stats
    from ml_workspace.tracking import tracked_run

    def git(*parts, cwd=ROOT):
        return subprocess.check_output(["git", *parts], cwd=cwd, text=True).strip()

    if git("rev-parse", "HEAD", cwd=ROOT / "vendor/lux-design-s3") != "c6d1e665d12a467b299fb586b3f876e47847746b":
        raise RuntimeError("official environment commit differs")
    metadata = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "selected_files": [p.name for p in paths], "selection": "all 400 manifest Match IDs; archived split and file hashes locked",
        "train_files": [p.name for p in train_paths], "validation_files": [p.name for p in val_paths],
        "train_frames": train_data.total_frames, "validation_frames": val_data.total_frames,
        "train_valid_units": train_data.valid_units, "validation_valid_units": val_data.valid_units,
        "train_original_valid_units": train_data.original_valid_units,
        "validation_original_valid_units": val_data.original_valid_units,
        "npz_sha256": dataset_hashes,
        "reference_manifest_sha256": sha256(args.reference_manifest),
        "train_games": sorted({p.stem.rsplit("_p", 1)[0] for p in train_paths}),
        "validation_games": sorted({p.stem.rsplit("_p", 1)[0] for p in val_paths}),
        "data_manifest_sha256": sha256(args.data_dir / "manifest.json"),
        "source_sha256": {name: sha256(ROOT / name) for name in (
            "agents/v3/dataset.py", "agents/v3/network.py", "agents/v3/losses.py",
            "scripts/train_spatial_bc.py", "scripts/replay_parser.py",
            "scripts/train_bc.py", "scripts/train_full_bc.py")},
        "git_revision": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
        "runtime": {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                    "capability": torch.cuda.get_device_capability(), "numpy": np.__version__,
                    "scikit_learn": sklearn.__version__, "mlflow": mlflow.__version__},
        "current_planes": [BASE_CHANNEL_NAMES[i] for i in CAUSAL_CURRENT_CHANNELS],
        "past_planes_each": [BASE_CHANNEL_NAMES[i] for i in HISTORY_CHANNELS],
        "excluded_source_channels": [6, 7, 8, 21, 22, 23, 24],
        "feature_cutoff": "T and contiguous same-match T-1/T-2 only; no backcast source planes",
        "label_cutoff": "future-position mode through T+12; heuristic, label-only lifecycle exclusions",
        "objective": f"raw logits + normalized Gaussian sigma {args.sigma} + valid-unit KLDivLoss",
        "early_stopping_monitor": "raw validation per-unit KL",
        "target_normalization": "Y / sum_576(Y), per unit; no soft targets on disk",
        "spatial_mask": "decoding diagnostic only; neither logits nor Gaussian tails masked in KL",
        "deterministic": True, "learning_rate": args.lr, "weight_decay": args.weight_decay,
        "memory_budget": "32 unit BF16 microbatch, one replay cache per split; estimated total <2GiB",
    }
    started = time.perf_counter()
    with tracked_run(args.output_dir.name, experiment_name="lux-s3-causal-gaussian-kl", params=metadata["config"]) as run:
        metadata["mlflow_run_id"] = run.info.run_id
        metadata["mlflow_tracking_uri"] = mlflow.get_tracking_uri()
        mlflow.set_tags({"git.commit": metadata["git_revision"], "git.dirty": "True",
                         "feature_boundary": "causal T,T-1,T-2; archived backcast planes excluded",
                         "label_type": "future heuristic, not expert intention"})
        model = Network().to(device)
        metadata["parameters"] = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        with cuda_memory_guard("causal-gaussian-kl-synthetic"):
            probe_x = torch.zeros(args.batch_size, 35, 24, 24, device=device)
            probe_y = normalize_spatial_targets(gaussian_heatmaps(torch.tensor([[12, 12]] * args.batch_size), args.sigma)).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                probe_logits = model(probe_x)
            probe_loss = masked_spatial_kl(probe_logits, probe_y, torch.ones(args.batch_size, dtype=torch.bool, device=device))
            probe_loss.backward()
        optimizer.zero_grad(set_to_none=True)
        del probe_x, probe_y, probe_logits, probe_loss
        metadata["synthetic_memory"] = asdict(cuda_memory_stats())
        (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        # One real-data batch, backwards only: validate probabilities and gradients.
        with cuda_memory_guard("causal-gaussian-kl-real-preflight"):
            real_x, real_y, real_v = next(iter(mean_loader))
            real_x = real_x[:args.batch_size].to(device)
            real_y = real_y[:args.batch_size].to(device)
            real_v = real_v[:args.batch_size].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                real_logits = model(real_x)
            real_loss = masked_spatial_kl(real_logits, real_y, real_v)
            real_loss.backward()
            if (not torch.isfinite(real_loss)
                    or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)):
                raise RuntimeError("preflight nonfinite KL/gradients")
            preflight = {"optimizer_updates": 0, "real_units": int(real_v.sum()),
                         "real_kl": real_loss.item(), "target_sum_min": real_y.flatten(1).sum(1).min().item(),
                         "target_sum_max": real_y.flatten(1).sum(1).max().item(),
                         "train_games": len(metadata["train_games"]),
                         "validation_games": len(metadata["validation_games"])}
        optimizer.zero_grad(set_to_none=True)
        del real_x, real_y, real_v, real_logits, real_loss
        preflight["cuda_memory"] = asdict(cuda_memory_stats())
        (args.output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
        if args.preflight_only:
            mlflow.log_artifacts(str(args.output_dir))
            print(json.dumps({"run_id": run.info.run_id, "preflight": preflight}), flush=True)
            return
        train_mean, count = torch.zeros(24, 24, dtype=torch.float64), 0
        for _, y, v in mean_loader:
            train_mean += y[v].double().sum(0)
            count += int(v.sum())
        train_mean = normalize_spatial_targets((train_mean / count).float()[None])[0].to(device)
        torch.save(train_mean.cpu(), args.output_dir / "train_mean_heatmap.pth")

        def prediction(x):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model(x)

        def anchor_prediction(x):
            anchor = x[:, CAUSAL_ANCHOR_CHANNEL].flatten(1).argmax(1)
            xy = torch.stack((anchor // 24, anchor % 24), 1)
            axis = torch.arange(24, device=x.device, dtype=torch.float32)
            gx, gy = torch.meshgrid(axis, axis, indexing="ij")
            distance2 = ((gx[None] - xy[:, 0, None, None]).square()
                         + (gy[None] - xy[:, 1, None, None]).square())
            return (-distance2 / (2 * args.sigma**2))[:, None]

        baselines = {
            "uniform_probability": evaluate_distributions(lambda x: x.new_zeros((len(x), 1, 24, 24)), val_loader, device, args.batch_size),
            "train_mean_probability": evaluate_distributions(lambda x: train_mean.clamp_min(torch.finfo(torch.float32).tiny).log()[None, None].expand(len(x), 1, 24, 24), val_loader, device, args.batch_size),
            "anchor_gaussian_probability": evaluate_distributions(anchor_prediction, val_loader, device, args.batch_size),
        }
        (args.output_dir / "baselines.json").write_text(json.dumps(baselines, indent=2) + "\n")
        print(json.dumps({"run_id": run.info.run_id, "train_files": metadata["train_files"],
                          "validation_files": metadata["validation_files"], "baselines": baselines}), flush=True)
        rows, best, stale, best_epoch = [], float("inf"), 0, 0
        with cuda_memory_guard("causal-gaussian-kl-training"):
            for epoch in range(1, args.max_epochs + 1):
                sampler.set_epoch(epoch)
                train_metrics = train_epoch(model, train_loader, optimizer, device, args.batch_size)
                model.eval()
                val = evaluate_distributions(prediction, val_loader, device, args.batch_size)
                row = {"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val.items()},
                       "elapsed_seconds": time.perf_counter() - started}
                rows.append(row)
                with (args.output_dir / "metrics.csv").open("w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(rows)
                mlflow.log_metrics({k: float(v) for k, v in row.items() if k != "epoch"}, step=epoch)
                print(json.dumps(row), flush=True)
                if val["kl"] < best:
                    best, stale, best_epoch = val["kl"], 0, epoch
                    torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                                "sigma": args.sigma, "input_channels": 35,
                                "objective": "normalized_gaussian_kl", "metrics": val},
                               args.output_dir / "best_model.pth")
                else:
                    stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(torch.load(args.output_dir / "best_model.pth", map_location=device, weights_only=True)["model_state_dict"])
        model.eval()
        best_metrics = evaluate_distributions(prediction, val_loader, device, args.batch_size)
        summary = {"best_epoch": best_epoch, "stopped_epoch": rows[-1]["epoch"],
                   "early_stopped": stale >= args.patience, "best_val_kl": best_metrics["kl"],
                   "best_validation": best_metrics,
                   "baselines": baselines, "wall_seconds": time.perf_counter() - started,
                   "rss_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   "cuda_memory": asdict(cuda_memory_stats()), "mlflow_run_id": run.info.run_id}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        plot_curves(rows, args.output_dir)
        mlflow.log_metrics({f"best_{k}": float(v) for k, v in best_metrics.items()})
        mlflow.log_artifacts(str(args.output_dir))
        print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
