"""Train compact unit-centric behavior cloning with replay-group holdout validation."""

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
import torch
from sklearn.model_selection import GroupKFold
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.v3.dataset import (
    LegacyProjectionFrameDataset as UnitFrameDataset,
)
from agents.v3.dataset import (
    ShardFrameSampler,
    collate_unit_frames,
    coordinate_planes,
)
from agents.v3.losses import apply_destination_mask, focal_spatial_loss
from agents.v3.mask_audit import audit_physical_masks
from agents.v3.residual_network import Network as ResidualNetwork
from agents.v3.unet28_network import Network
from scripts.replay_parser import SCHEMA_VERSION
from scripts.train_bc import set_seed


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grouped_replay_split(paths, fold=0):
    """Keep both player views of each source replay in the same fold."""
    paths = sorted(Path(path) for path in paths)
    groups = [path.stem.rsplit("_p", 1)[0] for path in paths]
    if len(set(groups)) < 5:
        raise ValueError("need at least five replay groups")
    if any(groups.count(group) not in (1, 2) for group in set(groups)):
        raise ValueError("each replay must have one or two unique player views")
    folds = list(GroupKFold(n_splits=5).split(np.arange(len(paths)), groups=groups))
    if not 0 <= fold < len(folds):
        raise ValueError("fold must be in [0, 4]")
    train_indices, val_indices = folds[fold]
    train_paths = [paths[int(index)] for index in train_indices]
    val_paths = [paths[int(index)] for index in val_indices]
    train_groups = {groups[int(index)] for index in train_indices}
    val_groups = {groups[int(index)] for index in val_indices}
    if not train_groups or not val_groups or train_groups & val_groups:
        raise AssertionError("replay groups overlap")
    return train_paths, val_paths


def drop_future_features(features, probability):
    """Compatibility helper; training dropout is applied per frame in Dataset."""
    if features.ndim != 4 or features.shape[1] not in (25, 26, 28):
        raise ValueError("expected 25, 26 or 28 input channels")
    if not 0 <= probability <= 1:
        raise ValueError("dropout probability must be in [0, 1]")
    dropped = torch.rand(features.shape[0], device=features.device) < probability
    if dropped.any():
        features = features.clone()
        features[dropped, 21:25] = 0
    return features, int(dropped.sum())


def unit_intent_loss(logits, target):
    if logits.ndim != 4 or logits.shape[1:] != (1, 24, 24):
        raise ValueError("expected unit logits [K,1,24,24]")
    if target.shape != (len(logits),) or target.dtype != torch.long:
        raise ValueError("expected K spatial class IDs")
    return F.cross_entropy(logits.float().flatten(1), target)


class FrozenResidualReference(torch.nn.Module):
    """Replay the archived 26-channel checkpoint on the effective label subset."""

    def __init__(self, checkpoint):
        super().__init__()
        self.model = ResidualNetwork()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.requires_grad_(False)

    def forward(self, features):
        return self.model(features[:, :26])


def evaluate_model(model, loader, device, amp_enabled, zero_future=False,
                   unit_batch=32, focal_gamma=2.0):
    model.eval()
    totals = torch.zeros(13, dtype=torch.float64, device=device)
    with torch.no_grad():
        for features, target, _, _ in loader:
            for start in range(0, len(target), unit_batch):
                x = features[start:start + unit_batch].to(device, non_blocking=True)
                y = target[start:start + unit_batch].to(device, non_blocking=True)
                if zero_future:
                    x = x.clone()
                    x[:, 21:25] = 0
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=amp_enabled):
                    raw = model(x)
                logits, allowed = apply_destination_mask(raw, x, y)
                ce = unit_intent_loss(logits, y)
                focal = focal_spatial_loss(logits, y, focal_gamma)
                raw_ce = unit_intent_loss(raw, y)
                pred, raw_pred = logits.flatten(1).argmax(1), raw.flatten(1).argmax(1)
                anchor = x[:, 25].flatten(1).argmax(1)
                stay = y == anchor
                correct = pred == y
                counts = torch.stack((correct.sum(), (raw_pred == y).sum(),
                                      (pred == anchor).sum(), (raw_pred == anchor).sum(),
                                      (correct & stay).sum(), stay.sum(),
                                      (correct & ~stay).sum(), (~stay).sum(),
                                      allowed.sum(), y.new_tensor(len(y)))).double()
                totals[:3] += torch.stack((ce, focal, raw_ce)).double() * len(y)
                totals[3:] += counts
    values = totals.cpu().tolist()
    units = int(values[12])
    if not units or not np.isfinite(values).all():
        raise RuntimeError("empty validation or nonfinite metrics")
    return {"loss": values[0] / units, "focal_loss": values[1] / units,
            "raw_loss": values[2] / units, "accuracy": values[3] / units,
            "raw_accuracy": values[4] / units,
            "anchor_prediction_rate": values[5] / units,
            "raw_anchor_prediction_rate": values[6] / units,
            "stay_accuracy": values[7] / values[8] if values[8] else 0.0,
            "stay_units": int(values[8]),
            "moving_accuracy": values[9] / values[10] if values[10] else 0.0,
            "moving_units": int(values[10]),
            "allowed_cell_fraction": values[11] / (units * 576), "valid_units": units}


def score_model(model, loader, device, amp_enabled, zero_future=False, unit_batch=32):
    metrics = evaluate_model(model, loader, device, amp_enabled, zero_future, unit_batch)
    return metrics["loss"], metrics["accuracy"], metrics["valid_units"]


def train_epoch(model, loader, optimizer, device, amp_enabled, unit_batch=32, focal_gamma=2.0):
    model.train()
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    dropped_sum = frame_sum = updates = 0
    for features, target, dropped, frames in loader:
        dropped_sum += dropped
        frame_sum += frames
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(target), unit_batch):
            x = features[start:start + unit_batch].to(device, non_blocking=True)
            y = target[start:start + unit_batch].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=amp_enabled):
                raw = model(x)
            logits, _ = apply_destination_mask(raw, x, y)
            loss = focal_spatial_loss(logits, y, focal_gamma)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite focal training loss")
            (loss * (len(y) / len(target))).backward()
            with torch.no_grad():
                ce = unit_intent_loss(logits, y)
                pred = logits.flatten(1).argmax(1)
                anchor = x[:, 25].flatten(1).argmax(1)
                totals[:2] += torch.stack((loss.detach(), ce)).double() * len(y)
                totals[2:] += torch.stack(((pred == y).sum(), (pred == anchor).sum(),
                                          y.new_tensor(len(y)))).double()
        optimizer.step()
        updates += 1
    values = totals.cpu().tolist()
    if not values[4] or not np.isfinite(values).all():
        raise RuntimeError("empty training or nonfinite metrics")
    return {"train_loss": values[0] / values[4], "train_ce": values[1] / values[4],
            "train_accuracy": values[2] / values[4],
            "train_anchor_prediction_rate": values[3] / values[4],
            "train_valid_units": int(values[4]), "dropout_fraction": dropped_sum / frame_sum,
            "optimizer_updates": updates}


def plot_curves(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_ce"] for row in rows], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in rows], label="val with projections")
    axes[0].plot(epochs, [row["val_fallback_loss"] for row in rows],
                 label="val without projections")
    axes[0].set(xlabel="Epoch", ylabel="Masked CE", yscale="log")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in rows], label="train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in rows], label="val with projections")
    axes[1].plot(epochs, [row["val_fallback_accuracy"] for row in rows],
                 label="val without projections")
    axes[1].set(xlabel="Epoch", ylabel="Valid-unit accuracy", ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "data/processed_replays_unitcentric25")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/bc_unet_coord_focal400_fold0_seed42")
    parser.add_argument("--reference-manifest", type=Path,
                        default=ROOT / "outputs/bc_unitcentric_400_fold0_seed42/manifest.json",
                        help="lock the exact previous train/validation split and NPZ hashes")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--preflight-only", action="store_true",
                        help="tracked synthetic forward/backward only; no real-data optimizer updates")
    parser.add_argument("--mask-audit", type=Path,
                        default=ROOT / "outputs/bc_unet_mask_audit_seed42.json",
                        help="cached complete label-coverage check, invalidated on data/code changes")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="maximum units per CUDA microbatch (1..32)")
    parser.add_argument("--frames-per-batch", type=int, default=4,
                        help="frames whose unit-weighted gradients form one optimizer update")
    parser.add_argument("--min-games", type=int, default=400)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout-probability", type=float, default=0.20)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    if (args.frames_per_batch < 1 or args.min_games < 5 or not 1 <= args.batch_size <= 32 or args.max_epochs < 1 or args.patience != 5
            or not np.isfinite(args.focal_gamma) or args.focal_gamma < 0
            or args.lr <= 0 or args.weight_decay != 1e-4
            or args.dropout_probability != 0.20):
        parser.error("invalid batch, epoch, patience, learning rate, dropout, or weight decay")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory is nonempty; choose a fresh output directory")
    data_manifest_path = args.data_dir / "manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text())
    if data_manifest.get("schema_version") != SCHEMA_VERSION or len(data_manifest.get("channel_names", [])) != 25:
        parser.error("expected compact 25-channel schema 3 dataset")

    # Only committed manifest entries are eligible; interrupted partial shards
    # must not silently enter the dataset.
    paths = sorted({args.data_dir / Path(row["output"]).name
                    for row in data_manifest["replays"]})
    if any(not path.is_file() for path in paths):
        raise ValueError("manifest references missing NPZ files")
    expected_hashes = {name: expected
                       for entry in data_manifest.get("episodes", {}).values()
                       for name, expected in entry.get("output_sha256", {}).items()}
    for path in paths:
        if path.name in expected_hashes and sha256(path) != expected_hashes[path.name]:
            raise ValueError(f"{path}: NPZ changed after stream verification")
    games = {path.stem.rsplit("_p", 1)[0] for path in paths}
    if len(games) < args.min_games:
        parser.error(f"need {args.min_games} games, found {len(games)}")
    train_paths, val_paths = grouped_replay_split(paths, args.fold)
    reference = json.loads(args.reference_manifest.read_text())
    if sha256(ROOT / "agents/v3/residual_network.py") != reference["source_sha256"]["agents/v3/network.py"]:
        parser.error("archived reference model source differs from the old experiment")
    if ([p.name for p in train_paths] != reference["train_files"]
            or [p.name for p in val_paths] != reference["validation_files"]
            or args.seed != reference["config"]["seed"]):
        parser.error("split/seed differs from reference experiment")
    dataset_hashes = {p.name: sha256(p) for p in paths}
    if dataset_hashes != reference["npz_sha256"]:
        parser.error("NPZ data differs from reference experiment")
    if data_manifest.get("intent_horizon") != 12:
        parser.error("physical mask requires the fixed 12-step target horizon")
    torch.set_num_threads(4)
    mask_audit = audit_physical_masks(paths, dataset_hashes, args.mask_audit)
    train_data = UnitFrameDataset(train_paths, future_dropout=args.dropout_probability)
    val_data = UnitFrameDataset(val_paths)
    torch.set_num_threads(4)
    set_seed(args.seed)
    device = torch.device("cuda")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required for this run")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < 2 * 1024**3:
        raise RuntimeError("less than 2 GiB of CUDA memory is free")
    train_sampler = ShardFrameSampler(train_data, args.seed)
    train_loader = DataLoader(train_data, batch_size=args.frames_per_batch,
                              sampler=train_sampler, collate_fn=collate_unit_frames,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.frames_per_batch,
                            collate_fn=collate_unit_frames, num_workers=0, pin_memory=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLFLOW_TRACKING_URI",
                          "sqlite:////home/issue/ml-workspace/artifacts/mlflow.db")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment = "lux-s3-bc-unet-coord-focal"
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment,
                                 artifact_location="file:///home/issue/ml-workspace/artifacts/mlruns")
    from ml_workspace.resource_guard import cuda_memory_guard, cuda_memory_stats
    from ml_workspace.tracking import tracked_run

    git_revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           cwd=ROOT, text=True).strip()
    environment_revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                   cwd=ROOT / "vendor/lux-design-s3", text=True).strip()
    if environment_revision != "c6d1e665d12a467b299fb586b3f876e47847746b":
        raise RuntimeError("official environment commit changed; physical-mask bounds require review")
    metadata = {
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "train_files": [path.name for path in train_paths],
        "validation_files": [path.name for path in val_paths],
        "train_games": sorted({path.stem.rsplit("_p", 1)[0] for path in train_paths}),
        "validation_games": sorted({path.stem.rsplit("_p", 1)[0] for path in val_paths}),
        "train_frames": train_data.total_frames,
        "train_eligible_frames": len(train_data),
        "train_valid_units": train_data.valid_units,
        "train_original_valid_units": train_data.original_valid_units,
        "train_lifecycle_labels_removed": train_data.original_valid_units - train_data.valid_units,
        "validation_frames": val_data.total_frames,
        "validation_eligible_frames": len(val_data),
        "validation_valid_units": val_data.valid_units,
        "validation_original_valid_units": val_data.original_valid_units,
        "validation_lifecycle_labels_removed": val_data.original_valid_units - val_data.valid_units,
        "data_manifest_sha256": sha256(data_manifest_path),
        "npz_sha256": dataset_hashes,
        "reference_manifest_sha256": sha256(args.reference_manifest),
        "mask_audit": {k: v for k, v in mask_audit.items() if k != "dataset_sha256"},
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__), ROOT / "scripts/train_bc.py", ROOT / "agents/v3/network.py",
            ROOT / "agents/v3/dataset.py", ROOT / "agents/v3/losses.py", ROOT / "agents/v3/mask_audit.py",
            ROOT / "agents/v3/residual_network.py",
            ROOT / "scripts/replay_parser.py")},
        "git_revision": git_revision,
        "official_environment_commit": environment_revision,
        "git_status": subprocess.check_output(["git", "status", "--short"],
                                              cwd=ROOT, text=True),
        "runtime": {"python": sys.version, "executable": sys.executable,
                    "torch": torch.__version__, "numpy": np.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                    "compute_capability": torch.cuda.get_device_capability(),
                    "environment_prefix": sys.prefix,
                    "free_cuda_bytes_before": free_bytes,
                    "total_cuda_bytes": total_bytes},
        "data_boundary": "GroupKFold 5 splits; source episode IDs isolated; seed uniqueness unverified",
        "deterministic": True, "seed": args.seed,
        "model_shape": "28 input channels, 1 output; U-Net 24/12/6 plus global bottleneck context; GroupNorm",
        "training_objective": "multiclass focal loss, gamma=" + str(args.focal_gamma),
        "early_stopping_monitor": "masked validation CE; patience 5 (unchanged)",
        "output_mask": "optimistic 12-step reachability; current visible terrain/energy plus aggregate own energy upper bound",
        "mask_energy_bounds": {"minimum_move_cost": 1, "maximum_recovery_per_step": 20,
                               "maximum_unit_energy": 400, "unknown_nebula_drain_lower_bound": 0},
        "mask_limitations": "no individual-energy or per-match nebula-drain fields in legacy NPZ; unknown recovery/drain optimistic, no strict deep-nebula pruning",
        "feature_boundary": data_manifest["drift_speed_cutoff"],
        "label_boundary": "future-position heuristic, horizon 12; not expert macro ground truth",
        "runtime_label_filter": "exclude slot jumps, impossible singleton energy gains and match-step-zero transitions; disk targets unchanged",
        "memory_budget": "one decoded replay per split (~40 MiB each); <=32 unit BF16 microbatch; U-Net multi-scale activations, gradients/Adam and workspace budget <2 GiB, measured synthetic preflight before training",
    }
    started = time.perf_counter()
    with tracked_run(args.output_dir.name, experiment_name=experiment,
                     params=metadata["config"]) as run:
        metadata["mlflow_run_id"] = run.info.run_id
        metadata["mlflow_tracking_uri"] = mlflow.get_tracking_uri()
        (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        mlflow.set_tags({"feature_leakage": "offline backcast before drift convergence",
                         "label_type": "heuristic future-position pseudo-target"})
        with cuda_memory_guard("synthetic-smoke"):
            probe = Network().to(device)
            probe_x = torch.zeros(args.batch_size, 28, 24, 24, device=device)
            probe_x[:, 25, 3, 7] = 1
            probe_x[:, 26:28] = coordinate_planes().to(device)
            probe_x[:, 14, 3, 7] = 100 / 6400
            probe_y = torch.full((args.batch_size,), 3 * 24 + 7, dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                probe_logits = probe(probe_x)
            probe_logits, _ = apply_destination_mask(probe_logits, probe_x, probe_y)
            probe_loss = focal_spatial_loss(probe_logits, probe_y, args.focal_gamma)
            probe_loss.backward()
            if not torch.isfinite(probe_loss):
                raise RuntimeError("nonfinite synthetic smoke loss")
        probe_parameters = sum(p.numel() for p in probe.parameters())
        probe_summary = {"parameters": probe_parameters, "synthetic_focal_loss": float(probe_loss.detach()),
                         "cuda_memory": asdict(cuda_memory_stats()), "mlflow_run_id": run.info.run_id,
                         "train_games": len(metadata["train_games"]),
                         "validation_games": len(metadata["validation_games"])}
        del probe, probe_x, probe_y, probe_logits, probe_loss
        if args.preflight_only:
            (args.output_dir / "preflight.json").write_text(json.dumps(probe_summary, indent=2) + "\n")
            mlflow.log_artifacts(str(args.output_dir))
            print(json.dumps(probe_summary), flush=True)
            return
        reference_checkpoint_path = args.reference_manifest.parent / "best_model.pth"
        print("evaluating frozen reference on the same effective validation labels", flush=True)
        reference_checkpoint = torch.load(reference_checkpoint_path, map_location="cpu", weights_only=False)
        reference_model = FrozenResidualReference(reference_checkpoint).to(device)
        reference_metrics = evaluate_model(reference_model, val_loader, device, True,
                                           unit_batch=args.batch_size, focal_gamma=args.focal_gamma)
        hold_correct = sum(mask_audit["file_stats"][p.name]["hold_correct"] for p in val_paths)
        baselines = {"reference_checkpoint_sha256": sha256(reference_checkpoint_path),
                     "reference_run_id": reference.get("mlflow_run_id"),
                     "frozen_reference": reference_metrics,
                     "hold_accuracy": hold_correct / val_data.valid_units,
                     "validation_valid_units": val_data.valid_units}
        (args.output_dir / "baselines.json").write_text(json.dumps(baselines, indent=2) + "\n")
        metadata["baselines"] = baselines
        mlflow.log_metrics({"reference_raw_accuracy": reference_metrics["raw_accuracy"],
                            "reference_masked_accuracy": reference_metrics["accuracy"],
                            "hold_accuracy": baselines["hold_accuracy"]}, step=0)
        print(json.dumps({"reference_baselines": baselines}), flush=True)
        del reference_model, reference_checkpoint
        set_seed(args.seed)
        model = Network().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        metadata["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        best_loss = float("inf")
        best_epoch = None
        stale_epochs = 0
        rows = []
        with cuda_memory_guard("full-bc-training"):
            for epoch in range(1, args.max_epochs + 1):
                train_sampler.set_epoch(epoch)
                train_metrics = train_epoch(model, train_loader, optimizer, device,
                                            True, args.batch_size, args.focal_gamma)
                normal = evaluate_model(model, val_loader, device, True,
                                        unit_batch=args.batch_size, focal_gamma=args.focal_gamma)
                fallback = evaluate_model(model, val_loader, device, True, zero_future=True,
                                          unit_batch=args.batch_size, focal_gamma=args.focal_gamma)
                val_loss = normal["loss"]
                row = {"epoch": epoch, **train_metrics,
                       **{"val_" + k: v for k, v in normal.items()},
                       **{"val_fallback_" + k: v for k, v in fallback.items()},
                       "elapsed_seconds": time.perf_counter() - started}
                rows.append(row)
                with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                mlflow.log_metrics({key: value for key, value in row.items()
                                    if key not in ("epoch", "elapsed_seconds")}, step=epoch)
                print(json.dumps(row), flush=True)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    stale_epochs = 0
                    torch.save({"model_state_dict": model.state_dict(),
                                "epoch": epoch, "validation_loss": val_loss,
                                "metadata": metadata}, args.output_dir / "best_model.pth")
                else:
                    stale_epochs += 1
                    if stale_epochs >= args.patience:
                        break
        checkpoint = torch.load(args.output_dir / "best_model.pth",
                                map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        normal = evaluate_model(model, val_loader, device, True,
                                unit_batch=args.batch_size, focal_gamma=args.focal_gamma)
        fallback = evaluate_model(model, val_loader, device, True, zero_future=True,
                                  unit_batch=args.batch_size, focal_gamma=args.focal_gamma)
        summary = {
            "best_epoch": best_epoch, "stopped_epoch": rows[-1]["epoch"],
            "early_stopped": stale_epochs >= args.patience,
            **{"best_val_" + k: v for k, v in normal.items()},
            **{"best_val_fallback_" + k: v for k, v in fallback.items()},
            "validation_valid_units": normal["valid_units"],
            "wall_seconds": time.perf_counter() - started,
            "rss_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "cuda_memory": asdict(cuda_memory_stats()), "mlflow_run_id": run.info.run_id,
            "baselines": baselines,
        }
        with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plot_curves(rows, args.output_dir / "validation_curves.png")
        (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_metrics({key: value for key, value in summary.items()
                            if key.startswith("best_val_")})
        mlflow.log_artifacts(str(args.output_dir))
        print(json.dumps({"run_id": run.info.run_id, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
