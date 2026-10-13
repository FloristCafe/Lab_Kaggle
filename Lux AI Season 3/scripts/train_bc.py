"""Memorize one fixed replay batch. This is not a generalization experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.v3.dataset import LegacyProjectionFrameDataset as UnitFrameDataset
from agents.v3.dataset import coordinate_planes
from agents.v3.losses import apply_destination_mask, focal_spatial_loss
from agents.v3.unet28_network import Network


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def masked_intent_loss(logits, target, valid):
    """CE over x*height+y, averaged over valid units (not frames).

    Invalid slots are neutralized before CE, so even invalid NaNs cannot
    contaminate the objective. This equals masked, unreduced CE for finite inputs.
    """
    if logits.shape != target.shape or logits.ndim != 4:
        raise ValueError("logits and one-hot target must have equal [B, K, X, Y] shapes")
    if valid.shape != logits.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("valid must be bool [B, K]")
    if not valid.any():
        return logits[valid].sum()  # Differentiable zero, even with invalid NaNs.
    flat = logits.float().flatten(2)
    indices = target.flatten(2).argmax(dim=-1)
    flat = torch.where(valid[..., None], flat, torch.zeros_like(flat))
    indices = torch.where(valid, indices, torch.zeros_like(indices))
    raw = F.cross_entropy(flat.reshape(-1, flat.shape[-1]),
                          indices.reshape(-1), reduction="none").reshape(valid.shape)
    return (raw * valid.float()).sum() / valid.sum()


class ReplayFrameDataset(Dataset):
    """Seeded fixed frame sample, expanding valid units with temporary anchors."""

    def __init__(self, data_dir, size=32, seed=42):
        self.data_dir = Path(data_dir)
        paths = sorted(self.data_dir.glob("*.npz"))
        dataset = UnitFrameDataset(paths)
        self.total_frames = dataset.total_frames
        self.eligible_frames = len(dataset)
        if not 1 <= size <= len(dataset):
            raise ValueError(f"size must be in [1, {len(dataset)}]")
        selected = np.random.default_rng(seed).choice(len(dataset), size, replace=False)
        self.records, self.rows = [], []
        self.source_hashes = {path.name: sha256(path) for path in paths}
        for index in selected:
            x, labels, _ = dataset[int(index)]
            y = torch.zeros((len(labels), 1, 24, 24), dtype=torch.uint8)
            y.flatten(2)[:, 0].scatter_(1, labels[:, None], 1)
            self.rows.append((x, y, torch.ones((len(labels), 1), dtype=torch.bool)))
            shard = int(np.searchsorted(dataset.offsets, index, side="right") - 1)
            frame = int(dataset.frame_ids[shard][index - dataset.offsets[shard]])
            _, _, valid, _, _ = dataset._load(shard)
            with np.load(paths[shard], allow_pickle=False) as data:
                self.records.append({"npz": paths[shard].name, "frame": frame,
                                     "step": int(data["steps"][frame]),
                                     "match_step": int(data["match_steps"][frame]),
                                     "valid_units": np.flatnonzero(valid[frame]).tolist()})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_query_frames(rows):
    return tuple(torch.cat([row[i] for row in rows]) for i in range(3))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def batch_audit(features, target, valid):
    duplicate_pairs, conflicting_labels = 0, 0
    for left in range(len(features)):
        for right in range(left + 1, len(features)):
            if torch.equal(features[left], features[right]):
                duplicate_pairs += 1
                shared = valid[left] & valid[right]
                conflicting_labels += int(((target[left].flatten(1).argmax(1) !=
                                           target[right].flatten(1).argmax(1)) & shared).sum())
    return {
        "valid_labels": int(valid.sum()),
        "valid_labels_per_unit": valid.sum(0).tolist(),
        "duplicate_feature_pairs": duplicate_pairs,
        "conflicting_duplicate_labels": conflicting_labels,
        "batch_bytes": sum(t.numel() * t.element_size() for t in (features, target, valid)),
        "unique_target_tiles": int(target[valid].flatten(1).argmax(1).unique().numel()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed_replays_unitcentric25")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/bc_single_batch")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0:
        parser.error("epochs and lr must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory is nonempty; use a fresh directory")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; pass --device cpu explicitly")
    if args.device == "cpu" and args.precision != "fp32":
        parser.error("CPU run requires --precision fp32")

    import mlflow
    from ml_workspace.resource_guard import cuda_memory_guard, cuda_memory_stats
    from ml_workspace.tracking import tracked_run

    started = time.perf_counter()
    torch.set_num_threads(4)
    set_seed(args.seed)
    dataset = ReplayFrameDataset(args.data_dir, args.batch_size, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_query_frames)
    features, target, valid = next(iter(loader))  # Loaded once, reused for every optimizer step.
    audit = batch_audit(features, target, valid)
    manifest_path = args.data_dir / "manifest.json"
    data_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    config = vars(args).copy()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    config.update({"optimizer": "Adam", "augmentation": False, "dropout": False,
                   "deterministic": True, "torch_threads": 4, "validation_split": "none"})
    metadata = {
        "config": config, "batch": dataset.records, "audit": audit,
        "total_frames": dataset.total_frames, "eligible_frames": dataset.eligible_frames,
        "npz_sha256": dataset.source_hashes,
        "manifest_sha256": sha256(manifest_path) if manifest_path.exists() else None,
        "channel_names": data_manifest.get("channel_names", []),
        "feature_schema_version": data_manifest.get("schema_version"),
        "label_method": data_manifest.get("intent_label_method", "unverified"),
        "label_horizon": data_manifest.get("intent_horizon"),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
            Path(__file__), ROOT / "agents/v3/network.py")},
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "runtime": {"python": sys.version, "executable": sys.executable,
                    "torch": torch.__version__, "numpy": np.__version__,
                    "cuda": torch.version.cuda},
        "terrain_geometry_cutoff": data_manifest.get("terrain_geometry_cutoff", "unverified"),
        "drift_speed_cutoff": data_manifest.get("drift_speed_cutoff", "unverified"),
        "label_cutoff": "future physical positions t+1 onward, horizon=12; heuristic pseudo-labels",
        "leakage_audit": "backcast speed is inferred from future public observations; unavailable online before convergence",
        "memory_budget": "~5 MiB parameters/gradients/Adam; <1 GiB estimated activations/workspace, 2 GiB reserve",
    }
    device = torch.device(args.device)
    amp_enabled = args.device == "cuda" and args.precision == "bf16"
    if args.device == "cuda":
        free, total = torch.cuda.mem_get_info()
        metadata["runtime"].update({"gpu": torch.cuda.get_device_name(),
                                   "free_cuda_bytes_before": free, "total_cuda_bytes": total})
        if free < 2 * 1024**3:
            raise RuntimeError("less than the 2 GiB CUDA budget is free")
        if amp_enabled and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 unsupported; use --precision fp32")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:////home/issue/ml-workspace/artifacts/mlflow.db")
    experiment = "lux-s3-bc-single-batch"
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location="file:///home/issue/ml-workspace/artifacts/mlruns")

    def forward(model, x, y, mask):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            logits = model(x)
        labels = y.flatten(2)[:, 0].argmax(1)
        logits, _ = apply_destination_mask(logits, x, labels)
        return logits, focal_spatial_loss(logits, labels, gamma=2.0)

    with tracked_run(args.output_dir.name, experiment_name=experiment, params=config) as run:
        metadata["mlflow_run_id"] = run.info.run_id
        metadata["mlflow_tracking_uri"] = mlflow.get_tracking_uri()
        metadata["mlflow_artifact_uri"] = run.info.artifact_uri
        # Synthetic one-frame forward/backward before the real batch; no data iteration.
        with cuda_memory_guard("synthetic-smoke"):
            probe = Network().to(device)
            probe_x = torch.zeros(1, 28, 24, 24, device=device)
            probe_x[:, 25, 3, 7] = 1
            probe_x[:, 26:28] = coordinate_planes().to(device)
            probe_x[:, 14, 3, 7] = 100 / 6400
            probe_y = torch.zeros(1, 1, 24, 24, dtype=torch.uint8, device=device)
            probe_y[:, :, 3, 7] = 1
            probe_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
            _, probe_loss = forward(probe, probe_x, probe_y, probe_mask)
            probe_loss.backward()
            if not torch.isfinite(probe_loss):
                raise RuntimeError("nonfinite smoke loss")
        del probe, probe_x, probe_y, probe_mask, probe_loss, _
        set_seed(args.seed)
        x, y, mask = (tensor.to(device) for tensor in (features, target, valid))
        model = Network().to(device)
        metadata["parameters"] = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        rows = []
        train_started = time.perf_counter()
        with cuda_memory_guard("fixed-batch-training"):
            model.train()
            logits, loss = forward(model, x, y, mask)
            with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("step", "loss", "accuracy", "elapsed_seconds"))
                writer.writeheader()
                for step in range(args.epochs + 1):
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"nonfinite loss at step {step}")
                    with torch.no_grad():
                        correct = ((logits.flatten(2).argmax(2) == y.flatten(2).argmax(2)) & mask).sum()
                        accuracy = float(correct / mask.sum()) if mask.any() else 0.0
                    row = {"step": step, "loss": float(loss.detach()), "accuracy": accuracy,
                           "elapsed_seconds": time.perf_counter() - train_started}
                    rows.append(row)
                    writer.writerow(row)
                    stream.flush()
                    mlflow.log_metrics({"batch_train_loss": row["loss"],
                                        "batch_train_accuracy": accuracy}, step=step)
                    if step % 25 == 0 or step == args.epochs:
                        print(json.dumps(row), flush=True)
                    if step == args.epochs:
                        break
                    optimizer.zero_grad(set_to_none=True)
                    if mask.any():
                        loss.backward()
                        optimizer.step()
                    logits, loss = forward(model, x, y, mask)
            model.eval()
            with torch.no_grad():
                eval_logits, eval_loss = forward(model, x, y, mask)
                eval_correct = ((eval_logits.flatten(2).argmax(2) == y.flatten(2).argmax(2)) & mask).sum()
                eval_accuracy = float(eval_correct / mask.sum()) if mask.any() else 0.0
        summary = {
            "initial": rows[0], "final_train": rows[-1],
            "final_eval_same_batch": {"loss": float(eval_loss), "accuracy": eval_accuracy},
            "optimizer_steps": args.epochs if mask.any() else 0,
            "first_100_percent_step": next((r["step"] for r in rows if r["accuracy"] == 1.0), None),
            "first_loss_below_0_01_step": next((r["step"] for r in rows if r["loss"] < .01), None),
            "passed": rows[-1]["loss"] < .01 and rows[-1]["accuracy"] == 1.0
                      and float(eval_loss) < .01 and eval_accuracy == 1.0,
            "training_wall_seconds": time.perf_counter() - train_started,
            "rss_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
        if args.device == "cuda":
            from dataclasses import asdict
            summary["cuda_memory"] = asdict(cuda_memory_stats())
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "metadata": metadata, "summary": summary}, args.output_dir / "checkpoint.pt")
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot([r["step"] for r in rows], [r["loss"] for r in rows])
        axes[0].set(yscale="log", xlabel="Optimizer steps", ylabel="Masked CE (log scale)")
        axes[1].plot([r["step"] for r in rows], [r["accuracy"] * 100 for r in rows])
        axes[1].set(xlabel="Optimizer steps", ylabel="Valid-unit top-1 accuracy (%)", ylim=(0, 102))
        fig.suptitle("One fixed batch: training-mode metrics, no holdout")
        fig.tight_layout()
        fig.savefig(args.output_dir / "loss_curve.png", dpi=160)
        plt.close(fig)
        summary["wall_seconds_to_artifacts"] = time.perf_counter() - started
        metadata["summary"] = summary
        (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        mlflow.log_metrics({"final_loss": rows[-1]["loss"], "final_accuracy": rows[-1]["accuracy"],
                            "eval_same_batch_loss": float(eval_loss), "eval_same_batch_accuracy": eval_accuracy,
                            "wall_seconds": summary["wall_seconds_to_artifacts"],
                            "valid_labels": audit["valid_labels"], "rss_peak_mib": summary["rss_peak_mib"]})
        mlflow.log_artifacts(str(args.output_dir))
        print(json.dumps({"run_id": run.info.run_id, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
