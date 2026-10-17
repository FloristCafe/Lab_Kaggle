"""Read-only re-evaluation of baseline_v1_waypoint_il; no optimizer."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.strategic.experiment import digest, provenance, save_json, setup
from agents.v3.dataset import UnitFrameDataset, collate_spatial_frames
from agents.v3.network import Network
from ml_workspace.tracking import tracked_run
from scripts.train_spatial_bc import evaluate_distributions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "outputs/bc_causal_gaussian_kl400_fold0_seed42",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "experiments/A_waypoint_reproduction"
    )
    args = parser.parse_args()
    import json

    manifest = json.loads((args.reference / "manifest.json").read_text())
    paths = [
        ROOT / "data/processed_replays_unitcentric25" / p
        for p in manifest["validation_files"]
    ]
    for path in paths:
        assert digest(path) == manifest["npz_sha256"][path.name]
    setup()
    dataset = UnitFrameDataset(paths, sigma=1.5)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_spatial_frames)
    model = Network().cuda().eval()
    checkpoint = torch.load(args.reference / "best_model.pth", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    args.output.mkdir(parents=True, exist_ok=True)
    with tracked_run(
        "baseline_v1_waypoint_il_reproduction",
        experiment_name="lux-s3-strategic-baseline",
        params={
            "seed": 42,
            "checkpoint_epoch": checkpoint["epoch"],
            "optimizer_updates": 0,
        },
    ) as run:

        def prediction(x):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model(x)

        metrics = evaluate_distributions(prediction, loader, torch.device("cuda"))
        hits = np.zeros(3, dtype=np.int64)
        count = 0
        with torch.no_grad():
            for x, y, valid in loader:
                for start in range(0, len(y), 32):
                    logits = prediction(x[start : start + 32].cuda()).flatten(1)
                    target = y[start : start + 32].flatten(1).argmax(1).cuda()
                    ranked = logits.topk(10, 1).indices
                    for i, k in enumerate((1, 5, 10)):
                        hits[i] += int((ranked[:, :k] == target[:, None]).any(1).sum())
                    count += len(target)
        metrics.update(
            {f"top_{k}": int(hits[i]) / count for i, k in enumerate((1, 5, 10))}
        )
        summary = dict(
            baseline="baseline_v1_waypoint_il",
            checkpoint_epoch=checkpoint["epoch"],
            checkpoint_sha256=digest(args.reference / "best_model.pth"),
            metrics=metrics,
            validation_games=80,
            optimizer_updates=0,
            provenance=provenance(),
            note="Requested ~12.44%/16.2% describes final epoch17, not saved best epoch12.",
            mlflow_run_id=run.info.run_id,
        )
        save_json(args.output / "summary.json", summary)
        import mlflow

        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(args.output))
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
