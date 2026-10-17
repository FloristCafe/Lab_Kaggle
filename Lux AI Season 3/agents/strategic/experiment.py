"""Small reproducibility helpers shared by all new experiments."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def setup(seed=42):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault(
        "MLFLOW_TRACKING_URI", "sqlite:////home/issue/ml-workspace/artifacts/mlflow.db"
    )
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def provenance():
    def git(*args, cwd=ROOT):
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()

    return dict(
        git_commit=git("rev-parse", "HEAD"),
        git_status=git("status", "--short"),
        environment_commit=git("rev-parse", "HEAD", cwd=ROOT / "vendor/lux-design-s3"),
        python=sys.executable,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        source_sha256={
            str(p.relative_to(ROOT)): digest(p)
            for folder in [ROOT / "agents/strategic"]
            for p in sorted(folder.glob("*.py"))
        },
    )
