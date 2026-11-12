"""No-environment static preflight for a clean or existing WSL setup."""
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import torch

from training.contracts import ARTIFACTS, CONFIG, engine_contract, map_pool, read, sha, source_lock

pool = map_pool()
external = None
if os.environ.get("LUX_OPPONENT_REGISTRY"):
    from training.opponents.pool import registry
    external=[name for name,spec in registry()["agents"].items() if spec["enabled"]]
locked = root / "release_source_lock.json"
if locked.is_file() and read(locked) != source_lock():
    raise RuntimeError("release source lock drift")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable for the default formal configuration")
fs = subprocess.check_output(["stat","-f","-c","%T",str(root)],text=True).strip()
if fs != "ext2/ext3":
    raise RuntimeError(f"expected Linux ext4 worktree, found {fs}")
free, total = torch.cuda.mem_get_info()
if free < 2*2**30 or shutil.disk_usage(root).free < 4*2**30:
    raise RuntimeError("insufficient free resources")
print(json.dumps({"status":"PASS","python":sys.executable,"torch":torch.__version__,
                  "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),
                  "gpu_free_mib":free//2**20,"gpu_total_mib":total//2**20,
                  "kernel":platform.release(),"filesystem":fs,
                  "engine":engine_contract(),"artifact_dir":str(ARTIFACTS),
                  "map_sha256":sha(CONFIG/"maps.json"),"train_maps":len(pool["training"]),
                  "dev_maps":len(pool["development"]),"admitted_external_opponents":external,"formal_RL_initialization":"matching_calibration_selected_BC_required","environment_starts":0},indent=2))
