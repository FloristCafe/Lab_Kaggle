"""Source and data identity, RNG, and durable checkpoint primitives."""
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "training/configs"
ARTIFACTS = Path(os.environ.get("LUX_ARTIFACTS", ROOT / "runs")).expanduser()


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def json_write(path, value):
    with Path(path).open("x", encoding="utf8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def sync(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def config(name):
    spec = read(CONFIG / "rl" / f"model_{name}.json")
    common = read(CONFIG / "rl" / spec["inherits"])
    return dict(common, model=spec["model"], initialization=spec["initialization"], teacher_kl=spec["teacher_kl"])


def map_pool():
    raw = read(CONFIG / "maps.json")
    roots = raw["training"] + raw["development"]
    if (len(raw["training"]) != 900 or len(raw["development"]) != 24 or
        not all(type(r) is int and 2**31+104 <= r <= 2**32-104 for r in roots) or
        len({x for r in roots for x in (r, r+103)}) != 2 * len(roots)):
        raise RuntimeError("map range, allocation or derived collision")
    return raw


def source_lock():
    paths = [p for d in ("agents", "training", "scripts", "vendor")
             for p in (ROOT/d).rglob("*") if p.is_file() and not p.is_symlink()
             and p.suffix in (".py", ".json")]
    paths += [ROOT/"requirements.txt", ROOT/"requirements-engine.txt", ROOT/"requirements-opponents.txt",
              ROOT/"vendor/lux-design-s3/REVISION"]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def setup(device, seed):
    torch.set_num_threads(2)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        free, total = torch.cuda.mem_get_info()
        if free < 2 * 2**30:
            raise RuntimeError("CUDA requires at least 2 GiB free")
        torch.cuda.set_per_process_memory_fraction(min(.75, (free - 2**30) / total))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def engine_contract():
    path = Path(os.environ.get("LUX_ENGINE_PYTHON", sys.executable))
    code = ("import importlib.metadata as m,sys,json;"
            "print(json.dumps({'python':sys.version.split()[0], **{n:m.version(n) "
            "for n in ('numpy','scipy','jax','jaxlib','flax','chex','gymnax','tyro')}}))")
    return json.loads(subprocess.check_output([str(path), "-c", code], text=True))


def git_identity():
    def query(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": query("rev-parse", "HEAD"), "dirty": bool(query("status", "--porcelain"))}


def optimizer_pair(actor, critic, cfg):
    return tuple(torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=0)
                 for m in (actor, critic))


def training_slot(index, pool):
    cycle, position = divmod(index, len(pool))
    return pool[position], (cycle + position) % 2


def committed(run):
    commits = sorted(p for p in run.glob("batch_*") if p.is_dir())
    if not commits:
        return None
    for i, path in enumerate(commits):
        if path.name != f"batch_{i:06d}":
            raise RuntimeError("committed checkpoints are not contiguous")
    last = commits[-1]
    receipt = read(last / "commit.json")
    if any(sha(last / n) != h for n, h in receipt.items()):
        raise RuntimeError("last committed files do not match the receipt")
    return last


def commit(run, staged, count, payload, stats):
    torch.save(payload, staged / "checkpoint.pt")
    json_write(staged / "metrics.json", stats)
    paths = sorted(staged.iterdir())
    for p in paths:
        sync(p)
    json_write(staged / "commit.json", {p.name: sha(p) for p in paths})
    sync(staged)
    dest = run / f"batch_{count:06d}"
    if dest.exists():
        raise FileExistsError(dest)
    staged.rename(dest)
    sync(run)
    return dest
