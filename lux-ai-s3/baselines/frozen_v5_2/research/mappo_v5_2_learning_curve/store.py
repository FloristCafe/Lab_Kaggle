"""Durable full-game transactions. last.pth is the committed authority."""

import hashlib
import json
import os
from pathlib import Path

import torch


class InjectedFailure(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_dir(path.parent)


def sync_tree(path):
    for file in path.rglob("*"):
        if file.is_file():
            with file.open("rb") as stream:
                os.fsync(stream.fileno())
    for directory in sorted((p for p in path.rglob("*") if p.is_dir()), reverse=True):
        sync_dir(directory)
    sync_dir(path)


class Store:
    def __init__(self, root, milestones):
        self.root = Path(root)
        self.milestones = tuple(milestones)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "checkpoints").mkdir(exist_ok=True)
        (self.root / "attempts").mkdir(exist_ok=True)
        (self.root / "quarantine").mkdir(exist_ok=True)

    def episode_dir(self, attempt, index):
        path = self.root / "attempts" / f"{attempt:04d}" / "episodes" / f"{index:06d}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def load(self):
        state = torch.load(
            self.root / "last.pth", map_location="cpu", weights_only=False
        )
        for commit in state["progress"]["commits"]:
            directory = self.root / commit["directory"]
            for name, digest in commit["files"].items():
                if sha(directory / name) != digest:
                    raise ValueError(
                        "committed artifact missing/corrupted: " + str(directory / name)
                    )
        self.repair_milestone(state)
        return state

    def repair_milestone(self, state):
        count = state["progress"]["next_game"]
        if count not in self.milestones:
            return
        target = self.root / "checkpoints" / f"episode_{count:04d}.pth"
        if target.exists():
            if sha(target) != sha(self.root / "last.pth"):
                raise ValueError(
                    "immutable milestone differs from committed checkpoint"
                )
        else:
            os.link(self.root / "last.pth", target)
            sync_dir(target.parent)

    def commit(self, state, directory, fault=None):
        # All episode journals become durable BEFORE last.pth can reference them.
        target = (
            self.root
            / "checkpoints"
            / f"episode_{state['progress']['next_game']:04d}.pth"
        )
        if state["progress"]["next_game"] in self.milestones and target.exists():
            raise ValueError("immutable milestone already exists; authority unchanged")
        sync_tree(directory)
        parent = directory.parent
        while parent.is_relative_to(self.root):
            sync_dir(parent)
            if parent == self.root:
                break
            parent = parent.parent
        if fault == "after_logs":
            raise InjectedFailure("logs durable, checkpoint not submitted")
        pending = directory / "checkpoint.pending"
        if fault == "partial_save":
            with pending.open("wb") as stream:
                stream.write(b"interrupted checkpoint")
                stream.flush()
                os.fsync(stream.fileno())
            raise InjectedFailure("interrupted checkpoint serialization")
        with pending.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if fault == "before_replace":
            raise InjectedFailure("checkpoint prepared, not committed")
        os.replace(pending, self.root / "last.pth")
        sync_dir(self.root)
        if fault == "after_replace":
            raise InjectedFailure("committed checkpoint, milestone link not finalized")
        self.repair_milestone(state)

    def reconcile(self, state):
        retained = {item["directory"] for item in state["progress"]["commits"]}
        moved = []
        for directory in sorted((self.root / "attempts").glob("*/episodes/*")):
            relative = str(directory.relative_to(self.root))
            if relative in retained:
                continue
            destination = (
                self.root
                / "quarantine"
                / (directory.parent.parent.name + "_" + directory.name)
            )
            if destination.exists():
                raise ValueError("quarantine collision; do not overwrite evidence")
            os.replace(directory, destination)
            moved.append(relative)
        if moved:
            sync_dir(self.root / "quarantine")
        return moved


def artifacts(directory):
    return {
        str(file.relative_to(directory)): sha(file)
        for file in sorted(directory.rglob("*"))
        if file.is_file()
    }


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone_tree(item) for item in value)
    return value


def tree_hash(value):
    """Canonical numerical hash; excludes archive timestamps and object identity."""
    import numpy as np

    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(
                b"tensor"
                + str(tensor.dtype).encode()
                + str(tuple(tensor.shape)).encode()
            )
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(
                b"array"
                + str(item.dtype).encode()
                + str(item.shape).encode()
                + item.tobytes()
            )
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + str(len(item)).encode())
            for child in item:
                visit(child)
        else:
            digest.update(type(item).__name__.encode() + repr(item).encode() + b"\0")

    visit(value)
    return digest.hexdigest()
