"""Read-only full-label coverage audit for cached physical destination masks."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from agents.v3.dataset import coordinate_planes, decode_features, validate_labels
from agents.v3.losses import optimistic_destination_mask


def audit_physical_masks(paths, dataset_hashes, output_path, batch_size=128):
    output_path = Path(output_path)
    dependencies = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ("dataset.py", "losses.py", "mask_audit.py")}
    identity = {"dataset_sha256": dataset_hashes, "code_sha256": dependencies, "horizon": 12}
    if output_path.exists():
        previous = json.loads(output_path.read_text())
        if all(previous.get(k) == v for k, v in identity.items()):
            if previous.get("excluded_supervised_targets") != 0:
                raise ValueError("previous physical mask audit excludes targets")
            return previous
    started = time.perf_counter()
    total = excluded = allowed_cells = original_total = 0
    file_stats = {}
    coords = coordinate_planes()
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            valid, _, target, positions = validate_labels(data)
            features = decode_features(data)
            frames, slots = np.where(valid)
            original_count = int(data["intent_valid"].sum())
            original_total += original_count
            hold_correct = 0
            for start in range(0, len(frames), batch_size):
                fs, us = frames[start:start + batch_size], slots[start:start + batch_size]
                count = len(fs)
                x = torch.zeros(count, 28, 24, 24)
                x[:, :25] = torch.from_numpy(features[fs])
                x[:, 26:28] = coords
                xy = positions[fs, us]
                x[torch.arange(count), 25, xy[:, 0], xy[:, 1]] = 1
                labels = target[fs, us].reshape(count, 576).argmax(1)
                hold_correct += int((labels == xy[:, 0] * 24 + xy[:, 1]).sum())
                allowed = optimistic_destination_mask(x).flatten(1)
                rejected = ~allowed[torch.arange(count), labels]
                total += count
                allowed_cells += int(allowed.sum())
                excluded += int(rejected.sum())
                if rejected.any():
                    first = int(torch.nonzero(rejected)[0])
                    raise ValueError(f"{path.name}: frame={fs[first]}, unit={us[first]}, "
                                     f"target={labels[first]} excluded by physical mask")
            file_stats[path.name] = {"original_valid_units": original_count, "valid_units": len(frames),
                                     "filtered_lifecycle_labels": original_count - len(frames),
                                     "hold_correct": hold_correct}
        if (index + 1) % 50 == 0:
            print(f"mask audit {index + 1}/{len(paths)} files; targets={total}", flush=True)
    report = {**identity, "files": len(paths), "valid_units": total,
              "original_valid_units": original_total,
              "filtered_lifecycle_labels": original_total - total, "file_stats": file_stats,
              "excluded_supervised_targets": excluded,
              "allowed_cell_fraction": allowed_cells / (total * 576),
              "wall_seconds": time.perf_counter() - started,
              "scope": "CPU read-only existing labels, no model or optimizer updates"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output_path)
    return report
