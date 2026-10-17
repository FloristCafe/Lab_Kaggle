"""Compact replay shards with temporary unit anchors and bounded host memory."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from scripts.replay_parser import MASK_CHANNELS, SCHEMA_VERSION, VALUE_CHANNELS


def coordinate_planes():
    axis = torch.linspace(-1, 1, 24, dtype=torch.float32)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((x, y))


def decode_features(data):
    if int(data["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("expected compact schema 3; regenerate legacy NPZ files")
    masks, values = data["feature_masks"], data["feature_values"]
    frames = len(masks)
    if (masks.shape != (frames, len(MASK_CHANNELS), 24, 24)
            or masks.dtype != np.uint8 or not np.isin(masks, (0, 1)).all()
            or values.shape != (frames, len(VALUE_CHANNELS), 24, 24)
            or values.dtype != np.float16 or not np.isfinite(values).all()):
        raise ValueError("invalid compact feature schema")
    features = np.empty((frames, 25, 24, 24), dtype=np.float32)
    features[:, MASK_CHANNELS] = masks
    features[:, VALUE_CHANNELS] = values
    return features


def lifecycle_exclusions(positions, active, steps, match_steps, horizon=12, feature_values=None):
    """Exclude supervision crossing physically discontinuous active-slot reuse.

    A living unit moves at most one Manhattan cell per consecutive observation.
    Disk targets stay archived; this is a label-only filter, not a feature.
    """
    jumps = np.zeros_like(active)
    jumps[1:] = (active[1:] & active[:-1]
                 & (np.abs(positions[1:] - positions[:-1]).sum(-1) > 1)
                 & (np.diff(steps)[:, None] == 1)
                 & (np.diff(match_steps)[:, None] >= 0))
    if feature_values is not None:
        # At singleton tiles the stored aggregate identifies the unit energy.
        # A >20 gain cannot come from legal recovery and reveals same-cell
        # respawning too. Use ceil/floor bounds to avoid FP16 false positives.
        frames = np.arange(len(active))[:, None]
        x, y = positions[..., 0].clip(0, 23), positions[..., 1].clip(0, 23)
        counts = feature_values[:, VALUE_CHANNELS.index(13)][frames, x, y] * 16
        energies = feature_values[:, VALUE_CHANNELS.index(14)][frames, x, y].astype(np.float32) * 6400
        reset_energy = ((counts[1:] == 1) & (counts[:-1] == 1)
                        & (np.floor(energies[1:]) > np.ceil(energies[:-1]) + 20))
        jumps[1:] |= (reset_energy & active[1:] & active[:-1]
                      & (np.diff(steps)[:, None] == 1)
                      & (np.diff(match_steps)[:, None] >= 0))
    match_id = np.cumsum(np.r_[False, np.diff(match_steps) < 0])
    excluded = np.zeros_like(active)
    # Match-step zero retains prior units until the next tick clears/spawns
    # them. Their energy is not the new unit's initial 100 energy.
    excluded[(match_steps == 0) & (steps > 0)] = active[(match_steps == 0) & (steps > 0)]
    for lag in range(1, min(horizon + 1, len(active))):
        same_segment = ((match_id[lag:] == match_id[:-lag])
                        & (steps[lag:] - steps[:-lag] == lag))
        excluded[:-lag] |= jumps[lag:] & same_segment[:, None]
    return excluded


def validate_labels(data):
    valid, active = data["intent_valid"], data["active"]
    target, positions = data["intent_target"], data["positions"]
    frames = len(valid)
    if (valid.shape != (frames, 16) or valid.dtype != np.bool_
            or active.shape != valid.shape or active.dtype != np.bool_
            or positions.shape != (frames, 16, 2)
            or target.shape != (frames, 16, 24, 24) or target.dtype != np.uint8):
        raise ValueError("invalid compact label schema")
    if (valid & ~active).any():
        raise ValueError("valid label on a dead unit")
    if ((positions[valid] < 0) | (positions[valid] >= 24)).any():
        raise ValueError("valid unit anchor outside map")
    labels = target[valid]
    if not np.isin(labels, (0, 1)).all() or not (labels.sum((1, 2)) == 1).all():
        raise ValueError("valid targets must be one-hot")
    valid = valid & ~lifecycle_exclusions(positions, active, data["steps"], data["match_steps"],
                                         feature_values=data["feature_values"])
    return valid, active, target, positions


class LegacyProjectionFrameDataset(Dataset):
    """Each item is [K,28,24,24] and K spatial class IDs, for one valid frame.

    Index construction loads only labels, one replay at a time. The LRU holds
    at most cache_size decoded shards; it never holds the full dataset. Future
    dropout is drawn once per frame, before expanding its K unit queries.
    """

    def __init__(self, paths, future_dropout=0.0, cache_size=1):
        self.paths = sorted(Path(p) for p in paths)
        if not self.paths or cache_size < 1 or not 0 <= future_dropout <= 1:
            raise ValueError("require replay paths, positive cache and valid dropout")
        self.future_dropout, self.cache_size = future_dropout, cache_size
        self.coordinates = coordinate_planes()
        self.cache = OrderedDict()
        self.frame_ids, sizes = [], []
        self.total_frames = self.valid_units = self.original_valid_units = 0
        for path in self.paths:
            with np.load(path, allow_pickle=False) as data:
                if int(data["schema_version"]) != SCHEMA_VERSION:
                    raise ValueError(f"{path}: expected compact schema 3")
                valid, _, _, _ = validate_labels(data)
                ids = np.flatnonzero(valid.any(1)).astype(np.int32)
                self.frame_ids.append(ids)
                sizes.append(len(ids))
                self.total_frames += len(valid)
                self.valid_units += int(valid.sum())
                self.original_valid_units += int(data["intent_valid"].sum())
        self.offsets = np.concatenate(([0], np.cumsum(sizes)))
        if not len(self):
            raise ValueError("dataset has no valid intent labels")

    def __len__(self):
        return int(self.offsets[-1])

    def _load(self, shard):
        if shard in self.cache:
            self.cache.move_to_end(shard)
            return self.cache[shard]
        with np.load(self.paths[shard], allow_pickle=False) as data:
            valid, active, target, positions = validate_labels(data)
            result = (decode_features(data), target, valid, active, positions)
        self.cache[shard] = result
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = int(np.searchsorted(self.offsets, index, side="right") - 1)
        frame = int(self.frame_ids[shard][index - self.offsets[shard]])
        features, targets, valid, active, positions = self._load(shard)
        units = np.flatnonzero(valid[frame] & active[frame])
        global_map = torch.from_numpy(features[frame].copy())
        dropped = bool(torch.rand(()) < self.future_dropout)
        if dropped:
            global_map[21:25] = 0
        queries = torch.zeros((len(units), 28, 24, 24), dtype=torch.float32)
        queries[:, :25] = global_map
        queries[:, 26:28] = self.coordinates
        xy = positions[frame, units]
        queries[torch.arange(len(units)), 25, xy[:, 0], xy[:, 1]] = 1
        labels = torch.from_numpy(targets[frame, units].reshape(len(units), -1)
                                  .argmax(1).astype(np.int64))
        return queries, labels, dropped


def collate_unit_frames(rows):
    """Concatenate variable K; carry dropout statistics at frame granularity."""
    return (torch.cat([row[0] for row in rows]),
            torch.cat([row[1] for row in rows]),
            sum(row[2] for row in rows), len(rows))


class ShardFrameSampler(Sampler):
    """Shuffle replay order and frames within each replay, avoiding ZLIB thrash."""

    def __init__(self, dataset, seed=42):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for shard in rng.permutation(len(self.dataset.paths)):
            start, end = self.dataset.offsets[shard:shard + 2]
            yield from (int(i) for i in rng.permutation(np.arange(start, end)))


# Backcast only overwrote 6..8 and appended 21..24. Every selected current
# plane below was built by ReplayReconstructor._feature in chronological order.
CAUSAL_CURRENT_CHANNELS = (0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
HISTORY_CHANNELS = (0, 3, 4, 5, 9, 15, 16)
CAUSAL_ANCHOR_CHANNEL = 32
CAUSAL_INPUT_CHANNELS = 35


def gaussian_heatmaps(centers, sigma=1.5):
    """Euclidean Gaussian, peak 1, no normalization, computed only in memory."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian sigma must be finite and positive")
    centers = torch.as_tensor(centers, dtype=torch.float32)
    if centers.ndim != 2 or centers.shape[1] != 2 or ((centers < 0) | (centers > 23)).any():
        raise ValueError("expected target centers [K,2] inside the map")
    axis = torch.arange(24, dtype=torch.float32, device=centers.device)
    gx, gy = torch.meshgrid(axis, axis, indexing="ij")
    distance2 = ((gx[None] - centers[:, 0, None, None]).square()
                 + (gy[None] - centers[:, 1, None, None]).square())
    return torch.exp(-distance2 / (2 * sigma**2))


def normalize_spatial_targets(target):
    """Normalize each floating [K,24,24] heatmap to a probability distribution."""
    if target.ndim != 3 or target.shape[1:] != (24, 24):
        raise ValueError("expected spatial targets [K,24,24]")
    target = target.float()
    if not torch.isfinite(target).all() or (target < 0).any():
        raise ValueError("targets must be finite and nonnegative")
    mass = target.flatten(1).sum(1)
    if (mass <= 0).any():
        raise ValueError("target must have positive probability mass")
    return target / mass[:, None, None]


def causal_frame_maps(features, steps, match_steps, frame):
    """T current legal planes + physical planes at T-1/T-2; never backcast.

    History is zero padded at missing observations or match reset. Visibility
    is present in each history block so padded history is distinguishable.
    A lag-2 read requires *both* transitions to be contiguous in the same match.
    """
    output = np.zeros((32, 24, 24), dtype=np.float32)
    output[:18] = features[frame, CAUSAL_CURRENT_CHANNELS]
    contiguous = True
    for lag in (1, 2):
        past = frame - lag
        contiguous = (contiguous and past >= 0
                      and steps[past + 1] - steps[past] == 1
                      and match_steps[past + 1] - match_steps[past] == 1)
        if contiguous:
            output[18 + (lag - 1) * 7:18 + lag * 7] = features[past, HISTORY_CHANNELS]
    return output


class UnitFrameDataset(LegacyProjectionFrameDataset):
    """Active causal Dataset: [K,35,24,24], [K,24,24] Gaussian probabilities, bool [K].

    Only compressed archival NPZ are read; neither history nor soft labels are
    stored. Each target sums to one; normalize_targets=False is archival
    MSE compatibility. Future evidence affects label validity only, never inputs.
    """

    def __init__(self, paths, sigma=1.5, cache_size=1, normalize_targets=True):
        super().__init__(paths, future_dropout=0.0, cache_size=cache_size)
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Gaussian sigma must be finite and positive")
        self.sigma = sigma
        self.normalize_targets = normalize_targets
        self.timelines = []
        for path in self.paths:
            with np.load(path, allow_pickle=False) as data:
                self.timelines.append((data["steps"], data["match_steps"]))

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = int(np.searchsorted(self.offsets, index, side="right") - 1)
        frame = int(self.frame_ids[shard][index - self.offsets[shard]])
        features, targets, valid, active, positions = self._load(shard)
        units = np.flatnonzero(valid[frame] & active[frame])
        steps, match_steps = self.timelines[shard]
        current_history = torch.from_numpy(causal_frame_maps(features, steps, match_steps, frame))
        queries = torch.zeros((len(units), CAUSAL_INPUT_CHANNELS, 24, 24), dtype=torch.float32)
        queries[:, :32] = current_history
        queries[:, 33:35] = self.coordinates
        xy = positions[frame, units]
        queries[torch.arange(len(units)), CAUSAL_ANCHOR_CHANNEL, xy[:, 0], xy[:, 1]] = 1
        centers = targets[frame, units].reshape(len(units), -1).argmax(1)
        centers = np.stack((centers // 24, centers % 24), axis=1)
        heatmaps = gaussian_heatmaps(centers, self.sigma)
        if self.normalize_targets:
            heatmaps = normalize_spatial_targets(heatmaps)
        return queries, heatmaps, torch.ones(len(units), dtype=torch.bool)


def collate_spatial_frames(rows):
    return tuple(torch.cat([row[i] for row in rows]) for i in range(3))
