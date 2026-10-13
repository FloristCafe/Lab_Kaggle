"""Build observation-only tensors and heuristic macro-intent labels from Kaggle replays."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.v2.equation_solver import EquationSolver
from agents.v2.obs_parser import ObsParser

BASE_CHANNEL_NAMES = (
    "visible", "unseen", "observation_age", "visible_empty",
    "visible_nebula", "visible_asteroid", "last_empty", "last_nebula",
    "last_asteroid", "visible_energy", "known_relic_support",
    "score_probability", "score_confidence", "own_count", "own_energy",
    "enemy_count", "enemy_energy", "previous_occupied", "current_occupied",
    "known_safe_zero", "known_score_one",
)
PROJECTION_CHANNEL_NAMES = (
    "future_nebula_5", "future_asteroid_5",
    "future_nebula_10", "future_asteroid_10",
)
CHANNEL_NAMES = BASE_CHANNEL_NAMES + PROJECTION_CHANNEL_NAMES
MASK_CHANNELS = (0, 1, 3, 4, 5, 6, 7, 8, 10, 12, 17, 18, 19, 20, 21, 22, 23, 24)
VALUE_CHANNELS = tuple(i for i in range(25) if i not in MASK_CHANNELS)
SCHEMA_VERSION = 3


def pack_features(features):
    """Keep binary masks exact; quantize continuous values separately to FP16."""
    masks = features[:, MASK_CHANNELS]
    values = features[:, VALUE_CHANNELS]
    if not np.isin(masks, (0, 1)).all():
        raise ValueError("mask channels must be binary before quantization")
    if not np.isfinite(values).all() or np.abs(values).max() > np.finfo(np.float16).max:
        raise ValueError("numeric feature is not representable in float16")
    return {"feature_masks": masks.astype(np.uint8),
            "feature_values": values.astype(np.float16)}


def dataset_manifest(summaries, source_hashes, horizon=12):
    return {
        "schema_version": SCHEMA_VERSION,
        "parser_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "channel_names": list(CHANNEL_NAMES),
        "mask_channels": list(MASK_CHANNELS), "value_channels": list(VALUE_CHANNELS),
        "storage": "uint8 masks + float16 values; savez_compressed; no anchor on disk",
        "intent_label_method": "future_position_mode_with_move_evidence_and_stationary_filter",
        "intent_filter": "reject max Manhattan displacement <3 unless current visible enemy is cardinal-adjacent or unit is on map edge",
        "intent_horizon": horizon,
        "observation_source": "steps[*][side].observation.obs",
        "terrain_geometry_cutoff": "public observations through current frame only",
        "drift_speed_cutoff": "end-of-replay hindsight; future observation leakage before convergence",
        "drift_evidence_policy": "insufficient candidate overlap is inconclusive; conflict requires every candidate to be compared and refuted",
        "projection_horizons": [5, 10],
        "projection_semantics": "known nebula/asteroid masks; unknown tiles remain zero",
        "hidden_state_fields_used": [], "source_sha256": source_hashes,
        "replays": summaries,
    }

# The environment uses two different coordinate symmetries.
TILE_SYMMETRY_CHANNELS = (6, 7, 8)
ENTITY_SYMMETRY_CHANNELS = (9, 10, 11, 12, 19, 20)
NEBULA_DRIFT_SPEEDS = (-0.15, -0.1, -0.05, -0.025, 0.025, 0.05, 0.1, 0.15)


class TileDriftEstimator:
    """Eliminate nebula drift speeds using only consecutive legal observations."""

    def __init__(self, width=24, height=24, min_overlap=8):
        self.width, self.height = width, height
        self.min_overlap = min_overlap
        self.candidate_speeds = tuple(NEBULA_DRIFT_SPEEDS)
        self.previous_tile = None
        self.previous_visible = None
        self.previous_step = None
        self.history = []
        self.converged_step = None

    @staticmethod
    def _moves_at_step(speed, step):
        magnitude = abs(float(speed))
        # The engine increments state.steps after shifting and before get_obs.
        return (step - 2) * magnitude % 1 > (step - 1) * magnitude % 1

    @staticmethod
    def _shift_for_speed(speed):
        direction = 1 if speed > 0 else -1
        return direction, -direction

    def observe(self, step, tile_type, sensor_mask):
        """Update candidates and return the current mean speed and diagnostics."""
        tile_type = np.asarray(tile_type)
        sensor_mask = np.asarray(sensor_mask, dtype=bool)
        valid = sensor_mask & (tile_type >= 0)
        if tile_type.shape != (self.width, self.height):
            raise ValueError(f"expected {(self.width, self.height)} tile map, got {tile_type.shape}")
        if (self.previous_tile is None or self.previous_step is None
                or step != self.previous_step + 1):
            self.previous_tile = tile_type.copy()
            self.previous_visible = valid.copy()
            self.previous_step = int(step)
            return self._record(step, None, False, None)

        scored = []
        overlap_counts = []
        for speed in self.candidate_speeds:
            moves = self._moves_at_step(speed, int(step))
            if moves:
                shift = self._shift_for_speed(speed)
                expected = np.roll(self.previous_tile, shift=shift, axis=(0, 1))
                expected_valid = np.roll(self.previous_visible, shift=shift, axis=(0, 1))
            else:
                expected = self.previous_tile
                expected_valid = self.previous_visible
            overlap = valid & expected_valid
            overlap_count = int(overlap.sum())
            overlap_counts.append(overlap_count)
            if overlap_count < self.min_overlap:
                scored.append((speed, None))
                continue
            mismatches = int(np.count_nonzero(tile_type[overlap] != expected[overlap]))
            scored.append((speed, mismatches))

        exact = [speed for speed, mismatches in scored if mismatches == 0]
        survivors = (
            [speed for speed, mismatches in scored if mismatches in (None, 0)]
            if exact else [speed for speed, _ in scored]
        )

        if self.candidate_speeds:
            self.candidate_speeds = tuple(survivors)
        if len(self.candidate_speeds) == 1 and self.converged_step is None:
            self.converged_step = int(step)
        self.previous_tile = tile_type.copy()
        self.previous_visible = valid.copy()
        self.previous_step = int(step)
        comparable = max(overlap_counts, default=0)
        # A compared candidate can fail while the true speed has too little
        # visible overlap to compare. Only an exhaustive refutation conflicts
        # with the model. Keep the existing conservative candidate updates.
        has_uncompared_candidate = any(mismatches is None for _, mismatches in scored)
        evidence = True if exact else (None if has_uncompared_candidate else False)
        return self._record(
            step, comparable, len(survivors) != len(overlap_counts),
            evidence,
        )

    def _record(self, step, comparable_cells, changed, has_exact_candidate):
        speed = (float(np.mean(self.candidate_speeds))
                 if self.candidate_speeds else np.nan)
        record = {
            "step": int(step),
            "candidate_speeds": tuple(self.candidate_speeds),
            "candidate_count": len(self.candidate_speeds),
            "speed_mean": speed,
            "comparable_cells": comparable_cells,
            "changed": bool(changed),
            "has_exact_candidate": has_exact_candidate,
        }
        self.history.append(record)
        return record


def _dual_symmetry_map(array):
    """Map (x, y) to (23-y, 23-x)."""
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"expected a square spatial map, got {array.shape}")
    return np.flip(array, axis=(0, 1)).T


def _center_symmetry_map(array):
    """Map (x, y) to (23-x, 23-y)."""
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"expected a square spatial map, got {array.shape}")
    return np.flip(array, axis=(0, 1))


def apply_dual_symmetry(features):
    """Fuse deterministic mirrored evidence in the observed feature planes."""
    output = np.asarray(features).copy()
    if output.ndim != 3 or output.shape[0] != len(BASE_CHANNEL_NAMES):
        raise ValueError(f"expected ({len(BASE_CHANNEL_NAMES)}, W, H), got {output.shape}")
    for channel in TILE_SYMMETRY_CHANNELS:
        output[channel] = np.maximum(output[channel], _dual_symmetry_map(output[channel]))
    for channel in ENTITY_SYMMETRY_CHANNELS:
        output[channel] = np.maximum(output[channel], _center_symmetry_map(output[channel]))
    return output


def backcast_terrain_projections(steps, side, speed, width=24, height=24):
    """Use a hindsight speed, but only terrain observed through each frame.

    The known map is rolled with the terrain before each new public observation.
    Unseen tiles remain unknown; the four output planes mark known positive tiles.
    """
    known_tile = np.full((width, height), -1, dtype=np.int8)
    previous_step = None
    current_planes, future_planes = [], []
    direction = TileDriftEstimator._shift_for_speed(speed)
    for frame in steps:
        obs = _as_obs(frame[side])
        step = int(obs["steps"])
        if previous_step is not None:
            if step <= previous_step:
                raise ValueError(f"non-increasing observation step {step}")
            shifts = sum(TileDriftEstimator._moves_at_step(speed, t)
                         for t in range(previous_step + 1, step + 1))
            known_tile = np.roll(known_tile, shift=(direction[0] * shifts,
                                                    direction[1] * shifts), axis=(0, 1))
        terrain = np.asarray(obs["map_features"]["tile_type"], dtype=np.int8)
        visible = np.asarray(obs["sensor_mask"], dtype=bool) & (terrain >= 0)
        known_tile[visible] = terrain[visible]
        mirrored_visible = _dual_symmetry_map(visible)
        mirrored_terrain = _dual_symmetry_map(terrain)
        known_tile[mirrored_visible] = mirrored_terrain[mirrored_visible]
        current_planes.append(np.stack((known_tile == 0, known_tile == 1,
                                        known_tile == 2)).astype(np.float32))

        horizon_planes = []
        for horizon in (5, 10):
            shifts = sum(TileDriftEstimator._moves_at_step(speed, t)
                         for t in range(step + 1, step + horizon + 1))
            projected = np.roll(known_tile, shift=(direction[0] * shifts,
                                                   direction[1] * shifts), axis=(0, 1))
            horizon_planes.extend((projected == 1, projected == 2))
        future_planes.append(np.stack(horizon_planes).astype(np.float32))
        previous_step = step
    return np.stack(current_planes), np.stack(future_planes)


def _as_obs(record):
    payload = record.get("observation", {})
    raw = payload.get("obs", payload)
    if isinstance(raw, str):
        raw = json.loads(raw)
    return raw


def _as_action(record, max_units):
    raw = record.get("action", [])
    if isinstance(raw, dict):
        raw = raw.get("action", [])
    result = np.zeros((max_units, 3), dtype=np.int16)
    if raw is None or raw == {} or (isinstance(raw, (list, tuple)) and not raw):
        return result
    values = np.asarray(raw, dtype=np.int16)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"expected action shape (K, 3), got {values.shape}")
    result[:min(max_units, values.shape[0])] = values[:max_units]
    return result


def _units(obs, team_id, width, height):
    mask = np.asarray(obs["units_mask"], dtype=bool)
    positions = np.asarray(obs["units"]["position"], dtype=np.int16)
    energies = np.asarray(obs["units"]["energy"], dtype=np.int16).reshape(mask.shape)
    output = []
    for unit_id in np.flatnonzero(mask[team_id]):
        x, y = map(int, positions[team_id, unit_id])
        if energies[team_id, unit_id] >= 0 and 0 <= x < width and 0 <= y < height:
            output.append((int(unit_id), (x, y), int(energies[team_id, unit_id])))
    return output


class ReplayReconstructor:
    """Rebuild only public, player-specific state and V2 belief features."""

    def __init__(self, team_id, width=24, height=24, max_units=16,
                 max_steps_in_match=100):
        self.team_id = team_id
        self.width, self.height, self.max_units = width, height, max_units
        self.solver = EquationSolver(width, height)
        self.parser = ObsParser(solver=self.solver, team_id=team_id,
                                max_steps_in_match=max_steps_in_match)
        self.tile_type = np.full((width, height), -1, dtype=np.int8)
        self.last_seen = np.full((width, height), -1, dtype=np.int32)
        self.energy_field = np.zeros((width, height), dtype=np.float32)
        self.energy_seen = np.full((width, height), -1, dtype=np.int32)
        self.known_relics = {}
        self.parser_errors = 0
        self.previous_occupied = np.zeros((width, height), dtype=bool)
        self.current_occupied = np.zeros((width, height), dtype=bool)
        self.last_equation_key = None
        self.match_id = 0
        self.tile_drift = TileDriftEstimator(width, height)
        self.last_drift = None

    def _update_memory(self, step, obs):
        visible = np.asarray(obs["sensor_mask"], dtype=bool)
        terrain = np.asarray(obs["map_features"]["tile_type"], dtype=np.int8)
        energy = np.asarray(obs["map_features"]["energy"], dtype=np.float32)
        valid = visible & (terrain >= 0)
        self.tile_type[valid] = terrain[valid]
        self.last_seen[valid] = step
        self.energy_field[visible] = energy[visible]
        self.energy_seen[visible] = step
        relic_mask = np.asarray(obs["relic_nodes_mask"], dtype=bool)
        relic_positions = np.asarray(obs["relic_nodes"], dtype=np.int16)
        for relic_id in np.flatnonzero(relic_mask):
            x, y = map(int, relic_positions[relic_id])
            if 0 <= x < self.width and 0 <= y < self.height:
                self.known_relics[int(relic_id)] = (x, y)

    def _feature(self, step, obs):
        terrain_observation = np.asarray(obs["map_features"]["tile_type"], dtype=np.int8)
        sensor_observation = np.asarray(obs["sensor_mask"], dtype=bool)
        self.last_drift = self.tile_drift.observe(
            step, terrain_observation, sensor_observation
        )
        self._update_memory(step, obs)
        previous = self.current_occupied.copy() if int(obs["match_steps"]) else np.zeros_like(self.current_occupied)
        self.previous_occupied = previous
        self.current_occupied.fill(False)
        own = _units(obs, self.team_id, self.width, self.height)
        occupied = frozenset(position for _, position, _ in own)
        for _, (x, y), _ in own:
            self.current_occupied[x, y] = True
        points = int(np.asarray(obs["team_points"])[self.team_id])
        previous_points = self.parser.last_points
        delta = None if previous_points is None else points - previous_points
        match_step = int(obs["match_steps"])
        if (self.parser.last_match_step is not None
                and match_step < self.parser.last_match_step):
            self.match_id += 1
        epoch = "dynamic" if step < self.parser.stable_from_step else "static"
        equation_key = (epoch, self.match_id, occupied, delta)
        duplicate = previous_points is not None and equation_key == self.last_equation_key
        if duplicate:
            self.parser.last_step = step
            self.parser.last_match_step = match_step
            self.parser.last_points = points
            self.parser.last_update = None
            self.parser.last_reason = "duplicate_equation"
        else:
            try:
                self.parser.observe(obs)
            except (KeyError, TypeError, ValueError):
                self.parser_errors += 1
                self.parser.last_update = None
                self.parser.last_reason = "parser_error"
            if delta is not None and match_step > 0 and points >= previous_points:
                self.last_equation_key = equation_key

        visible = np.asarray(obs["sensor_mask"], dtype=np.float32)
        seen = self.last_seen >= 0
        age = np.where(seen, np.minimum(step - self.last_seen, 100) / 100, 1)
        support = np.zeros((self.width, self.height), dtype=np.float32)
        for x, y in self.known_relics.values():
            support[max(0, x - 2):min(self.width, x + 3),
                    max(0, y - 2):min(self.height, y + 3)] = 1
        enemy = _units(obs, 1 - self.team_id, self.width, self.height)
        own_count = np.zeros((self.width, self.height), dtype=np.float32)
        own_energy = np.zeros_like(own_count)
        enemy_count = np.zeros_like(own_count)
        enemy_energy = np.zeros_like(own_count)
        for _, (x, y), energy in own:
            own_count[x, y] += 1
            own_energy[x, y] += energy / 400
        for _, (x, y), energy in enemy:
            enemy_count[x, y] += 1
            enemy_energy[x, y] += energy / 400
        probability = self.solver.feature_map()
        confidence = np.zeros_like(probability)
        for x, y in self.solver.values:
            confidence[x, y] = 1
        terrain = self.tile_type
        energy_visible = np.where(self.energy_seen >= 0,
                                  self.energy_field / 20, 0)
        features = np.stack((
            visible, ~seen, age,
            visible * (terrain == 0), visible * (terrain == 1),
            visible * (terrain == 2),
            seen * (terrain == 0), seen * (terrain == 1), seen * (terrain == 2),
            energy_visible, support, probability, confidence,
            own_count / 16, own_energy / 16, enemy_count / 16, enemy_energy / 16,
            self.previous_occupied.astype(np.float32),
            self.current_occupied.astype(np.float32),
            (probability == 0), (probability == 1),
        )).astype(np.float32)
        features = apply_dual_symmetry(features)
        if features.shape != (len(BASE_CHANNEL_NAMES), self.width, self.height):
            raise AssertionError(f"unexpected feature shape {features.shape}")
        return features, own


def infer_intents(positions, active, actions, match_steps, width, height,
                  horizon=12, enemy_occupied=None, observation_steps=None):
    """Infer a stable destination from a short future action/position window.

    Labels require at least one observed move and terminate at a death, match
    reset, or horizon boundary. They are behavior-cloning heuristics, not
    direct ground-truth macro targets.
    """
    frames, max_units = active.shape
    target = np.zeros((frames, max_units, width, height), dtype=np.uint8)
    valid = np.zeros((frames, max_units), dtype=bool)
    for frame in range(frames):
        for unit_id in np.flatnonzero(active[frame]):
            future = []
            saw_move = False
            previous = tuple(positions[frame, unit_id])
            for index in range(frame, min(frames, frame + horizon + 1)):
                if (index > frame and observation_steps is not None
                        and observation_steps[index] != observation_steps[index - 1] + 1):
                    break
                if index > frame and match_steps[index] < match_steps[index - 1]:
                    break
                if index > frame and not active[index, unit_id]:
                    break
                if actions[index, unit_id, 0] in (1, 2, 3, 4):
                    saw_move = True
                if index > frame:
                    current = tuple(positions[index, unit_id])
                    if current != previous:
                        saw_move = True
                    future.append(current)
                    previous = current
            if not saw_move or len(future) < 2:
                continue
            x0, y0 = map(int, positions[frame, unit_id])
            displacement = max(abs(int(x) - x0) + abs(int(y) - y0)
                               for x, y in future)
            blocked = False
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                x1, y1 = x0 + dx, y0 + dy
                if not (0 <= x1 < width and 0 <= y1 < height) or enemy_occupied is not None and enemy_occupied[frame, x1, y1]:
                    blocked = True
            if displacement < 3 and not blocked:
                continue
            counts = {}
            for tile in future:
                counts[tile] = counts.get(tile, 0) + 1
            best_count = max(counts.values())
            candidates = {tile for tile, count in counts.items() if count == best_count}
            chosen = next(tile for tile in reversed(future) if tile in candidates)
            x, y = chosen
            if 0 <= x < width and 0 <= y < height:
                target[frame, unit_id, x, y] = 1
                valid[frame, unit_id] = True
    return target, valid


def parse_replay(path, output_dir, width=24, height=24, max_units=16,
                 horizon=12, overwrite=False, sides=(0, 1)):
    data = json.loads(path.read_text(encoding="utf-8"))
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{path}: missing non-empty steps list")
    if any(not isinstance(frame, list) or len(frame) < 2 for frame in steps):
        raise ValueError(f"{path}: expected two player records per frame")

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for side in sides:
        reconstructor = None
        feature_rows, action_rows, position_rows = [], [], []
        active_rows, step_rows, match_rows, point_rows = [], [], [], []
        drift_candidate_rows = []
        enemy_rows = []
        for frame in steps:
            record = frame[side]
            obs = _as_obs(record)
            if reconstructor is None:
                map_features = np.asarray(obs["map_features"]["tile_type"])
                if map_features.shape != (width, height):
                    raise ValueError(f"{path}: expected {(width, height)} map, got {map_features.shape}")
                reconstructor = ReplayReconstructor(
                    side, width, height, max_units,
                    max_steps_in_match=100,
                )
            step = int(obs["steps"])
            features, own = reconstructor._feature(step, obs)
            enemy_mask = np.zeros((width, height), dtype=bool)
            for _, (x, y), _ in _units(obs, 1 - side, width, height):
                enemy_mask[x, y] = True
            enemy_rows.append(enemy_mask)
            feature_rows.append(features)
            drift_candidate_rows.append(reconstructor.last_drift["candidate_count"])
            action_rows.append(_as_action(record, max_units))
            positions = np.full((max_units, 2), -1, dtype=np.int16)
            active = np.zeros(max_units, dtype=bool)
            for unit_id, position, _ in own:
                positions[unit_id] = position
                active[unit_id] = True
            position_rows.append(positions)
            active_rows.append(active)
            step_rows.append(step)
            match_rows.append(int(obs["match_steps"]))
            point_rows.append(np.asarray(obs["team_points"], dtype=np.int16))

        if len(reconstructor.tile_drift.candidate_speeds) != 1:
            raise ValueError(f"{path} side {side}: drift speed did not converge")
        if any(record["has_exact_candidate"] is False
               for record in reconstructor.tile_drift.history):
            raise ValueError(f"{path} side {side}: observed terrain conflicts with drift model")
        confirmed_speed = reconstructor.tile_drift.candidate_speeds[0]
        current_terrain, future_terrain = backcast_terrain_projections(
            steps, side, confirmed_speed, width, height
        )
        features = np.stack(feature_rows)
        features[:, 6:9] = current_terrain
        features = np.concatenate((features, future_terrain), axis=1)
        if features.shape[1:] != (len(CHANNEL_NAMES), width, height):
            raise AssertionError(f"unexpected backcast feature shape {features.shape}")
        actions = np.stack(action_rows)
        positions = np.stack(position_rows)
        active = np.stack(active_rows)
        match_steps = np.asarray(match_rows, dtype=np.int16)
        intent_target, intent_valid = infer_intents(
            positions, active, actions, match_steps, width, height, horizon,
            enemy_occupied=np.stack(enemy_rows), observation_steps=np.asarray(step_rows),
        )
        output = output_dir / f"{path.stem}_p{side}.npz"
        if output.exists() and not overwrite:
            raise FileExistsError(f"{output}; pass --overwrite to replace")
        temporary = output.with_suffix(".npz.part")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, schema_version=np.asarray(SCHEMA_VERSION, dtype=np.uint8),
                **pack_features(features), intent_target=intent_target,
                intent_valid=intent_valid, actions=actions, positions=positions,
                active=active, steps=np.asarray(step_rows, dtype=np.int32),
                match_steps=match_steps, team_points=np.stack(point_rows),
                drift_candidate_count=np.asarray(drift_candidate_rows, dtype=np.int8),
            )
        temporary.replace(output)
        summaries.append({
            "source": str(path), "output": str(output), "side": side,
            "frames": len(feature_rows), "feature_shape": list(features.shape),
            "intent_shape": list(intent_target.shape),
            "valid_intent_labels": int(intent_valid.sum()),
            "active_decisions": int(active.sum()),
            "intent_valid_rate": float(intent_valid.sum() / max(1, active.sum())),
            "parser_errors": reconstructor.parser_errors,
            "static_rank": reconstructor.solver.rank,
            "static_values": len(reconstructor.solver.values),
            "dynamic_rows_final": len(reconstructor.solver.dynamic_pool),
            "drift_candidate_speeds": list(reconstructor.tile_drift.candidate_speeds),
            "drift_confirmed_speed": confirmed_speed,
            "drift_converged_step": reconstructor.tile_drift.converged_step,
            "drift_first_reduction_step": next(
                (record["step"] for record in reconstructor.tile_drift.history
                 if record["candidate_count"] < len(NEBULA_DRIFT_SPEEDS)),
                None,
            ),
            "drift_min_candidate_count": min(
                record["candidate_count"] for record in reconstructor.tile_drift.history
            ),
            "drift_conflicting_frames": sum(
                record["has_exact_candidate"] is False
                for record in reconstructor.tile_drift.history
            ),
        })
    return summaries


class OmniscientParser:
    """Historical name; consumes player observations and never hidden replay state."""

    def __init__(self, horizon=12):
        self.horizon = horizon

    def parse(self, path, output_dir, *, overwrite=False, sides=(0, 1)):
        return parse_replay(Path(path), Path(output_dir), horizon=self.horizon,
                            overwrite=overwrite, sides=sides)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "data" / "raw_replays")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data" / "processed_replays_unitcentric25")
    parser.add_argument("--source-manifest", type=Path,
                        help="select replay names and verify hashes from an existing manifest")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.horizon < 2:
        parser.error("horizon must be at least 2")
    selected_hashes = None
    if args.source_manifest:
        selected_hashes = json.loads(args.source_manifest.read_text())["source_sha256"]
        paths = [args.input_dir / name for name in sorted(selected_hashes)]
        for path in paths:
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != selected_hashes[path.name]:
                parser.error(f"missing or changed replay: {path}")
    else:
        paths = sorted(args.input_dir.glob("*.json"))
    if args.limit is not None:
        paths = paths[:args.limit]
    if not paths:
        parser.error(f"no JSON replays found in {args.input_dir}")
    summaries = []
    for path in paths:
        print(f"processing {path.name}", flush=True)
        summaries.extend(parse_replay(path, args.output_dir, horizon=args.horizon,
                                       overwrite=args.overwrite))
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    manifest = dataset_manifest(summaries, source_hashes, args.horizon)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
