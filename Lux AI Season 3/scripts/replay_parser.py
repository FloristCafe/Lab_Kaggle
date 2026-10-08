"""Build observation-only tensors and heuristic macro-intent labels from Kaggle replays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.v2.equation_solver import EquationSolver
from agents.v2.obs_parser import ObsParser


CHANNEL_NAMES = (
    "visible", "unseen", "observation_age", "visible_empty",
    "visible_nebula", "visible_asteroid", "last_empty", "last_nebula",
    "last_asteroid", "visible_energy", "known_relic_support",
    "score_probability", "score_confidence", "own_count", "own_energy",
    "enemy_count", "enemy_energy", "previous_occupied", "current_occupied",
    "known_safe_zero", "known_score_one",
)


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
        if features.shape != (len(CHANNEL_NAMES), self.width, self.height):
            raise AssertionError(f"unexpected feature shape {features.shape}")
        return features, own


def infer_intents(positions, active, actions, match_steps, width, height,
                  horizon=12):
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
            for index in range(frame, min(frames, frame + horizon)):
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
                 horizon=12, overwrite=False):
    data = json.loads(path.read_text(encoding="utf-8"))
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{path}: missing non-empty steps list")
    if any(not isinstance(frame, list) or len(frame) < 2 for frame in steps):
        raise ValueError(f"{path}: expected two player records per frame")

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for side in (0, 1):
        reconstructor = None
        feature_rows, action_rows, position_rows = [], [], []
        active_rows, step_rows, match_rows, point_rows = [], [], [], []
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
            feature_rows.append(features)
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

        features = np.stack(feature_rows)
        actions = np.stack(action_rows)
        positions = np.stack(position_rows)
        active = np.stack(active_rows)
        match_steps = np.asarray(match_rows, dtype=np.int16)
        intent_target, intent_valid = infer_intents(
            positions, active, actions, match_steps, width, height, horizon
        )
        output = output_dir / f"{path.stem}_p{side}.npz"
        if output.exists() and not overwrite:
            raise FileExistsError(f"{output}; pass --overwrite to replace")
        np.savez_compressed(
            output, features=features, intent_target=intent_target,
            intent_valid=intent_valid, actions=actions, positions=positions,
            active=active, steps=np.asarray(step_rows, dtype=np.int32),
            match_steps=match_steps, team_points=np.stack(point_rows),
        )
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
        })
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "data" / "raw_replays")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data" / "processed_replays")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.horizon < 2:
        parser.error("horizon must be at least 2")
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
    manifest = {
        "schema_version": 1,
        "parser_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "channel_names": list(CHANNEL_NAMES),
        "intent_label_method": "future_position_mode_with_move_evidence",
        "intent_horizon": args.horizon,
        "observation_source": "steps[*][side].observation.obs",
        "hidden_state_fields_used": [],
        "source_sha256": source_hashes,
        "replays": summaries,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
