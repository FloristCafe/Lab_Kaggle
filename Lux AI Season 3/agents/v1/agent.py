"""Observation-only v1 teacher: relic inference and joint target dispatch."""

import os

import numpy as np

from dispatcher import Dispatcher
from inference import RelicInference


FEATURE_NAMES = (
    "visible", "never_seen", "observation_age", "visible_empty",
    "visible_nebula", "visible_asteroid", "last_empty", "last_nebula",
    "last_asteroid", "visible_energy", "known_relic_center",
    "relic_support", "score_probability", "score_confidence", "own_count",
    "own_energy", "enemy_count", "enemy_energy", "enemy_proximity",
    "own_proximity", "previous_target_count", "previous_occupied",
)


class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.team_id = 0 if player == "player_0" else 1
        self.width = int(env_cfg["map_width"])
        self.height = int(env_cfg["map_height"])
        self.max_units = int(env_cfg["max_units"])
        self.move_cost = int(env_cfg["unit_move_cost"])
        self.max_steps_in_match = int(env_cfg.get("max_steps_in_match", 100))
        self.tile_type = np.full((self.width, self.height), -1, dtype=np.int8)
        self.last_seen = np.full((self.width, self.height), -1, dtype=np.int32)
        self.energy_field = np.zeros((self.width, self.height), dtype=np.float32)
        self.energy_seen = np.full((self.width, self.height), -1, dtype=np.int32)
        self.inference = RelicInference(
            self.width, self.height, self.max_steps_in_match
        )
        self.dispatcher = Dispatcher(
            self.width, self.height, self.max_units, self.move_cost,
            int(env_cfg["unit_sensor_range"]), self.max_steps_in_match,
        )
        fixed_target = os.environ.get("LUX_V1_FIXED_TARGET")
        if fixed_target:
            x, y = (int(value) for value in fixed_target.split(","))
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError("LUX_V1_FIXED_TARGET must be inside the map")
            self.dispatcher.fixed_target = (x, y)
        self.record = bool(os.environ.get("LUX_V1_TRACE_DIR"))
        self.last_trace = None

    def _features(self, step, visible, own, enemy, probability, confidence):
        width, height = self.width, self.height
        grid_x, grid_y = np.indices((width, height))
        own_count = np.zeros((width, height), dtype=np.float32)
        own_energy = np.zeros_like(own_count)
        enemy_count = np.zeros_like(own_count)
        enemy_energy = np.zeros_like(own_count)
        proximity = np.zeros_like(own_count)
        own_proximity = np.zeros_like(own_count)
        previous_target_count = np.zeros_like(own_count)
        for _, (x, y), energy in own:
            own_count[x, y] += 1
            own_energy[x, y] += energy / 400
            distance = np.abs(grid_x - x) + np.abs(grid_y - y)
            own_proximity += np.maximum(0, 3 - distance) / 3
        for target in self.dispatcher.previous_targets.values():
            previous_target_count[target] += 1
        for x, y, energy in enemy:
            enemy_count[x, y] += 1
            enemy_energy[x, y] += energy / 400
            distance = np.abs(grid_x - x) + np.abs(grid_y - y)
            proximity = np.maximum(proximity, np.maximum(0, 4 - distance) / 4)
        seen = self.last_seen >= 0
        age = np.where(seen, np.minimum(step - self.last_seen, 100) / 100, 1)
        centers = np.zeros((width, height), dtype=np.float32)
        for center in self.inference.known_relics.values():
            centers[center] = 1
        return np.stack((
            visible, ~seen, age,
            visible & (self.tile_type == 0),
            visible & (self.tile_type == 1),
            visible & (self.tile_type == 2),
            seen & (self.tile_type == 0),
            seen & (self.tile_type == 1),
            seen & (self.tile_type == 2),
            np.where(visible, self.energy_field / 20, 0),
            centers, self.inference.support(), probability, confidence,
            own_count / 16, own_energy / 16, enemy_count / 16,
            enemy_energy / 16, proximity, own_proximity / 16,
            previous_target_count / 16, self.inference.previous_occupied,
        )).astype(np.float16)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        visible = np.asarray(obs["sensor_mask"], dtype=bool)
        terrain = np.asarray(obs["map_features"]["tile_type"])
        energy_map = np.asarray(obs["map_features"]["energy"])
        valid = visible & (terrain >= 0)
        self.tile_type[valid] = terrain[valid]
        self.last_seen[valid] = step
        self.energy_field[visible] = energy_map[visible]
        self.energy_seen[visible] = step

        self.inference.update(step, obs, self.team_id)
        unit_mask = np.asarray(obs["units_mask"], dtype=bool)
        positions = np.asarray(obs["units"]["position"])
        energies = np.asarray(obs["units"]["energy"]).reshape(2, -1)
        own = []
        for unit_id in np.flatnonzero(unit_mask[self.team_id]):
            x, y = map(int, positions[self.team_id, unit_id])
            if 0 <= x < self.width and 0 <= y < self.height:
                own.append((int(unit_id), (x, y), int(energies[self.team_id, unit_id])))
        other = 1 - self.team_id
        enemy = []
        for unit_id in np.flatnonzero(unit_mask[other]):
            x, y = map(int, positions[other, unit_id])
            if 0 <= x < self.width and 0 <= y < self.height:
                enemy.append((x, y, int(energies[other, unit_id])))

        probability = self.inference.probability()
        confidence = self.inference.confidence()
        previous_targets = self.dispatcher.previous_targets.copy()
        features = None
        if self.record:
            features = self._features(step, visible, own, enemy, probability, confidence)
        actions, targets, reach, costs, energy_costs = self.dispatcher.plan(
            step, int(obs["match_steps"]), own, self.tile_type, self.last_seen,
            self.energy_field, self.energy_seen, enemy, probability, confidence,
            previous_occupied=self.inference.previous_occupied,
        )
        if self.record:
            unit_features = np.full((self.max_units, 9), -1, dtype=np.float16)
            active = np.zeros(self.max_units, dtype=bool)
            for unit_id, (x, y), energy in own:
                previous = previous_targets.get(unit_id, (-1, -1))
                nearest = min(
                    enemy,
                    key=lambda item: abs(item[0] - x) + abs(item[1] - y),
                    default=None,
                )
                relative = (-1, -1, -1, 0)
                if nearest is not None:
                    distance = abs(nearest[0] - x) + abs(nearest[1] - y)
                    relative = (
                        (nearest[0] - x) / self.width,
                        (nearest[1] - y) / self.height,
                        distance / (self.width + self.height), 1,
                    )
                unit_features[unit_id] = (
                    x / self.width, y / self.height, energy / 400,
                    previous[0] / self.width, previous[1] / self.height,
                    *relative,
                )
                active[unit_id] = True
            label_valid = np.zeros(self.max_units, dtype=bool)
            for unit_id, _, _ in own:
                tx, ty = targets[unit_id]
                label_valid[unit_id] = reach[unit_id, tx, ty]
            self.last_trace = {
                "global_features": features,
                "unit_features": unit_features,
                "active": active,
                "label_valid": label_valid,
                "target": targets,
                "action": actions,
                "reachable": reach,
                "path_cost": costs.astype(np.float16),
                "energy_cost": energy_costs.astype(np.float16),
                "step": np.array(step, dtype=np.int32),
                "match_step": np.array(int(obs["match_steps"]), dtype=np.int16),
                "points": np.asarray(obs["team_points"], dtype=np.int16),
                "wins": np.asarray(obs["team_wins"], dtype=np.int16),
                "team_id": np.array(self.team_id, dtype=np.int8),
                "probe_mode": np.array(self.dispatcher.fixed_target is not None),
            }
        return actions
