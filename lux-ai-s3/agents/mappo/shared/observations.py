"""Causal runtime subset of the legacy replay reconstructor."""
import numpy as np

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
        if features.shape != (len(BASE_CHANNEL_NAMES), self.width, self.height):
            raise AssertionError(f"unexpected feature shape {features.shape}")
        return features, own

