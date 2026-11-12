"""Observation-only v2 agent with epoch-isolated inference and reserved routing."""

import numpy as np
import time

from dispatcher import Dispatcher
from equation_solver import EquationSolver
from obs_parser import ObsParser
from router import Router


class Agent:
    def __init__(self, player, env_cfg):
        self.team_id = 0 if player == "player_0" else 1
        self.width = int(env_cfg["map_width"])
        self.height = int(env_cfg["map_height"])
        self.max_units = int(env_cfg["max_units"])
        self.max_steps_in_match = int(env_cfg["max_steps_in_match"])
        self.sensor_range = int(env_cfg["unit_sensor_range"])
        self.solver = EquationSolver(self.width, self.height)
        self.parser = ObsParser(self.solver, self.team_id, self.max_steps_in_match)
        self.dispatcher = Dispatcher(self.solver)
        self.router = Router(self.width, self.height, self.max_units,
                             int(env_cfg["unit_move_cost"]))
        self.tile_type = np.full((self.width, self.height), -1, dtype=np.int8)
        self.last_seen = np.full((self.width, self.height), -1, dtype=np.int32)
        self.known_relics = {}
        self.contradictions = 0
        self.last_trace = None

    def act(self, step, obs, remainingOverageTime=60):
        started = time.perf_counter()
        visible = np.asarray(obs["sensor_mask"], dtype=bool)
        terrain = np.asarray(obs["map_features"]["tile_type"])
        valid = visible & (terrain >= 0)
        self.tile_type[valid] = terrain[valid]
        self.last_seen[valid] = step
        try:
            self.parser.observe(obs)
        except ValueError:
            self.contradictions += 1
            self.parser.last_update = None
            self.parser.last_reason = "contradiction"

        relic_mask = np.asarray(obs["relic_nodes_mask"], dtype=bool)
        relics = np.asarray(obs["relic_nodes"])
        for relic_id in np.flatnonzero(relic_mask):
            x, y = map(int, relics[relic_id])
            if 0 <= x < self.width and 0 <= y < self.height:
                self.known_relics[int(relic_id)] = (x, y)
        support = np.zeros((self.width, self.height), dtype=bool)
        for x, y in self.known_relics.values():
            support[max(0, x - 2):min(self.width, x + 3),
                    max(0, y - 2):min(self.height, y + 3)] = True

        mask = np.asarray(obs["units_mask"], dtype=bool)
        positions = np.asarray(obs["units"]["position"])
        energies = np.asarray(obs["units"]["energy"]).reshape(mask.shape)
        own = []
        for unit_id in np.flatnonzero(mask[self.team_id]):
            x, y = map(int, positions[self.team_id, unit_id])
            if energies[self.team_id, unit_id] >= 0 and 0 <= x < self.width and 0 <= y < self.height:
                own.append((int(unit_id), (x, y), int(energies[self.team_id, unit_id])))
        enemy = []
        for unit_id in np.flatnonzero(mask[1 - self.team_id]):
            x, y = map(int, positions[1 - self.team_id, unit_id])
            if 0 <= x < self.width and 0 <= y < self.height:
                enemy.append((x, y))

        blocked = (self.tile_type == 2) & (self.last_seen >= step - 2)
        for tile in enemy:
            blocked[tile] = True
        passable = ~blocked
        compact_units = [(unit_id, position) for unit_id, position, _ in own]
        macro = self.dispatcher.assign_macro(
            compact_units, passable, self.last_seen < 0, support,
            int(obs["match_steps"]), self.max_steps_in_match, self.sensor_range,
        )
        targets = self.dispatcher.plan(
            compact_units, self.parser.last_update, passable,
            enemy_positions=enemy, base_targets=macro,
        )
        probability = self.solver.feature_map()
        priorities = {unit_id: float(probability[target]) for unit_id, target in targets.items()}
        actions = self.router.route(own, targets, blocked, priorities)
        self.last_trace = {
            "step": step,
            "match_step": int(obs["match_steps"]),
            "rank": self.solver.rank,
            "dynamic_rows": len(self.solver.dynamic_pool),
            "static_rows": len(self.solver.static_pool),
            "parser_reason": self.parser.last_reason,
            "contradictions": self.contradictions,
            "decision_seconds": time.perf_counter() - started,
            "yielded_unit": self.dispatcher.last_yield_unit,
            "independent": (self.parser.last_update.independent
                            if self.parser.last_update is not None else None),
            "positions": {str(unit_id): position for unit_id, position, _ in own},
            "targets": {str(unit_id): target for unit_id, target in targets.items()},
            "actions": {str(unit_id): int(actions[unit_id, 0])
                        for unit_id, _, _ in own},
        }
        return actions
