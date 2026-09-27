from collections import deque

import numpy as np


MOVES = ((0, -1, 1), (1, 0, 2), (0, 1, 3), (-1, 0, 4))


class Agent:
    """Observation-only exploration and relic-area coverage baseline."""

    def __init__(self, player: str, env_cfg) -> None:
        self.team_id = 0 if player == "player_0" else 1
        self.width = int(env_cfg["map_width"])
        self.height = int(env_cfg["map_height"])
        self.max_units = int(env_cfg["max_units"])
        self.move_cost = int(env_cfg["unit_move_cost"])
        self.sensor_range = int(env_cfg["unit_sensor_range"])
        self.tile_type = np.full((self.width, self.height), -1, dtype=np.int8)
        self.last_seen = np.full((self.width, self.height), -1, dtype=np.int32)
        self.relics = {}
        self.targets = {}

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _paths_from(self, start):
        distances = {start: 0}
        first_moves = {start: 0}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy, action in MOVES:
                neighbor = (x + dx, y + dy)
                if not self._in_bounds(*neighbor) or neighbor in distances:
                    continue
                if self.tile_type[neighbor] == 2:
                    continue
                distances[neighbor] = distances[(x, y)] + 1
                first_moves[neighbor] = action if (x, y) == start else first_moves[(x, y)]
                queue.append(neighbor)
        return distances, first_moves

    def _relic_candidates(self):
        candidates = set()
        for rx, ry in self.relics.values():
            for x in range(max(0, rx - 2), min(self.width, rx + 3)):
                for y in range(max(0, ry - 2), min(self.height, ry + 3)):
                    if self.tile_type[x, y] != 2:
                        candidates.add((x, y))
        return candidates

    def _exploration_value(self, target):
        x, y = target
        radius = self.sensor_range
        unseen = 0
        for nx in range(max(0, x - radius), min(self.width, x + radius + 1)):
            for ny in range(max(0, y - radius), min(self.height, y + radius + 1)):
                unseen += self.last_seen[nx, ny] < 0
        return unseen

    def _choose_target(self, unit_id, position, distances, reserved, relic_candidates):
        previous = self.targets.get(unit_id)
        if relic_candidates:
            if previous in relic_candidates and previous in distances and previous not in reserved:
                return previous
            return min(
                (tile for tile in relic_candidates if tile in distances),
                key=lambda tile: (
                    distances[tile] + 12 * (tile in reserved),
                    -self._exploration_value(tile),
                    tile,
                ),
                default=None,
            )

        if previous in distances and self.last_seen[previous] < 0 and previous not in reserved:
            return previous
        unknown = (tile for tile in distances if self.last_seen[tile] < 0)
        return min(
            unknown,
            key=lambda tile: (
                distances[tile] + 8 * any(
                    abs(tile[0] - other[0]) + abs(tile[1] - other[1]) <= self.sensor_range
                    for other in reserved
                ) - 0.15 * self._exploration_value(tile),
                tile,
            ),
            default=None,
        )

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = np.zeros((self.max_units, 3), dtype=np.int32)
        visible = np.asarray(obs["sensor_mask"], dtype=bool)
        observed_tiles = np.asarray(obs["map_features"]["tile_type"])
        valid = visible & (observed_tiles >= 0)
        self.tile_type[valid] = observed_tiles[valid]
        self.last_seen[valid] = step

        relic_mask = np.asarray(obs["relic_nodes_mask"], dtype=bool)
        relic_positions = np.asarray(obs["relic_nodes"])
        for relic_id in np.flatnonzero(relic_mask):
            pos = tuple(map(int, relic_positions[relic_id]))
            if self._in_bounds(*pos):
                self.relics[int(relic_id)] = pos

        unit_mask = np.asarray(obs["units_mask"][self.team_id], dtype=bool)
        positions = np.asarray(obs["units"]["position"][self.team_id])
        energies = np.asarray(obs["units"]["energy"][self.team_id]).reshape(-1)
        relic_candidates = self._relic_candidates()
        reserved = set()

        for unit_id in np.flatnonzero(unit_mask):
            position = tuple(map(int, positions[unit_id]))
            if not self._in_bounds(*position):
                continue
            distances, first_moves = self._paths_from(position)
            target = self._choose_target(
                int(unit_id), position, distances, reserved, relic_candidates
            )
            if target is None:
                self.targets.pop(int(unit_id), None)
                continue
            self.targets[int(unit_id)] = target
            reserved.add(target)
            if energies[unit_id] >= self.move_cost:
                actions[unit_id, 0] = first_moves[target]
        return actions
