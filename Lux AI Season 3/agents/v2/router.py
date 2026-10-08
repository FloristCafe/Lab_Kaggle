"""Prioritized space-time routing for one-step Lux movement actions."""

import heapq

import numpy as np


MOVES = ((0, 0, 0), (0, -1, 1), (1, 0, 2),
         (0, 1, 3), (-1, 0, 4))


class Router:
    def __init__(self, width, height, max_units, move_cost, horizon=10):
        self.width, self.height = width, height
        self.max_units, self.move_cost = max_units, move_cost
        self.horizon = horizon
        self.reservations = np.zeros((horizon + 1, width, height), dtype=bool)
        self.routes = {}

    def _search(self, start, goal, blocked, occupied, edges):
        start_state = (*start, 0)
        queue = [(abs(start[0] - goal[0]) + abs(start[1] - goal[1]),
                  0, start[0], start[1])]
        parent = {start_state: None}
        best = start_state
        best_key = (abs(start[0] - goal[0]) + abs(start[1] - goal[1]), 0)
        while queue:
            _, t, x, y = heapq.heappop(queue)
            distance = abs(x - goal[0]) + abs(y - goal[1])
            can_hold = not self.reservations[t:, x, y].any()
            if can_hold and (distance, t) < best_key:
                best, best_key = (x, y, t), (distance, t)
            if distance == 0 or t == self.horizon:
                if distance == 0 and can_hold:
                    best = (x, y, t)
                    break
                if t == self.horizon:
                    continue
            for dx, dy, _ in MOVES:
                nx, ny, nt = x + dx, y + dy, t + 1
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if blocked[nx, ny] or (nx, ny) in occupied:
                    continue
                if self.reservations[nt, nx, ny]:
                    continue
                if ((nx, ny), (x, y), nt) in edges:
                    continue
                node = (nx, ny, nt)
                if node in parent:
                    continue
                parent[node] = (x, y, t)
                heuristic = abs(nx - goal[0]) + abs(ny - goal[1])
                heapq.heappush(queue, (nt + heuristic, nt, nx, ny))
        path = []
        while best is not None:
            path.append((best[0], best[1]))
            best = parent[best]
        return list(reversed(path))

    def route(self, units, targets, blocked, priorities=None):
        """Units are (id, position, energy); return the official action array."""
        self.reservations.fill(False)
        self.routes = {}
        actions = np.zeros((self.max_units, 3), dtype=np.int32)
        priorities = priorities or {}
        starts = {unit_id: tuple(position) for unit_id, position, _ in units}
        edges = set()
        immobile = {unit_id for unit_id, _, energy in units
                    if energy < self.move_cost}
        for unit_id in immobile:
            self.reservations[:, starts[unit_id][0], starts[unit_id][1]] = True
            self.routes[unit_id] = [starts[unit_id]] * (self.horizon + 1)
        ordered = sorted((unit for unit in units if unit[0] not in immobile),
                         key=lambda unit: (-priorities.get(unit[0], 0), unit[0]))
        for unit_id, position, _ in ordered:
            start = tuple(position)
            goal = tuple(targets.get(unit_id, start))
            occupied = set(starts.values()) - {start}
            path = self._search(start, goal, blocked, occupied, edges)
            if not path:
                path = [start]
            padded = path + [path[-1]] * (self.horizon + 1 - len(path))
            self.routes[unit_id] = padded
            for t, tile in enumerate(padded):
                self.reservations[t, tile[0], tile[1]] = True
                if t:
                    edges.add((padded[t - 1], tile, t))
            if len(path) > 1:
                dx, dy = path[1][0] - start[0], path[1][1] - start[1]
                actions[unit_id, 0] = next(action for mx, my, action in MOVES
                                           if (mx, my) == (dx, dy))
        return actions
