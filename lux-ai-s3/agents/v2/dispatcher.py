"""Minimal one-step target selection for testing rank-driven active yielding."""

import numpy as np


class Dispatcher:
    def __init__(self, solver, recovery_steps=3, score_weight=1.0,
                 info_weight=1.0, safe_bonus=0.01):
        if recovery_steps < 1:
            raise ValueError("recovery_steps must be positive")
        self.solver = solver
        self.recovery_steps = recovery_steps
        self.score_weight = score_weight
        self.info_weight = info_weight
        self.safe_bonus = safe_bonus
        self.decay = np.ones((solver.width, solver.height), dtype=np.float32)
        self.last_yield_unit = None

    def utility_map(self):
        probability = self.solver.feature_map()
        entropy = 4 * probability * (1 - probability)
        utility = (self.score_weight * probability
                   + self.info_weight * entropy * self.decay)
        utility[probability == 0] += self.safe_bonus
        return utility

    def assign_macro(self, units, passable, unseen, support,
                     match_step, max_steps_in_match, sensor_range):
        """Globally assign distinct targets using score, visibility, and distance."""
        if not units:
            return {}
        from scipy.optimize import linear_sum_assignment

        probability = self.solver.feature_map()
        probability = np.where(probability == 0.5,
                               np.where(support, 0.25, 0.04), probability)
        width, height = probability.shape
        information = np.zeros((width, height), dtype=np.float32)
        for x in range(width):
            for y in range(height):
                information[x, y] = unseen[
                    max(0, x - sensor_range):min(width, x + sensor_range + 1),
                    max(0, y - sensor_range):min(height, y + sensor_range + 1),
                ].sum()
        remaining = max(0, max_steps_in_match - match_step)
        score_weight = min(remaining, 25) * 0.9
        base = score_weight * probability + 1.2 * np.sqrt(information)
        base += self.info_weight * 4 * probability * (1 - probability) * self.decay
        base += self.safe_bonus * (probability == 0)
        grid_x, grid_y = np.indices((width, height))
        utilities = []
        for _, (x, y) in units:
            distance = np.abs(grid_x - x) + np.abs(grid_y - y)
            score = base / (1 + 0.25 * distance)
            score[~passable] = -1e6
            utilities.append(score.ravel())
        rows, columns = linear_sum_assignment(np.stack(utilities), maximize=True)
        return {units[row][0]: divmod(int(column), height)
                for row, column in zip(rows, columns)}

    def plan(self, units, update=None, passable=None, enemy_positions=(),
             base_targets=None):
        """Return physical one-step targets; movement execution is outside this class."""
        positions = {unit_id: tuple(position) for unit_id, position in units}
        occupied = set(positions.values())
        redundant = set(update.redundant_cells) if update is not None else set()
        for tile in np.ndindex(self.decay.shape):
            if tile in redundant and tile in occupied:
                self.decay[tile] = 0
            elif tile not in occupied:
                self.decay[tile] = min(1, self.decay[tile] + 1 / self.recovery_steps)
        for tile in enemy_positions:
            self.decay[tuple(tile)] = 1

        targets = positions.copy() if base_targets is None else base_targets.copy()
        self.last_yield_unit = None
        safe = {tile for tile, value in self.solver.values.items() if value == 0}
        if passable is None:
            passable = np.ones(self.decay.shape, dtype=bool)
        # One deliberate retreat changes the next occupancy equation while the
        # other unit holds the unresolved tile. A tiny safe bonus alone cannot
        # outweigh an unknown tile's expected score.
        for unit_id in sorted(positions, reverse=True):
            x, y = positions[unit_id]
            if (x, y) not in redundant or self.solver.feature_map()[x, y] != 0.5:
                continue
            choices = [(nx, ny) for nx, ny in ((x - 1, y), (x + 1, y),
                                                 (x, y - 1), (x, y + 1))
                       if (nx, ny) in safe and (nx, ny) not in occupied
                       and (nx, ny) not in targets.values()
                       and passable[nx, ny]]
            if choices:
                targets[unit_id] = max(choices, key=lambda tile:
                                       (self.utility_map()[tile], -tile[0], -tile[1]))
                self.last_yield_unit = unit_id
                break
        return targets
