"""Belief over scoring tiles, updated only from the player's observations."""

import numpy as np


class RelicInference:
    def __init__(self, width, height, max_steps_in_match=100):
        self.width = width
        self.height = height
        self.max_steps_in_match = max_steps_in_match
        self.all_spawned_after = 2 * (max_steps_in_match + 1) + max_steps_in_match // 2
        self.known_relics = {}
        self.positive = np.zeros((width, height), dtype=bool)
        self.negative = np.zeros((width, height), dtype=bool)
        self.evidence_sum = np.zeros((width, height), dtype=np.float32)
        self.evidence_count = np.zeros((width, height), dtype=np.float32)
        self.equations = []
        self.previous_points = None
        self.previous_match_step = None
        self.previous_occupied = np.zeros((width, height), dtype=bool)
        self.current_occupied = np.zeros((width, height), dtype=bool)

    def _occupied(self, obs, team_id):
        mask = np.asarray(obs["units_mask"][team_id], dtype=bool)
        positions = np.asarray(obs["units"]["position"][team_id])
        return frozenset(
            (int(x), int(y)) for x, y in positions[mask]
            if 0 <= x < self.width and 0 <= y < self.height
        )

    def _propagate(self):
        # Exact equations are kept only after the last possible relic activation.
        equations = set(self.equations[-64:])
        for _ in range(8):
            simplified = set()
            for cells, score in equations:
                unknown = frozenset(
                    cell for cell in cells
                    if not self.positive[cell] and not self.negative[cell]
                )
                remainder = score - sum(bool(self.positive[cell]) for cell in cells)
                if remainder < 0 or remainder > len(unknown):
                    continue
                if unknown:
                    simplified.add((unknown, remainder))
            changed = False
            for cells, score in simplified:
                if score == 0 or score == len(cells):
                    for cell in cells:
                        if score == 0:
                            self.negative[cell] = True
                        else:
                            self.positive[cell] = True
                    changed = True
            if changed:
                equations = simplified
                continue
            derived = set()
            for small, small_score in simplified:
                for large, large_score in simplified:
                    if small < large:
                        diff = large - small
                        remaining = large_score - small_score
                        if 0 <= remaining <= len(diff):
                            derived.add((frozenset(diff), remaining))
            new_equations = simplified | derived
            if new_equations == equations:
                break
            equations = set(sorted(
                new_equations,
                key=lambda item: (len(item[0]), item[1], tuple(sorted(item[0]))),
            )[:64])
        self.equations = sorted(
            equations, key=lambda item: (len(item[0]), item[1], tuple(sorted(item[0])))
        )

    def update(self, step, obs, team_id):
        match_step = int(obs["match_steps"])
        self.previous_occupied[:] = self.current_occupied if match_step else False
        occupied = self._occupied(obs, team_id)
        self.current_occupied.fill(False)
        for cell in occupied:
            self.current_occupied[cell] = True
        if step in (self.max_steps_in_match + 1, 2 * (self.max_steps_in_match + 1)):
            self.evidence_sum.fill(0)
            self.evidence_count.fill(0)
        relic_mask = np.asarray(obs["relic_nodes_mask"], dtype=bool)
        relic_positions = np.asarray(obs["relic_nodes"])
        for relic_id in np.flatnonzero(relic_mask):
            x, y = map(int, relic_positions[relic_id])
            if 0 <= x < self.width and 0 <= y < self.height:
                self.known_relics[int(relic_id)] = (x, y)

        points = int(np.asarray(obs["team_points"])[team_id])
        if (
            self.previous_points is not None
            and match_step > self.previous_match_step
            and points >= self.previous_points
        ):
            delta = points - self.previous_points
            if 0 <= delta <= len(occupied) and occupied:
                if delta == len(occupied):
                    for cell in occupied:
                        self.positive[cell] = True
                if step > self.all_spawned_after:
                    self.equations.append((occupied, delta))
                    self._propagate()
                else:
                    fraction = delta / len(occupied)
                    for cell in occupied:
                        self.evidence_sum[cell] += fraction
                        self.evidence_count[cell] += 1
        self.previous_points = points
        self.previous_match_step = match_step

    def support(self):
        support = np.zeros((self.width, self.height), dtype=bool)
        for rx, ry in self.known_relics.values():
            support[max(0, rx - 2):min(self.width, rx + 3),
                    max(0, ry - 2):min(self.height, ry + 3)] = True
        return support

    def probability(self):
        support = self.support()
        prior = np.where(support, 0.25, 0.04).astype(np.float32)
        probability = (
            prior + self.evidence_sum * 0.5
        ) / (1 + self.evidence_count * 0.5)
        probability[self.negative] = 0
        probability[self.positive] = 1
        return probability

    def confidence(self):
        confidence = self.evidence_count / (self.evidence_count + 4)
        confidence[self.negative | self.positive] = 1
        return confidence.astype(np.float32)
