"""Turn consecutive public observations into time-valid binary score equations."""

import numpy as np


class ObsParser:
    def __init__(self, solver, team_id, max_steps_in_match=100):
        self.solver = solver
        self.team_id = team_id
        self.last_step = None
        self.last_match_step = None
        self.last_points = None
        # Last possible spawn is strictly before this state step.
        self.stable_from_step = 2 * (max_steps_in_match + 1) + max_steps_in_match // 2
        self.last_equation = None
        self.last_update = None
        self.last_reason = None
        self.static_epoch = False

    def observe(self, obs):
        step = int(obs["steps"])
        match_step = int(obs["match_steps"])
        points = int(np.asarray(obs["team_points"])[self.team_id])
        previous_points = self.last_points
        self.last_equation = None
        self.last_update = None
        self.last_reason = None

        valid_transition = (
            self.last_step is not None
            and step == self.last_step + 1
            and match_step == self.last_match_step + 1
            and match_step > 0
            and points >= self.last_points
        )
        self.last_step, self.last_match_step, self.last_points = step, match_step, points
        if not valid_transition:
            self.last_reason = "nonconsecutive_or_reset"
            return None
        mask = np.asarray(obs["units_mask"], dtype=bool)
        positions = np.asarray(obs["units"]["position"])
        energies = np.asarray(obs["units"]["energy"]).reshape(mask.shape)
        own = set()
        enemy = set()
        for team, destination in ((self.team_id, own), (1 - self.team_id, enemy)):
            for x, y in positions[team, mask[team] & (energies[team] >= 0)]:
                tile = (int(x), int(y))
                if 0 <= tile[0] < self.solver.width and 0 <= tile[1] < self.solver.height:
                    destination.add(tile)
        if any(abs(x - ex) + abs(y - ey) <= 1
               for x, y in own for ex, ey in enemy):
            self.last_reason = "visible_enemy_contact"
            return None

        delta = points - previous_points
        if step < self.stable_from_step:
            self.solver.add_dynamic_equation(own, delta)
            self.last_equation = {"tiles": sorted(own), "score": delta}
            self.last_reason = "dynamic_only"
            return None
        if not self.static_epoch:
            self.solver.enter_static_epoch()
            self.static_epoch = True
        return self._submit(own, delta)

    def _submit(self, occupied, delta):
        known = self.solver.values
        unknown = occupied - known.keys()
        residual = delta - sum(known[tile] for tile in occupied if tile in known)
        equation = {"tiles": sorted(unknown), "score": residual}
        if not 0 <= residual <= len(unknown):
            raise ValueError("observation contradicts known tile values")
        self.last_equation = equation
        self.last_update = self.solver.add_equation(equation["tiles"], residual)
        return equation
