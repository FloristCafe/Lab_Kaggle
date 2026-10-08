import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.v2.dispatcher import Dispatcher
from agents.v2.equation_solver import EquationSolver
from agents.v2.obs_parser import ObsParser


A, B, C, D = (1, 1), (1, 2), (1, 3), (0, 1)


def observation(step, points, own, enemy=(), match_step=None):
    positions = np.full((2, 2, 2), -1, dtype=int)
    energies = np.full((2, 2), -1, dtype=int)
    mask = np.zeros((2, 2), dtype=bool)
    for team, members in enumerate((own, enemy)):
        for index, tile in enumerate(members):
            positions[team, index] = tile
            energies[team, index] = 100
            mask[team, index] = True
    return {
        "steps": step,
        "match_steps": step - 202 if match_step is None else match_step,
        "team_points": [points, 0],
        "units_mask": mask,
        "units": {"position": positions, "energy": energies},
    }


class ObsCycleTests(unittest.TestCase):
    def test_probe_deadlock_yield_and_resolve_in_five_observations(self):
        solver = EquationSolver(4, 4)
        parser = ObsParser(solver, team_id=0)
        dispatcher = Dispatcher(solver)

        self.assertIsNone(parser.observe(observation(260, 0, (C, D))))
        self.assertEqual(parser.observe(observation(261, 0, (C, D))),
                         {"tiles": [D, C], "score": 0})
        self.assertEqual(solver.values[C], 0)
        self.assertEqual(parser.observe(observation(262, 1, (A, B))),
                         {"tiles": [A, B], "score": 1})
        self.assertEqual(solver.rank, 2)
        self.assertEqual(parser.observe(observation(263, 2, (A, B))),
                         {"tiles": [A, B], "score": 1})
        self.assertFalse(parser.last_update.independent)
        targets = dispatcher.plan([(0, A), (1, B)], parser.last_update)
        self.assertEqual(targets, {0: A, 1: C})
        self.assertEqual(dispatcher.decay[A], 0)
        self.assertEqual(dispatcher.decay[B], 0)

        next_positions = tuple(targets[unit_id] for unit_id in sorted(targets))
        self.assertEqual(parser.observe(observation(264, 3, next_positions)),
                         {"tiles": [A], "score": 1})
        self.assertEqual(parser.last_update.newly_resolved, {A: 1, B: 0})
        self.assertEqual(solver.values[B], 0)
        self.assertEqual(solver.rank, 3)

    def test_spawn_window_and_match_reset_are_not_submitted(self):
        solver = EquationSolver(4, 4)
        parser = ObsParser(solver, 0)
        parser.observe(observation(250, 0, (A,)))
        self.assertIsNone(parser.observe(observation(251, 0, (A,))))
        self.assertEqual(parser.last_reason, "dynamic_only")
        self.assertEqual(solver.dynamic_pool, [(frozenset((A,)), 0)])
        self.assertIsNone(parser.observe(observation(252, 0, (A,), match_step=0)))
        self.assertEqual(solver.rank, 0)

    def test_duplicate_occupancy_and_visible_contact(self):
        solver = EquationSolver(4, 4)
        parser = ObsParser(solver, 0)
        parser.observe(observation(260, 0, (A, A)))
        self.assertEqual(parser.observe(observation(261, 1, (A, A))),
                         {"tiles": [A], "score": 1})
        self.assertIsNone(parser.observe(observation(262, 2, (B,), (C,))))
        self.assertEqual(parser.last_reason, "visible_enemy_contact")

    def test_uses_returned_position_and_excludes_negative_energy(self):
        solver = EquationSolver(4, 4)
        parser = ObsParser(solver, 0)
        parser.observe(observation(260, 0, (A,)))
        returned = observation(261, 1, (B, C))
        returned["units"]["energy"][0, 1] = -1
        self.assertEqual(parser.observe(returned),
                         {"tiles": [B], "score": 1})
        self.assertEqual(solver.values[B], 1)
        self.assertNotIn(A, solver.values)
        self.assertNotIn(C, solver.values)

    def test_dynamic_equation_cannot_conflict_with_static_pool(self):
        solver = EquationSolver(4, 4)
        parser = ObsParser(solver, 0)
        parser.observe(observation(250, 0, (A,)))
        parser.observe(observation(251, 0, (A,)))
        self.assertEqual(solver.dynamic_pool, [(frozenset((A,)), 0)])
        self.assertEqual(solver.values, {})
        parser.observe(observation(252, 1, (A,)))
        self.assertEqual(solver.dynamic_pool, [])
        self.assertEqual(solver.values[A], 1)
        self.assertEqual(solver.rank, 1)


if __name__ == "__main__":
    unittest.main()
