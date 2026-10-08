import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.v2.router import Router


class RouterTests(unittest.TestCase):
    def test_immobile_unit_blocks_route_and_actions_are_legal(self):
        router = Router(4, 4, 2, move_cost=2, horizon=4)
        units = [(0, (0, 0), 10), (1, (1, 0), 1)]
        actions = router.route(units, {0: (2, 0), 1: (1, 0)},
                               np.zeros((4, 4), dtype=bool))
        self.assertEqual(actions.shape, (2, 3))
        self.assertEqual(actions[1, 0], 0)
        self.assertNotEqual(tuple(router.routes[0][1]), (1, 0))
        self.assertTrue(np.all((actions[:, 0] >= 0) & (actions[:, 0] <= 4)))

    def test_two_routes_reserve_goal_after_arrival(self):
        router = Router(4, 4, 2, move_cost=2, horizon=4)
        units = [(0, (0, 0), 10), (1, (2, 0), 10)]
        router.route(units, {0: (0, 1), 1: (0, 1)},
                     np.zeros((4, 4), dtype=bool), {0: 2, 1: 1})
        self.assertEqual(router.routes[0][-1], (0, 1))
        self.assertTrue(router.reservations[1:, 0, 1].all())
        self.assertTrue(all(a != b for a, b in zip(router.routes[0][1:],
                                                   router.routes[1][1:])))

    def test_existing_stack_cannot_be_undone_by_reservations(self):
        router = Router(3, 3, 2, move_cost=2, horizon=3)
        router.route([(0, (0, 0), 1), (1, (0, 0), 1)],
                     {0: (1, 0), 1: (0, 1)},
                     np.zeros((3, 3), dtype=bool))
        self.assertEqual(router.routes[0][1], router.routes[1][1])


if __name__ == "__main__":
    unittest.main()
