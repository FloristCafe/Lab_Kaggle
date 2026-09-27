import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents" / "v0"))
from agent import Agent


CONFIG = {
    "map_width": 5,
    "map_height": 5,
    "max_units": 2,
    "unit_move_cost": 2,
    "unit_sensor_range": 1,
}


def observation(positions=((0, 0), (4, 4)), energies=(10, 10)):
    return {
        "sensor_mask": np.zeros((5, 5), dtype=bool),
        "map_features": {"tile_type": np.full((5, 5), -1, dtype=int)},
        "relic_nodes_mask": np.array([False]),
        "relic_nodes": np.array([[-1, -1]]),
        "units_mask": np.array([[True, True], [False, False]]),
        "units": {
            "position": np.array([positions, [(-1, -1), (-1, -1)]]),
            "energy": np.array([energies, (0, 0)], dtype=int)[:, :, None],
        },
    }


class AgentTests(unittest.TestCase):
    def test_visible_terrain_persists_without_hidden_updates(self):
        agent = Agent("player_0", CONFIG)
        obs = observation()
        obs["sensor_mask"][1, 1] = True
        obs["map_features"]["tile_type"][1, 1] = 2
        agent.act(0, obs)
        agent.act(1, observation())
        self.assertEqual(agent.tile_type[1, 1], 2)
        self.assertEqual(agent.tile_type[2, 2], -1)

    def test_path_avoids_known_asteroids(self):
        agent = Agent("player_0", CONFIG)
        agent.tile_type[1, 0] = 2
        distances, first_moves = agent._paths_from((0, 0))
        self.assertNotIn((1, 0), distances)
        self.assertEqual(first_moves[(2, 0)], 3)

    def test_low_energy_unit_waits(self):
        agent = Agent("player_0", CONFIG)
        actions = agent.act(0, observation(energies=(1, 10)))
        self.assertEqual(actions.shape, (2, 3))
        self.assertEqual(actions[0].tolist(), [0, 0, 0])

    def test_units_receive_distinct_relic_targets(self):
        agent = Agent("player_0", CONFIG)
        obs = observation()
        obs["relic_nodes_mask"][0] = True
        obs["relic_nodes"][0] = (2, 2)
        agent.act(0, obs)
        self.assertNotEqual(agent.targets[0], agent.targets[1])


if __name__ == "__main__":
    unittest.main()
