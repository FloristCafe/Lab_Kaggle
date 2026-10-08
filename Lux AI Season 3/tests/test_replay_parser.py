import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.replay_parser import CHANNEL_NAMES, infer_intents, parse_replay


def public_obs(step, match_step, position, points=0):
    tile = np.zeros((4, 4), dtype=int).tolist()
    energy = np.zeros((4, 4), dtype=int).tolist()
    sensor = np.ones((4, 4), dtype=bool).tolist()
    positions = [[list(position), [-1, -1]], [[-1, -1], [-1, -1]]]
    energies = [[100, -1], [-1, -1]]
    return {
        "units": {"position": positions, "energy": energies},
        "units_mask": [[True, False], [False, False]],
        "sensor_mask": sensor,
        "map_features": {"tile_type": tile, "energy": energy},
        "relic_nodes": [[-1, -1]], "relic_nodes_mask": [False],
        "team_points": [points, 0], "team_wins": [0, 0],
        "steps": step, "match_steps": match_step,
    }


class ReplayParserTests(unittest.TestCase):
    def test_intent_window_requires_move_and_uses_future_mode(self):
        positions = np.array([[[0, 0]], [[1, 0]], [[2, 0]], [[2, 0]]])
        active = np.ones((4, 1), dtype=bool)
        actions = np.zeros((4, 1, 3), dtype=np.int16)
        actions[0, 0, 0] = 2
        actions[1, 0, 0] = 2
        match_steps = np.arange(4)
        target, valid = infer_intents(positions, active, actions, match_steps, 4, 4, 3)
        self.assertTrue(valid[0, 0])
        self.assertEqual(np.argwhere(target[0, 0]).tolist(), [[2, 0]])
        self.assertFalse(valid[-1, 0])

    def test_parse_replay_writes_tensor_contract(self):
        frames = []
        for step in range(4):
            obs = public_obs(step, step, (min(step, 3), 0))
            record = {
                "observation": {"obs": json.dumps(obs), "player": "player_0"},
                "action": [[2, 0, 0], [0, 0, 0]],
            }
            other = {
                "observation": {"obs": json.dumps(obs), "player": "player_1"},
                "action": [[0, 0, 0], [0, 0, 0]],
            }
            frames.append([record, other])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.json"
            source.write_text(json.dumps({"steps": frames}), encoding="utf-8")
            output = Path(directory) / "out"
            summaries = parse_replay(source, output, width=4, height=4,
                                     max_units=2, horizon=3)
            self.assertEqual(len(summaries), 2)
            data = np.load(output / "sample_p0.npz")
            self.assertEqual(data["features"].shape[1:], (len(CHANNEL_NAMES), 4, 4))
            self.assertEqual(data["intent_target"].shape, (4, 2, 4, 4))
            self.assertEqual(data["actions"].shape, (4, 2, 3))
            self.assertEqual(summaries[0]["parser_errors"], 0)


if __name__ == "__main__":
    unittest.main()
