import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.replay_parser import (
    BASE_CHANNEL_NAMES,
    CHANNEL_NAMES,
    ENTITY_SYMMETRY_CHANNELS,
    MASK_CHANNELS,
    TILE_SYMMETRY_CHANNELS,
    VALUE_CHANNELS,
    TileDriftEstimator,
    apply_dual_symmetry,
    backcast_terrain_projections,
    infer_intents,
    parse_replay,
)


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
    def test_backcast_projects_only_terrain_known_by_each_frame(self):
        frames = []
        for step in range(2):
            obs = public_obs(step, step, (0, 0))
            tile = np.zeros((4, 4), dtype=np.int8)
            tile[2, 2] = 1
            obs["map_features"]["tile_type"] = tile.tolist()
            visible = np.zeros((4, 4), dtype=bool)
            visible[0, 0] = True
            if step:
                visible[2, 2] = True
            obs["sensor_mask"] = visible.tolist()
            record = {"observation": {"obs": json.dumps(obs)}}
            frames.append([record, record])

        _, projections = backcast_terrain_projections(frames, 0, 0.05, 4, 4)

        self.assertEqual(int(projections[0].sum()), 0)
        self.assertGreater(int(projections[1].sum()), 0)

    def test_tile_drift_estimator_uses_stationary_frames(self):
        terrain = np.zeros((24, 24), dtype=np.int8)
        terrain[3, 4] = 1
        terrain[10, 12] = 1
        terrain[20, 1] = 1
        visible = np.ones((24, 24), dtype=bool)
        estimator = TileDriftEstimator()

        for step in range(80):
            if step and estimator._moves_at_step(0.05, step):
                terrain = np.roll(
                    terrain,
                    shift=estimator._shift_for_speed(0.05),
                    axis=(0, 1),
                )
            estimator.observe(step, terrain, visible)
            if estimator.converged_step is not None:
                break

        self.assertEqual(estimator.candidate_speeds, (0.05,))
        self.assertEqual(estimator.converged_step, 21)
        self.assertEqual(estimator.history[1]["candidate_count"], 4)
        self.assertTrue(all(
            record["has_exact_candidate"] is not False
            for record in estimator.history
        ))

    def test_tile_drift_estimator_skips_missing_frames(self):
        terrain = np.zeros((24, 24), dtype=np.int8)
        terrain[3, 4] = 1
        visible = np.ones((24, 24), dtype=bool)
        estimator = TileDriftEstimator()
        estimator.observe(0, terrain, visible)
        terrain = np.roll(terrain, shift=(1, -1), axis=(0, 1))

        record = estimator.observe(2, terrain, visible)

        self.assertEqual(record["candidate_count"], 8)
        self.assertIsNone(record["has_exact_candidate"])

    def test_partial_overlap_does_not_refute_uncompared_drift_candidates(self):
        terrain = np.zeros((24, 24), dtype=np.int8)
        terrain[22, 22] = 1
        visible = np.zeros((24, 24), dtype=bool)
        visible[21:24, 21:24] = True
        estimator = TileDriftEstimator()
        estimator.observe(7, terrain, visible)
        moved = np.roll(terrain, shift=(1, -1), axis=(0, 1))
        record = estimator.observe(8, moved, visible)
        # Static candidates have nine comparable cells and mismatch. Moving
        # candidates have only four comparable cells, so remain inconclusive.
        self.assertEqual(record["comparable_cells"], 9)
        self.assertIsNone(record["has_exact_candidate"])
        self.assertIn(0.15, estimator.candidate_speeds)

    def test_all_compared_candidates_refuted_still_reports_conflict(self):
        terrain = np.zeros((24, 24), dtype=np.int8)
        terrain[3, 4] = 1
        visible = np.ones((24, 24), dtype=bool)
        estimator = TileDriftEstimator()
        estimator.observe(7, terrain, visible)
        inconsistent = terrain.copy()
        inconsistent[10, 12] = 2
        record = estimator.observe(8, inconsistent, visible)
        self.assertIs(record["has_exact_candidate"], False)

    def test_dual_symmetry_maps_tiles_and_entities_separately(self):
        features = np.zeros((len(BASE_CHANNEL_NAMES), 24, 24), dtype=np.float32)
        features[TILE_SYMMETRY_CHANNELS[0], 1, 4] = 1
        features[ENTITY_SYMMETRY_CHANNELS[0], 2, 3] = 1
        features[13, 1, 4] = 1

        mapped = apply_dual_symmetry(features)

        self.assertEqual(mapped[TILE_SYMMETRY_CHANNELS[0], 19, 22], 1)
        self.assertEqual(mapped[ENTITY_SYMMETRY_CHANNELS[0], 21, 20], 1)
        self.assertEqual(mapped[13, 1, 4], 1)
        self.assertEqual(mapped[13, 22, 19], 0)
        self.assertEqual(mapped.shape, (len(BASE_CHANNEL_NAMES), 24, 24))

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
        tile = np.zeros((4, 4), dtype=np.int8)
        tile[0, 1] = tile[2, 3] = 1
        tile[1, 2] = 2
        for step in range(43):
            if step and TileDriftEstimator._moves_at_step(0.025, step):
                tile = np.roll(tile, shift=(1, -1), axis=(0, 1))
            obs = public_obs(step, step, (min(step, 3), 0))
            obs["map_features"]["tile_type"] = tile.tolist()
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
            features = np.zeros((43, 25, 4, 4), dtype=np.float32)
            features[:, MASK_CHANNELS] = data["feature_masks"]
            features[:, VALUE_CHANNELS] = data["feature_values"]
            self.assertEqual(data["feature_masks"].dtype, np.uint8)
            self.assertEqual(data["feature_values"].dtype, np.float16)
            self.assertNotIn("features", data.files)
            self.assertEqual(data["intent_target"].shape, (43, 2, 4, 4))
            self.assertEqual(data["actions"].shape, (43, 2, 3))
            self.assertNotIn("drift_speed", data.files)
            self.assertEqual(data["drift_candidate_count"].shape, (43,))
            self.assertEqual(int(data["drift_candidate_count"][0]), 8)
            self.assertEqual(features[0, CHANNEL_NAMES.index("future_nebula_5"), 1, 0], 1)
            self.assertEqual(features[0, CHANNEL_NAMES.index("future_asteroid_5"), 2, 1], 1)
            self.assertEqual(features[0, CHANNEL_NAMES.index("future_nebula_10"), 1, 0], 1)
            self.assertEqual(features[0, CHANNEL_NAMES.index("future_asteroid_10"), 2, 1], 1)
            self.assertEqual(summaries[0]["drift_confirmed_speed"], 0.025)
            self.assertEqual(summaries[0]["drift_converged_step"], 21)
            self.assertEqual(summaries[0]["parser_errors"], 0)


if __name__ == "__main__":
    unittest.main()
