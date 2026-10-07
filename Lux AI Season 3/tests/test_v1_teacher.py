import sys
import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents" / "v1"))

from dispatcher import (
    Dispatcher, assign_targets, diversify_assignment, marginal_information,
    path_intervals, shortest_paths,
)
from inference import RelicInference

spec = importlib.util.spec_from_file_location(
    "v1_agent", Path(__file__).resolve().parents[1] / "agents" / "v1" / "agent.py"
)
v1_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1_module)


def observation(step, points, position=(1, 1)):
    return {
        "match_steps": step,
        "team_points": [points, 0],
        "relic_nodes_mask": [False],
        "relic_nodes": [[-1, -1]],
        "units_mask": [[True], [False]],
        "units": {"position": [[position], [[-1, -1]]]},
    }


class InferenceTests(unittest.TestCase):
    def test_early_zero_is_not_permanent_negative(self):
        solver = RelicInference(4, 4, max_steps_in_match=10)
        solver.update(0, observation(0, 0), 0)
        solver.update(1, observation(1, 0), 0)
        self.assertFalse(solver.negative[1, 1])
        solver.update(2, observation(2, 1), 0)
        self.assertTrue(solver.positive[1, 1])

    def test_exact_negative_after_final_spawn(self):
        solver = RelicInference(4, 4, max_steps_in_match=10)
        solver.update(29, observation(3, 0), 0)
        solver.update(30, observation(4, 0), 0)
        self.assertTrue(solver.negative[1, 1])
        self.assertEqual(solver.probability()[1, 1], 0)

    def test_subtract_confirmed_positive_from_equation(self):
        solver = RelicInference(4, 4, max_steps_in_match=10)
        solver.positive[0, 0] = True
        solver.equations = [(frozenset(((0, 0), (1, 1))), 1)]
        solver._propagate()
        self.assertTrue(solver.negative[1, 1])

    def test_previous_occupancy_tracks_observations_and_resets_at_match_start(self):
        solver = RelicInference(4, 4, max_steps_in_match=10)
        solver.update(0, observation(0, 0, (1, 1)), 0)
        self.assertFalse(solver.previous_occupied.any())
        solver.update(1, observation(1, 0, (2, 2)), 0)
        self.assertTrue(solver.previous_occupied[1, 1])
        self.assertFalse(solver.previous_occupied[2, 2])
        solver.update(2, observation(2, 0, (2, 2)), 0)
        self.assertTrue(solver.previous_occupied[2, 2])
        solver.update(11, observation(0, 0, (2, 2)), 0)
        self.assertFalse(solver.previous_occupied.any())


class DispatcherTests(unittest.TestCase):
    def test_previous_occupancy_removes_information_but_not_score_value(self):
        unexplored = np.array([[9, 4]], dtype=np.float32)
        probability = np.array([[0.25, 1]], dtype=np.float32)
        confidence = np.zeros_like(probability)
        occupied = np.array([[True, False]])
        information = marginal_information(
            unexplored, probability, confidence, occupied, 1.2
        )
        self.assertEqual(information[0, 0], 0)
        self.assertGreater(information[0, 1], 0)
        seen = np.zeros((3, 3), dtype=np.int32)
        field = np.zeros((3, 3), dtype=np.float32)
        scoring = np.zeros((3, 3), dtype=np.float32)
        scoring[1, 1] = 1
        prior = np.zeros((3, 3), dtype=bool)
        prior[1, 1] = True
        dispatcher = Dispatcher(3, 3, 1, 2, 1, 100)
        actions, targets, _, _, _ = dispatcher.plan(
            1, 1, [(0, (1, 1), 10)], np.zeros((3, 3), dtype=np.int8),
            seen, field, seen, [], scoring, field, previous_occupied=prior,
        )
        self.assertEqual(tuple(targets[0]), (1, 1))
        self.assertEqual(actions[0, 0], 0)

    def test_occupied_uncertain_tile_loses_stationary_preference(self):
        terrain = np.zeros((3, 3), dtype=np.int8)
        seen = np.full((3, 3), -1, dtype=np.int32)
        seen[1, 1] = 1
        field = np.zeros((3, 3), dtype=np.float32)
        probability = np.zeros((3, 3), dtype=np.float32)
        units = [(0, (1, 1), 10)]
        dispatcher = Dispatcher(3, 3, 1, 2, 1, 100)
        _, targets, _, _, _ = dispatcher.plan(
            1, 1, units, terrain, seen, field, seen, [], probability, field
        )
        self.assertEqual(tuple(targets[0]), (1, 1))
        previous_occupied = np.zeros((3, 3), dtype=bool)
        previous_occupied[1, 1] = True
        dispatcher = Dispatcher(3, 3, 1, 2, 1, 100)
        _, targets, _, _, _ = dispatcher.plan(
            1, 1, units, terrain, seen, field, seen, [], probability, field,
            previous_occupied=previous_occupied,
        )
        self.assertNotEqual(tuple(targets[0]), (1, 1))

    def test_joint_assignment_avoids_duplicate_target(self):
        utility = np.array([[10, 9, 0], [10, 1, 0]], dtype=np.float32)
        choice = assign_targets(utility, np.ones_like(utility, dtype=bool))
        self.assertEqual(choice.tolist(), [1, 0])

    def test_stacked_mobile_units_get_distinct_physical_targets(self):
        dispatcher = Dispatcher(3, 3, 2, 2, 1, 10)
        terrain = np.zeros((3, 3), dtype=np.int8)
        seen = np.zeros((3, 3), dtype=np.int32)
        energy_field = np.zeros((3, 3), dtype=np.float32)
        probability = np.full((3, 3), 0.25, dtype=np.float32)
        probability[1, 1] = 1
        units = [(0, (1, 1), 10), (1, (1, 1), 10)]
        actions, targets, _, _, _ = dispatcher.plan(
            0, 0, units, terrain, seen, energy_field, seen, [],
            probability, np.zeros((3, 3), dtype=np.float32),
        )
        self.assertNotEqual(tuple(targets[0]), tuple(targets[1]))
        self.assertEqual(sum(action[0] == 0 for action in actions), 1)

    def test_low_energy_can_delay_physical_separation(self):
        dispatcher = Dispatcher(3, 3, 2, 2, 1, 10)
        terrain = np.zeros((3, 3), dtype=np.int8)
        seen = np.zeros((3, 3), dtype=np.int32)
        field = np.zeros((3, 3), dtype=np.float32)
        actions, targets, _, _, _ = dispatcher.plan(
            0, 0, [(0, (1, 1), 1), (1, (1, 1), 1)], terrain,
            seen, field, seen, [], np.ones((3, 3), dtype=np.float32),
            field,
        )
        self.assertNotEqual(tuple(targets[0]), tuple(targets[1]))
        self.assertTrue(np.all(actions[:, 0] == 0))

    def test_uncertain_adjacent_goals_get_spread_without_harming_true_pair(self):
        utility = np.zeros((2, 9), dtype=np.float32)
        utility[0, [0, 8]] = (10, 8)
        utility[1, [1, 6]] = (10, 8)
        valid = np.ones_like(utility, dtype=bool)
        first_moves = [np.zeros((3, 3), dtype=np.int8) for _ in range(2)]
        parents = [np.full((3, 3, 2), -1, dtype=np.int16) for _ in range(2)]
        units = [(0, (0, 0), 10), (1, (2, 2), 10)]
        initial = assign_targets(utility, valid)
        spread = diversify_assignment(
            utility, valid, initial, units, first_moves, parents,
            np.zeros((3, 3), dtype=np.float32), 2,
        )
        self.assertEqual(set(spread.tolist()), {6, 8})
        known = np.zeros((3, 3), dtype=np.float32)
        known[0, 0] = known[0, 1] = 1
        retained = diversify_assignment(
            utility, valid, initial, units, first_moves, parents, known, 2
        )
        self.assertEqual(retained.tolist(), initial.tolist())

    def test_path_mask_blocks_visible_asteroid_but_keeps_unknown(self):
        blocked = np.zeros((3, 3), dtype=bool)
        blocked[1, 0] = True
        unknown = np.zeros_like(blocked)
        unknown[2, 2] = True
        field = np.zeros((3, 3), dtype=np.float32)
        distance, energy, move, _ = shortest_paths(
            (0, 0), blocked, unknown, field, ~unknown, field, 2
        )
        self.assertFalse(np.isfinite(distance[1, 0]))
        self.assertTrue(np.isfinite(distance[2, 2]))
        self.assertEqual(move[2, 0], 3)
        self.assertGreater(energy[2, 2], 0)

    def test_path_intervals_identify_goal_on_another_route(self):
        blocked = np.zeros((1, 4), dtype=bool)
        field = np.zeros((1, 4), dtype=np.float32)
        _, _, _, parent = shortest_paths(
            (0, 0), blocked, blocked, field, ~blocked, field, 2
        )
        entry, exit_time = path_intervals(parent, (0, 0))
        self.assertLessEqual(entry[0, 1], entry[0, 3])
        self.assertLessEqual(exit_time[0, 3], exit_time[0, 1])
        self.assertGreater(entry[0, 2], entry[0, 1])


class AgentIntegrationTests(unittest.TestCase):
    def test_agent_emits_observation_only_training_row(self):
        config = {
            "map_width": 4, "map_height": 4, "max_units": 2,
            "unit_move_cost": 2, "unit_sensor_range": 1,
            "max_steps_in_match": 10,
        }
        obs = {
            "sensor_mask": np.zeros((4, 4), dtype=bool),
            "map_features": {
                "tile_type": np.full((4, 4), -1, dtype=int),
                "energy": np.full((4, 4), -1, dtype=int),
            },
            "relic_nodes_mask": [False],
            "relic_nodes": [[-1, -1]],
            "units_mask": [[True, False], [False, False]],
            "units": {
                "position": [[[0, 0], [-1, -1]], [[-1, -1], [-1, -1]]],
                "energy": [[10, -1], [-1, -1]],
            },
            "team_points": [0, 0],
            "team_wins": [0, 0],
            "match_steps": 0,
        }
        obs["sensor_mask"][0, 0] = True
        obs["map_features"]["tile_type"][0, 0] = 0
        obs["map_features"]["energy"][0, 0] = 5
        with patch.dict(os.environ, {"LUX_V1_TRACE_DIR": "trace-enabled"}):
            agent = v1_module.Agent("player_0", config)
            actions = agent.act(0, obs)
        self.assertEqual(actions.shape, (2, 3))
        self.assertEqual(agent.last_trace["global_features"].shape, (22, 4, 4))
        self.assertTrue(agent.last_trace["active"][0])
        self.assertFalse(agent.last_trace["active"][1])
        self.assertTrue(agent.last_trace["label_valid"][0])
        self.assertFalse(agent.last_trace["label_valid"][1])
        self.assertEqual(agent.last_trace["unit_features"].shape, (2, 9))
        self.assertFalse(agent.last_trace["probe_mode"])


if __name__ == "__main__":
    unittest.main()
