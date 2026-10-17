import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.strategic.features import CausalObservationBuilder
from agents.strategic.model import (
    StrategicActor,
    action_statistics,
    choose_actions,
    supervised_loss,
)


def public_obs(t=0):
    pos = np.full((2, 16, 2), -1, np.int16)
    pos[0, 0], pos[1, 0] = (2, 3), (22, 22)
    mask = np.zeros((2, 16), bool)
    mask[0, 0] = True
    energy = np.full((2, 16), -1, np.int16)
    energy[0, 0] = 100
    visible = np.zeros((24, 24), bool)
    visible[:6, :6] = True
    tiles = np.full((24, 24), -1, np.int16)
    tiles[visible] = 0
    field = np.full((24, 24), -1, np.int16)
    field[visible] = 3
    return dict(
        steps=t,
        match_steps=t,
        units={"position": pos, "energy": energy},
        units_mask=mask,
        sensor_mask=visible,
        map_features={"tile_type": tiles, "energy": field},
        relic_nodes=np.array([[4, 3], [-1, -1]]),
        relic_nodes_mask=[True, False],
        team_points=np.zeros(2, np.int16),
        team_wins=np.zeros(2, np.int16),
    )


class StrategicModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_shapes_unit_queries_and_finite_region_gradients(self):
        actor = StrategicActor()
        history = torch.randn(2, 3, 18, 24, 24)
        positions = torch.zeros(2, 16, 2, dtype=torch.long)
        positions[:, 1] = 23
        units = torch.zeros(2, 16, 22)
        units[:, 1, 2] = 1
        alive = torch.ones(2, 16, dtype=torch.bool)
        ml, gl = actor(history, positions, units, alive)
        self.assertEqual(ml.shape, (2, 16, 5))
        self.assertEqual(gl.shape, (2, 16, 5, 24, 24))
        self.assertFalse(torch.equal(gl[:, 0], gl[:, 1]))
        region = torch.zeros(2, 16, 24, 24, dtype=torch.bool)
        region[:, :, 5, 5] = True
        region[:, :, 5, 6] = True
        batch = dict(
            valid=alive,
            alive=alive,
            mode=torch.zeros(2, 16, dtype=torch.long),
            regions=region,
            confidence=torch.ones(2, 16),
        )
        loss, _ = supervised_loss(ml, gl, batch)
        loss.backward()
        self.assertTrue(
            all(
                torch.isfinite(p.grad).all()
                for p in actor.parameters()
                if p.grad is not None
            )
        )

    def test_joint_argmax_not_greedy_mode_then_goal(self):
        mode = torch.tensor([[[0.6, 0.4, 0.0, 0.0, 0.0]]]).clamp_min(1e-10).log()
        goal = torch.zeros(1, 1, 5, 24, 24)
        goal[0, 0, 1, 7, 8] = 10
        m, g, lp = choose_actions(mode, goal, greedy=True)
        self.assertEqual(m.item(), 1)
        self.assertEqual(g.item(), 7 * 24 + 8)
        new_lp, entropy = action_statistics(mode, goal, m, g)
        torch.testing.assert_close(lp, new_lp)
        self.assertTrue(torch.isfinite(entropy).all())

    def test_causal_prefix_and_no_center_entity_fabrication(self):
        cfg = {"unit_move_cost": 2, "unit_sap_cost": 30, "unit_sap_range": 4}
        b1, b2 = CausalObservationBuilder(0, cfg), CausalObservationBuilder(0, cfg)
        frames = [public_obs(i) for i in range(5)]
        a = [b1.observe(f) for f in frames]
        b = [b2.observe(f) for f in frames[:3]]
        np.testing.assert_array_equal(a[2]["history"], b[2]["history"])
        self.assertEqual(
            a[0]["base"][10, 19, 20], 0
        )  # incorrect center mirror of (4,3)
        self.assertEqual(a[0]["base"][9, 22, 22], 0)  # unobserved center energy
        self.assertEqual(a[0]["positions"][0].tolist(), [2, 3])


if __name__ == "__main__":
    unittest.main()
