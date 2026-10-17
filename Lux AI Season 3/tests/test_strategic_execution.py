import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.strategic.control import GoalManager, TacticalController
from agents.strategic.model import StrategicActor, action_statistics, choose_actions
from agents.strategic.semimdp import clipped_actor_loss, smdp_gae
from agents.strategic.environment import lifecycle_events


def frame(energy=150, enemies=()):
    positions = np.zeros((16, 2), np.int16)
    positions[0] = (8, 8)
    alive = np.zeros(16, bool)
    alive[0] = True
    energies = np.zeros(16, np.int16)
    energies[0] = energy
    return dict(positions=positions, alive=alive, energy=energies,
                enemy=list(enemies), base=np.zeros((21, 24, 24), np.float32), match_step=1)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((ROOT / 'configs/strategic_baseline.yaml').read_text())['tactical']
        self.public = dict(unit_move_cost=2, unit_sap_cost=30, unit_sap_range=4)
        self.manager = GoalManager(self.cfg)
        self.modes = np.zeros(16, np.int64)
        self.goals = np.full(16, 15 * 24 + 15, np.int64)

    def test_persistence_tactical_override_and_round_reset(self):
        f = frame(10)
        self.manager.assign(self.modes, self.goals, f['alive'])
        self.manager.advance(f)
        self.manager.assign(self.modes, self.goals, f['alive'])
        self.assertEqual(self.manager.ages[0], 1)
        original = self.manager.goals.copy()
        tactical = TacticalController(self.public, self.cfg)
        actions, _ = tactical.act(f, self.manager)
        self.assertEqual(tactical.states[0], 4)
        np.testing.assert_array_equal(self.manager.goals, original)
        self.assertEqual(actions.shape, (16, 3))
        f['match_step'] = 0
        self.manager.advance(f)
        self.assertFalse(self.manager.valid.any())

    def test_five_states_and_sap_legality(self):
        fixtures = [(frame(), 0), (frame(10), 4),
                    (frame(30, [(0, (9, 8), 200)]), 3),
                    (frame(100, [(0, (12, 8), 100)]), 1),
                    (frame(200, [(0, (10, 8), 20)]), 2)]
        for f, expected in fixtures:
            tactical = TacticalController(self.public, self.cfg)
            self.manager.assign(self.modes, self.goals, f['alive'])
            actions, _ = tactical.act(f, self.manager)
            self.assertEqual(int(tactical.states[0]), expected)
            if expected == 2:
                self.assertEqual(actions[0, 0], 5)
                self.assertLessEqual(np.abs(actions[0, 1:]).max(), 4)
                self.assertTrue(((f['positions'][0] + actions[0, 1:]) >= 0).all())

    def test_sap_no_friendly_fire_and_energy_guard(self):
        f = frame(200, [(0, (10, 8), 20)])
        tactical = TacticalController(self.public, self.cfg)
        delta, damage = tactical.sap(f, 0, {})
        self.assertGreater(damage, 0)
        # The official engine damages only the opposing team. Friendly occupancy
        # of the target does not make the shot illegal.
        f['alive'][1] = True
        f['positions'][1] = f['positions'][0] + delta
        f['energy'][1] = 100
        delta2, _ = tactical.sap(f, 0, {})
        np.testing.assert_array_equal(delta, delta2)
        f['energy'][0] = 44
        self.assertIsNone(tactical.sap(f, 0, {}))

    def test_lifecycle_reset_is_not_death_and_reused_slot_is_a_new_life(self):
        before, after = frame(), frame()
        before['match_step'], after['match_step'] = 100, 0
        after['alive'][:] = False
        reset, reused, lost, deaths = lifecycle_events(before, after)
        self.assertTrue(reset)
        self.assertTrue(lost[0])
        self.assertFalse(deaths.any())
        after = frame()
        before['match_step'], after['match_step'] = 50, 51
        after['positions'][0] = (0, 0)
        reset, reused, lost, deaths = lifecycle_events(before, after)
        self.assertFalse(reset)
        self.assertTrue(reused[0] and deaths[0])
        self.assertFalse(lost[0])


class SMDPTests(unittest.TestCase):
    def test_variable_duration_terminal_and_bootstrap(self):
        a, ret = smdp_gae([1., 2.], [2, 3], [.5, .8], [.8, 100.],
                          [False, True], [True, False], gamma=.9, lam=.8)
        last = 2. - .8
        first = 1. + .9**2 * .8 - .5 + (.9*.8)**2 * last
        np.testing.assert_allclose(a, [first, last])
        np.testing.assert_allclose(ret, a + [.5, .8])
        cut, _ = smdp_gae([1.], [2], [.5], [4.], [False], [False], gamma=.9)
        self.assertAlmostEqual(cut[0], 1. + .9**2 * 4. - .5)
        with self.assertRaises(ValueError):
            smdp_gae([0], [0], [0], [0], [False], [False])

    def test_start_alive_mask_retains_death_causing_action_credit(self):
        new = torch.zeros(2, 3, requires_grad=True)
        old = torch.zeros_like(new)
        start_alive = torch.tensor([[True, False, False], [True, True, False]])
        # End-of-option alive is deliberately not used here.
        loss, clipped = clipped_actor_loss(new, old, torch.tensor([1., -1.]), start_alive)
        loss.backward()
        self.assertEqual(float(clipped), 0.)
        self.assertNotEqual(float(new.grad[0, 0]), 0.)
        self.assertEqual(float(new.grad[0, 1]), 0.)

    def test_factorized_sampling_likelihood_matches_recomputation(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        actor = StrategicActor()
        data = (torch.randn(2, 3, 18, 24, 24), torch.zeros(2, 16, 2, dtype=torch.long),
                torch.randn(2, 16, 22), torch.ones(2, 16, dtype=torch.bool))
        ml, gl = actor(*data)
        m, g, old = choose_actions(ml, gl)
        recomputed, _ = action_statistics(ml, gl, m, g)
        torch.testing.assert_close(old, recomputed)
        torch.testing.assert_close((recomputed - old).exp(), torch.ones_like(old))


if __name__ == '__main__':
    unittest.main()
