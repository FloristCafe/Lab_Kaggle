import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.v3.dataset import coordinate_planes, lifecycle_exclusions
from agents.v3.losses import (
    apply_destination_mask,
    focal_spatial_loss,
    optimistic_destination_mask,
)
from agents.v3.residual_network import Network as ResidualNetwork
from agents.v3.unet28_network import Network
from scripts.replay_parser import VALUE_CHANNELS
from scripts.train_full_bc import FrozenResidualReference, evaluate_model


def query(position=(12, 12), energy=100):
    x = torch.zeros(1, 28, 24, 24)
    x[:, 25, position[0], position[1]] = 1
    x[:, 14, position[0], position[1]] = energy / 6400
    x[:, 26:28] = coordinate_planes()
    return x


class UNetFocalTests(unittest.TestCase):
    def test_frozen_reference_keeps_exact_26_channel_behavior(self):
        legacy = ResidualNetwork().eval()
        reference = FrozenResidualReference({"model_state_dict": legacy.state_dict()}).eval()
        x = query((0, 0))
        with torch.no_grad():
            torch.testing.assert_close(reference(x), legacy(x[:, :26]))
        self.assertFalse(any(p.requires_grad for p in reference.parameters()))

    def test_reused_slot_preserves_new_life_and_existing_reset_gap_boundaries(self):
        positions = np.array([[[14, 9]], [[14, 9]], [[0, 0]], [[0, 1]], [[0, 2]]])
        active = np.ones((5, 1), dtype=bool)
        excluded = lifecycle_exclusions(positions, active, np.arange(5), np.arange(5))
        self.assertEqual(excluded[:, 0].tolist(), [True, True, False, False, False])
        reset = lifecycle_exclusions(positions, active, np.arange(5), np.array([99, 100, 0, 1, 2]))
        self.assertEqual(reset[:, 0].tolist(), [False, False, True, False, False])
        gap = lifecycle_exclusions(positions, active, np.array([0, 1, 3, 4, 5]), np.arange(5))
        self.assertFalse(gap.any())

    def test_same_cell_respawn_energy_jump_detected_only_for_singletons(self):
        positions = np.zeros((3, 1, 2), dtype=np.int16)
        active = np.ones((3, 1), dtype=bool)
        values = np.zeros((3, len(VALUE_CHANNELS), 24, 24), dtype=np.float16)
        values[:, VALUE_CHANNELS.index(13), 0, 0] = 1 / 16
        values[:, VALUE_CHANNELS.index(14), 0, 0] = np.array([0, 100, 120]) / 6400
        mask = lifecycle_exclusions(positions, active, np.arange(3), np.arange(3), feature_values=values)
        self.assertEqual(mask[:, 0].tolist(), [True, False, False])
        values[:, VALUE_CHANNELS.index(13), 0, 0] = 2 / 16
        mask = lifecycle_exclusions(positions, active, np.arange(3), np.arange(3), feature_values=values)
        self.assertFalse(mask.any())

    def test_opposite_corner_output_has_gradient_to_anchor_corner(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        model = Network().eval()
        x = query((0, 0)).requires_grad_()
        logits = model(x)
        logits[0, 0, 23, 23].backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(abs(x.grad[0, 25, 0, 0].item()), 1e-10)

    def test_focal_gamma_zero_equals_ce_and_easy_loss_is_suppressed(self):
        target = torch.tensor([0])
        logits = torch.zeros(1, 1, 24, 24, requires_grad=True)
        expected = nn.functional.cross_entropy(logits.flatten(1), target)
        torch.testing.assert_close(focal_spatial_loss(logits, target, gamma=0), expected)
        easy = logits.detach().clone()
        easy[0, 0, 0, 0] = 10
        ce = nn.functional.cross_entropy(easy.flatten(1), target)
        self.assertLess(focal_spatial_loss(easy, target).item(), 0.01 * ce.item())
        focal_spatial_loss(logits, target).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_energy_delay_boundary_and_unknown_recovery_are_conservative(self):
        x = query((0, 0), energy=100)
        mask = optimistic_destination_mask(x)
        self.assertTrue(mask[0, 0, 12, 0])
        self.assertFalse(mask[0, 0, 13, 0])
        self.assertFalse(mask[0, 0, 23, 23])  # No wrapping across boundaries.
        empty = query((0, 0), energy=0)
        mask = optimistic_destination_mask(empty)
        self.assertTrue(mask[0, 0, 11, 0])  # Optimistic wait + recovery then move.
        self.assertFalse(mask[0, 0, 12, 0])
        empty[:, 0] = 1  # Current observed zero energy gain requires two waits.
        mask = optimistic_destination_mask(empty)
        self.assertTrue(mask[0, 0, 10, 0])
        self.assertFalse(mask[0, 0, 11, 0])

    def test_first_step_obstacle_is_not_a_permanent_future_block(self):
        x = query((0, 0), energy=1)
        x[:, 0] = 1
        x[:, 5, 1, 0] = 1
        x[:, 5, 0, 1] = 1
        mask = optimistic_destination_mask(x)
        self.assertTrue(mask[0, 0, 1, 0])  # Terrain can move after waiting.
        self.assertFalse(mask[0, 0, 12, 0])
        other = x.clone()
        other[:, 21:25] = 1
        torch.testing.assert_close(mask, optimistic_destination_mask(other))

    def test_fp16_tile_energy_upper_bound_never_rounds_below_true_energy(self):
        energies = torch.arange(401)
        normalized = (energies.float() / 6400).half().float()
        recovered = (normalized * 6400).ceil().clamp(0, 400)
        self.assertTrue((recovered >= energies).all())

    def test_masked_logits_gradients_and_target_guard(self):
        x = query((0, 0))
        raw = torch.randn(1, 1, 24, 24, dtype=torch.bfloat16, requires_grad=True)
        masked, allowed = apply_destination_mask(raw, x, torch.tensor([0]))
        self.assertEqual(masked.dtype, torch.float32)
        self.assertEqual(masked[0, 0, 23, 23].item(), -1e9)
        focal_spatial_loss(masked, torch.tensor([0])).backward()
        self.assertEqual(raw.grad[~allowed].count_nonzero().item(), 0)
        self.assertTrue(torch.isfinite(raw.grad).all())
        with self.assertRaisesRegex(ValueError, "excludes"):
            apply_destination_mask(raw.detach(), x, torch.tensor([575]))

    def test_anchor_prediction_metric_detects_identity_policy_directly(self):
        class Identity(nn.Module):
            def forward(self, x):
                return x[:, 25:26] * 20

        x = torch.cat((query((0, 0)), query((12, 12))))
        y = torch.tensor([0, 12 * 24 + 13])
        loader = [(x, y, 0, 2)]
        metrics = evaluate_model(Identity(), loader, torch.device("cpu"), False)
        self.assertEqual(metrics["anchor_prediction_rate"], 1.0)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["stay_accuracy"], 1.0)
        self.assertEqual(metrics["moving_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
