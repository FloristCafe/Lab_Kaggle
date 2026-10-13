import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.v3.dataset import (
    CAUSAL_CURRENT_CHANNELS,
    UnitFrameDataset,
    causal_frame_maps,
    collate_spatial_frames,
    gaussian_heatmaps,
)
from agents.v3.losses import (
    causal_destination_mask,
    masked_heatmap_mse,
    optimistic_destination_mask,
)
from agents.v3.network import Network
from scripts.replay_parser import pack_features
from scripts.train_spatial_mse_legacy import evaluate_heatmaps


class SpatialRegressionTests(unittest.TestCase):
    def test_gaussian_near_miss_is_less_costly_than_distant_miss(self):
        truth = gaussian_heatmaps([[12, 12]])
        close = gaussian_heatmaps([[13, 12]])[:, None].requires_grad_()
        far = gaussian_heatmaps([[23, 23]])[:, None]
        valid = torch.tensor([True])
        self.assertEqual(truth[0, 12, 12].item(), 1)
        self.assertAlmostEqual(truth[0, 13, 12].item(), np.exp(-1 / 4.5), places=6)
        self.assertLess(masked_heatmap_mse(close, truth, valid).item(),
                        masked_heatmap_mse(far, truth, valid).item())
        masked_heatmap_mse(close, truth, valid).backward()
        self.assertTrue(torch.isfinite(close.grad).all())

    def test_invalid_nan_units_do_not_contaminate_loss_or_gradient(self):
        p = torch.zeros(2, 1, 24, 24)
        p[1] = float("nan")
        p.requires_grad_()
        y = gaussian_heatmaps([[0, 0], [12, 12]])
        y[1] = float("nan")
        valid = torch.tensor([True, False])
        loss = masked_heatmap_mse(p, y, valid)
        torch.testing.assert_close(loss, y[0].square().mean())
        loss.backward()
        self.assertEqual(p.grad[1].count_nonzero().item(), 0)
        self.assertTrue(torch.isfinite(p.grad).all())
        zero = masked_heatmap_mse(p, y, torch.tensor([False, False]))
        self.assertEqual(zero.item(), 0)

    def test_future_and_backcast_mutations_leave_current_input_identical(self):
        rng = np.random.default_rng(42)
        features = rng.normal(size=(6, 25, 24, 24)).astype(np.float32)
        steps = np.arange(6)
        before = causal_frame_maps(features, steps, steps, 2)
        changed = features.copy()
        changed[3:] = 999
        changed[:, [6, 7, 8, 21, 22, 23, 24]] = -999
        after = causal_frame_maps(changed, steps, steps, 2)
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(before[:18], features[2, CAUSAL_CURRENT_CHANNELS])
        self.assertEqual(before[18, 0, 0], features[1, 0, 0, 0])
        self.assertEqual(before[25, 0, 0], features[0, 0, 0, 0])

    def test_history_does_not_cross_missing_observations_or_match_reset(self):
        f = np.ones((4, 25, 24, 24), dtype=np.float32)
        zero = causal_frame_maps(f, np.arange(4), np.arange(4), 0)
        self.assertEqual(zero[18:].sum(), 0)
        # lag2 must not jump over a missing T-1 transition.
        gap = causal_frame_maps(f, np.array([0, 1, 3, 4]), np.array([0, 1, 3, 4]), 3)
        self.assertGreater(gap[18:25].sum(), 0)
        self.assertEqual(gap[25:].sum(), 0)
        reset = causal_frame_maps(f, np.arange(4), np.array([99, 100, 0, 1]), 3)
        self.assertGreater(reset[18:25].sum(), 0)
        self.assertEqual(reset[25:].sum(), 0)

    def test_dataset_dynamic_gaussian_anchor_history_and_disk_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "1_p0.npz"
            f = np.zeros((3, 25, 24, 24), dtype=np.float32)
            f[:, 0] = 1
            f[:, 9] = np.arange(3)[:, None, None] / 10
            f[:, 6:9] = 1
            f[:, 21:25] = 1
            valid = np.zeros((3, 16), dtype=bool)
            valid[:, 3] = True
            positions = np.full((3, 16, 2), -1, dtype=np.int16)
            positions[:, 3] = (6, 8)
            targets = np.zeros((3, 16, 24, 24), dtype=np.uint8)
            targets[:, 3, 7, 8] = 1
            np.savez_compressed(p, schema_version=np.uint8(3), **pack_features(f),
                                intent_valid=valid, active=valid, positions=positions,
                                intent_target=targets, steps=np.arange(3), match_steps=np.arange(3))
            original = p.read_bytes()
            data = UnitFrameDataset([p], normalize_targets=False)
            x, y, v = data[2]
            self.assertEqual(x.shape, (1, 35, 24, 24))
            self.assertEqual(y.shape, (1, 24, 24))
            self.assertEqual(y.dtype, torch.float32)
            self.assertEqual(y[0, 7, 8].item(), 1)
            self.assertEqual(x[0, 32, 6, 8].item(), 1)
            self.assertEqual(x[0, 32].sum().item(), 1)
            self.assertEqual(x[0, 33, 23, 7].item(), 1)
            self.assertAlmostEqual(x[0, 22, 0, 0].item(), 0.1, places=3)
            self.assertEqual(x[0, 29, 0, 0].item(), 0)
            self.assertTrue(v.all())
            batch = collate_spatial_frames([data[0], data[2]])
            self.assertEqual(batch[0].shape, (2, 35, 24, 24))
            self.assertEqual(original, p.read_bytes())

    def test_causal_physical_mask_matches_legacy_for_same_current_evidence(self):
        x = torch.zeros(2, 35, 24, 24)
        x[:, 32, 0, 0] = 1
        x[:, 11, 0, 0] = 100 / 6400
        old = torch.zeros(2, 28, 24, 24)
        old[:, 25] = x[:, 32]
        old[:, 14] = x[:, 11]
        torch.testing.assert_close(causal_destination_mask(x), optimistic_destination_mask(old))
        changed = x.clone()
        changed[:, 18:32] = 999  # History has no influence on current mask.
        torch.testing.assert_close(causal_destination_mask(changed), causal_destination_mask(x))

    def test_new_network_and_metrics_distinguish_near_miss_from_anchor_copy(self):
        torch.set_num_threads(2)
        x = torch.zeros(2, 35, 24, 24)
        x[:, 32, 12, 12] = 1
        x[:, 11, 12, 12] = 100 / 6400
        y = gaussian_heatmaps([[12, 13], [12, 12]])
        model = Network()
        output = model(x)
        self.assertEqual(output.shape, (2, 1, 24, 24))
        masked_heatmap_mse(output.sigmoid(), y, torch.ones(2, dtype=torch.bool)).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        rows = [(x, y, torch.ones(2, dtype=torch.bool))]
        metrics = evaluate_heatmaps(lambda a: gaussian_heatmaps([[12, 12]] * len(a))[:, None],
                                   rows, torch.device("cpu"), unit_batch=1)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["within_1"], 1)
        self.assertEqual(metrics["mean_manhattan"], 0.5)
        self.assertEqual(metrics["anchor_prediction_rate"], 1)
        self.assertEqual(metrics["moving_accuracy"], 0)


if __name__ == "__main__":
    unittest.main()
