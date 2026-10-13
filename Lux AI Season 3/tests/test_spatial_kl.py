import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.v3.dataset import (
    UnitFrameDataset,
    gaussian_heatmaps,
    normalize_spatial_targets,
)
from agents.v3.losses import masked_spatial_kl, spatial_kl_per_unit
from agents.v3.network import Network
from scripts.replay_parser import pack_features
from scripts.train_full_bc import grouped_replay_split
from scripts.train_spatial_bc import (
    evaluate_distributions,
    select_manifest_replays,
    validate_reference_split,
)


def gaussian_logits(centers, sigma=1.5):
    centers = torch.as_tensor(centers, dtype=torch.float32)
    axis = torch.arange(24)
    gx, gy = torch.meshgrid(axis, axis, indexing="ij")
    return (-((gx[None] - centers[:, 0, None, None]).square()
              + (gy[None] - centers[:, 1, None, None]).square()) / (2 * sigma**2))[:, None]


class SpatialKLTests(unittest.TestCase):
    def test_center_and_edge_targets_are_probabilities_with_same_mode(self):
        centers = [[12, 12], [0, 0], [23, 23], [0, 12]]
        p = normalize_spatial_targets(gaussian_heatmaps(centers))
        torch.testing.assert_close(p.flatten(1).sum(1), torch.ones(4))
        self.assertTrue(torch.isfinite(p).all())
        self.assertTrue((p >= 0).all())
        self.assertEqual(p.flatten(1).argmax(1).tolist(), [x * 24 + y for x, y in centers])
        self.assertLess(p[0].max().item(), 1)
        with self.assertRaisesRegex(ValueError, "positive"):
            normalize_spatial_targets(torch.zeros(1, 24, 24))

    def test_kl_matches_formula_sum_over_pixels_and_soft_ce_gradients(self):
        torch.manual_seed(42)
        z = torch.randn(2, 1, 24, 24, requires_grad=True)
        q = normalize_spatial_targets(gaussian_heatmaps([[0, 0], [12, 12]]))
        valid = torch.tensor([True, True])
        logp = F.log_softmax(z.flatten(1), dim=1)
        flat_q = q.flatten(1)
        soft_ce = -(flat_q * logp).sum(1).mean()
        entropy = -torch.special.xlogy(flat_q, flat_q).sum(1).mean()
        kl = masked_spatial_kl(z, q, valid)
        torch.testing.assert_close(kl, soft_ce - entropy)
        expected_terms = torch.nn.KLDivLoss(reduction="none")(logp, flat_q)
        torch.testing.assert_close(kl, expected_terms.sum(1).mean())
        grad_kl = torch.autograd.grad(kl, z, retain_graph=True)[0]
        grad_ce = torch.autograd.grad(soft_ce, z)[0]
        torch.testing.assert_close(grad_kl, grad_ce)

    def test_matching_distribution_has_zero_kl_and_near_miss_costs_less(self):
        q = normalize_spatial_targets(gaussian_heatmaps([[12, 12]]))
        exact = spatial_kl_per_unit(gaussian_logits([[12, 12]]), q)
        near = spatial_kl_per_unit(gaussian_logits([[13, 12]]), q)
        far = spatial_kl_per_unit(gaussian_logits([[23, 23]]), q)
        self.assertLess(abs(exact.item()), 1e-6)
        self.assertLess(near.item(), far.item())
        with self.assertRaisesRegex(ValueError, "summing to one"):
            spatial_kl_per_unit(gaussian_logits([[12, 12]]), gaussian_heatmaps([[12, 12]]))

    def test_invalid_nan_units_are_neutral_before_log_softmax(self):
        z = torch.zeros(2, 1, 24, 24)
        z[1] = float("nan")
        z.requires_grad_()
        q = normalize_spatial_targets(gaussian_heatmaps([[0, 0], [12, 12]]))
        q[1] = float("nan")
        valid = torch.tensor([True, False])
        loss = masked_spatial_kl(z, q, valid)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(z.grad[1].count_nonzero().item(), 0)
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertEqual(masked_spatial_kl(z, q, torch.tensor([False, False])).item(), 0)

    def test_network_returns_unbounded_logits_without_final_activation(self):
        torch.set_num_threads(2)
        model = Network().eval()
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.fill_(4)
            output = model(torch.zeros(1, 35, 24, 24))
        torch.testing.assert_close(output, torch.full_like(output, 4))

    def test_active_dataset_normalizes_labels_in_memory_and_preserves_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1_p0.npz"
            features = np.zeros((2, 25, 24, 24), dtype=np.float32)
            valid = np.zeros((2, 16), dtype=bool)
            valid[:, 0] = True
            positions = np.full((2, 16, 2), -1, dtype=np.int16)
            positions[:, 0] = (0, 0)
            labels = np.zeros((2, 16, 24, 24), dtype=np.uint8)
            labels[:, 0, 0, 1] = 1
            np.savez_compressed(path, schema_version=np.uint8(3), **pack_features(features),
                                intent_valid=valid, active=valid, positions=positions,
                                intent_target=labels, steps=np.arange(2), match_steps=np.arange(2))
            original = path.read_bytes()
            x, q, v = UnitFrameDataset([path])[0]
            self.assertEqual(x.shape[1], 35)
            self.assertEqual(q.dtype, torch.float32)
            torch.testing.assert_close(q.sum((1, 2)), torch.ones(1))
            self.assertEqual(q.flatten(1).argmax(1).item(), 1)
            self.assertTrue(v.all())
            self.assertEqual(original, path.read_bytes())

    def test_raw_distribution_metrics_and_masked_decoding_are_separate(self):
        x = torch.zeros(2, 35, 24, 24)
        x[:, 32, 12, 12] = 1
        x[:, 11, 12, 12] = 100 / 6400
        q = normalize_spatial_targets(gaussian_heatmaps([[12, 13], [12, 12]]))
        rows = [(x, q, torch.tensor([True, True]))]
        result = evaluate_distributions(lambda a: gaussian_logits([[12, 12]] * len(a)),
                                        rows, torch.device("cpu"), unit_batch=1)
        self.assertAlmostEqual(result["soft_ce"], result["kl"] + result["target_entropy"], places=5)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["within_1"], 1)
        self.assertEqual(result["anchor_prediction_rate"], 1)
        self.assertEqual(result["mean_manhattan"], 0.5)

    def test_full_manifest_and_reference_require_exact_320_80_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            paths = [folder / f"{100000+i}_p0.npz" for i in range(400)]
            for path in paths:
                path.write_bytes(b"selection-only fixture")
            manifest = {"schema_version": 3, "intent_horizon": 12,
                        "replays": [{"output": str(path)} for path in paths]}
            (folder / "manifest.json").write_text(json.dumps(manifest))
            selected = select_manifest_replays(folder)
            self.assertEqual(len(selected), 400)
            train, val = grouped_replay_split(selected)
            reference = {"npz_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in selected},
                         "train_files": [p.name for p in train], "validation_files": [p.name for p in val]}
            validate_reference_split(selected, train, val, reference)
            self.assertEqual((len(train), len(val)), (320, 80))
            with self.assertRaisesRegex(ValueError, "group split"):
                validate_reference_split(selected, val, train, reference)
            manifest["replays"].pop()
            (folder / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "exactly 400"):
                select_manifest_replays(folder)


if __name__ == "__main__":
    unittest.main()
