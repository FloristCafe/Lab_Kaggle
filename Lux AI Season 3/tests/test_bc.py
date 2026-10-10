import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.v3.network import Network
from scripts.train_bc import ReplayFrameDataset, masked_intent_loss

torch.set_num_threads(4)


class BehaviorCloningTests(unittest.TestCase):
    def test_network_preserves_coordinates_and_outputs_logits(self):
        model = Network()
        self.assertEqual(model(torch.zeros(2, 21, 24, 24)).shape, (2, 16, 24, 24))
        for layer in model.modules():
            self.assertNotIsInstance(layer, (nn.MaxPool2d, nn.AvgPool2d, nn.Dropout, nn.Softmax))
            if isinstance(layer, nn.Conv2d):
                self.assertEqual(layer.stride, (1, 1))
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 22, 24, 24))

    def test_masked_loss_and_gradient_match_valid_units_only(self):
        logits = torch.randn(2, 3, 4, 5, requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[:, :, 2, 3] = 1  # Flatten index must be 2*5+3, not 3*4+2.
        valid = torch.tensor([[True, False, True], [False, False, True]])
        loss = masked_intent_loss(logits, targets, valid)
        expected = nn.functional.cross_entropy(logits[valid].flatten(1), torch.full((3,), 13))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(int(torch.count_nonzero(logits.grad[~valid])), 0)
        changed = logits.detach().clone()
        changed[~valid] = float("nan")
        torch.testing.assert_close(masked_intent_loss(changed, targets, valid), loss.detach())

    def test_empty_mask_returns_zero_with_zero_gradients(self):
        logits = torch.full((1, 2, 3, 3), float("nan"), requires_grad=True)
        target = torch.zeros_like(logits)
        loss = masked_intent_loss(logits, target, torch.zeros(1, 2, dtype=torch.bool))
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertEqual(int(torch.count_nonzero(logits.grad)), 0)

    def test_dataset_is_repeatable_and_rejects_corrupt_valid_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay_p0.npz"
            features = np.zeros((4, 21, 24, 24), dtype=np.float32)
            target = np.zeros((4, 16, 24, 24), dtype=np.uint8)
            valid = np.zeros((4, 16), dtype=bool)
            valid[:3, 0] = True
            target[:3, 0, 4, 7] = 1
            fields = {"features": features, "intent_target": target, "intent_valid": valid,
                      "steps": np.arange(4), "match_steps": np.arange(4)}
            np.savez(path, **fields)
            first = ReplayFrameDataset(directory, size=3, seed=42)
            second = ReplayFrameDataset(directory, size=3, seed=42)
            self.assertEqual(first.records, second.records)
            self.assertEqual({r["frame"] for r in first.records}, {0, 1, 2})
            self.assertEqual(first[0][0].shape, (21, 24, 24))
            target[0, 0, 5, 7] = 1
            np.savez(path, **fields)
            with self.assertRaisesRegex(ValueError, "one-hot"):
                ReplayFrameDataset(directory, size=3)


if __name__ == "__main__":
    unittest.main()
