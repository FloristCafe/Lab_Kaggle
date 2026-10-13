import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.v3.dataset import (
    LegacyProjectionFrameDataset as UnitFrameDataset,
)
from agents.v3.dataset import (
    ShardFrameSampler,
    collate_unit_frames,
)
from scripts.replay_parser import pack_features
from scripts.train_full_bc import (
    grouped_replay_split,
    score_model,
    train_epoch,
    unit_intent_loss,
)


def write_shard(path):
    features = np.zeros((3, 25, 24, 24), dtype=np.float32)
    features[:, 21:25] = 1
    features[:, 9] = -0.75
    target = np.zeros((3, 16, 24, 24), dtype=np.uint8)
    valid = np.zeros((3, 16), dtype=bool)
    valid[0, [2, 7]] = True
    valid[2, 7] = True
    target[valid, 3, 5] = 1
    positions = np.full((3, 16, 2), -1, dtype=np.int16)
    positions[0, 2], positions[0, 7], positions[2, 7] = (4, 8), (9, 2), (5, 3)
    np.savez_compressed(path, schema_version=np.uint8(3), **pack_features(features),
                        intent_target=target, intent_valid=valid, active=valid.copy(),
                        positions=positions, steps=np.arange(3), match_steps=np.arange(3))


class UnitDatasetTests(unittest.TestCase):
    def test_anchors_targets_frame_dropout_and_cache_are_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"{i}_p0.npz" for i in range(2)]
            for path in paths:
                write_shard(path)
            dataset = UnitFrameDataset(paths, future_dropout=1)
            self.assertEqual(len(dataset), 4)  # Entirely invalid frame excluded.
            x, y, dropped = dataset[0]
            self.assertEqual(tuple(x.shape), (2, 28, 24, 24))
            self.assertEqual(y.tolist(), [3 * 24 + 5] * 2)
            self.assertTrue(dropped)
            self.assertEqual(x[:, 25].sum((1, 2)).tolist(), [1, 1])
            self.assertEqual(x[0, 25, 4, 8].item(), 1)
            self.assertEqual(x[1, 25, 9, 2].item(), 1)
            self.assertEqual(x[0, 26, 0, 7].item(), -1)
            self.assertEqual(x[0, 26, 23, 7].item(), 1)
            self.assertEqual(x[0, 27, 7, 0].item(), -1)
            self.assertEqual(x[0, 27, 7, 23].item(), 1)
            torch.testing.assert_close(x[0, 26:28], x[1, 26:28])
            self.assertEqual(x[:, 21:25].count_nonzero().item(), 0)
            self.assertEqual(x[0, 9, 0, 0].item(), -0.75)
            dataset[2]
            self.assertEqual(len(dataset.cache), 1)
            with np.load(paths[0]) as data:
                self.assertEqual(data["feature_masks"].shape[1], 18)
                self.assertEqual(data["feature_values"].shape[1], 7)
                self.assertEqual(data["feature_masks"][:, -4:].sum(), 3 * 4 * 24 * 24)
            batch = collate_unit_frames([dataset[0], dataset[1]])
            self.assertEqual(batch[0].shape[0], 3)
            self.assertEqual(batch[2:], (2, 2))

    def test_sampler_visits_once_and_group_split_scales_to_400(self):
        paths = [Path(f"{i}_p{side}.npz") for i in range(400) for side in (0, 1)]
        train, val = grouped_replay_split(paths)
        self.assertEqual((len(train), len(val)), (640, 160))
        self.assertFalse({p.stem.rsplit("_p", 1)[0] for p in train} &
                         {p.stem.rsplit("_p", 1)[0] for p in val})
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"{i}_p0.npz" for i in range(5)]
            for path in paths:
                write_shard(path)
            dataset = UnitFrameDataset(paths)
            sampler = ShardFrameSampler(dataset, seed=42)
            self.assertEqual(sorted(sampler), list(range(len(dataset))))
            first = list(sampler)
            sampler.set_epoch(1)
            self.assertNotEqual(first, list(sampler))

    def test_unit_loss_matches_spatial_ce_and_microbatch_scoring(self):
        torch.set_num_threads(2)
        logits = torch.randn(3, 1, 24, 24, requires_grad=True)
        labels = torch.tensor([1, 77, 575])
        loss = unit_intent_loss(logits, labels)
        torch.testing.assert_close(loss, nn.functional.cross_entropy(logits.flatten(1), labels))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1_p0.npz"
            write_shard(path)
            data = UnitFrameDataset([path])
            loader = DataLoader(data, batch_size=2, collate_fn=collate_unit_frames)
            model = nn.Conv2d(28, 1, 1)
            full = score_model(model, loader, torch.device("cpu"), False, unit_batch=32)
            micro = score_model(model, loader, torch.device("cpu"), False, unit_batch=1)
            self.assertAlmostEqual(full[0], micro[0], places=5)
            self.assertEqual(full[1:], micro[1:])
            original = {key: tensor.clone() for key, tensor in model.state_dict().items()}
            train_epoch(model, loader, torch.optim.Adam(model.parameters()),
                        torch.device("cpu"), False, unit_batch=1)
            self.assertTrue(any(not torch.equal(value, model.state_dict()[key])
                                for key, value in original.items()))


if __name__ == "__main__":
    unittest.main()
