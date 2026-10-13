import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_full_bc import drop_future_features, grouped_replay_split


class FullBehaviorCloningTests(unittest.TestCase):
    def test_group_fold_keeps_both_views_together(self):
        paths = [Path(f"game_{game}_p{side}.npz")
                 for game in range(5) for side in (0, 1)]
        seen_validation_games = set()
        for fold in range(5):
            train_paths, val_paths = grouped_replay_split(paths, fold)
            train_games = {path.stem.rsplit("_p", 1)[0] for path in train_paths}
            val_games = {path.stem.rsplit("_p", 1)[0] for path in val_paths}
            self.assertEqual(len(train_paths), 8)
            self.assertEqual(len(val_paths), 2)
            self.assertEqual(len(train_games), 4)
            self.assertEqual(len(val_games), 1)
            self.assertFalse(train_games & val_games)
            seen_validation_games.update(val_games)
        self.assertEqual(len(seen_validation_games), 5)

    def test_projection_dropout_preserves_base_channels(self):
        features = torch.ones(3, 25, 24, 24)
        dropped, count = drop_future_features(features, 1.0)
        self.assertEqual(count, 3)
        self.assertEqual(int(torch.count_nonzero(dropped[:, 21:25])), 0)
        self.assertTrue(torch.equal(dropped[:, :21], features[:, :21]))
        self.assertTrue(torch.equal(features, torch.ones_like(features)))
        unchanged, count = drop_future_features(features, 0.0)
        self.assertEqual(count, 0)
        self.assertTrue(torch.equal(unchanged, features))


if __name__ == "__main__":
    unittest.main()
