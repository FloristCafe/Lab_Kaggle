"""Feature/label, streaming deletion, and unit-centric loader contracts."""

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.replay_parser import dataset_manifest, infer_intents, pack_features
from scripts.stream_replays import (
    COMPATIBLE_PARSER_TRANSITIONS,
    main,
    migrate_compatible_parser,
    parse_batch,
    select_balanced_games,
    select_local_games,
    sha256,
)


def intent_path(path, enemies=None, active=None, match_steps=None, steps=None):
    positions = np.asarray(path, dtype=np.int16)[:, None]
    frames = len(path)
    actions = np.zeros((frames, 1, 3), dtype=np.int16)
    actions[:, 0, 0] = 2
    return infer_intents(positions,
                         np.ones((frames, 1), dtype=bool) if active is None else active,
                         actions, np.arange(frames) if match_steps is None else match_steps,
                         24, 24, enemy_occupied=enemies, observation_steps=steps)


class IntentFilterTests(unittest.TestCase):
    def test_reject_stationary_and_short_interior_moves(self):
        for path in ([(8, 8)] * 13, [(8, 8), (9, 8), (10, 8)] + [(10, 8)] * 10):
            target, valid = intent_path(path)
            self.assertFalse(valid[0, 0])
            self.assertEqual(int(target[0].sum()), 0)

    def test_cardinal_enemy_and_boundary_exempt_but_diagonal_does_not(self):
        enemies = np.zeros((13, 24, 24), dtype=bool)
        enemies[0, 9, 8] = True
        self.assertTrue(intent_path([(8, 8)] * 13, enemies)[1][0, 0])
        enemies[0, 9, 8] = False
        enemies[0, 9, 9] = True
        self.assertFalse(intent_path([(8, 8)] * 13, enemies)[1][0, 0])
        self.assertTrue(intent_path([(0, 8)] * 13)[1][0, 0])

    def test_includes_twelfth_future_step_and_preserves_round_trip(self):
        path = [(8, 8)] * 12 + [(11, 8)]
        self.assertTrue(intent_path(path)[1][0, 0])
        self.assertTrue(intent_path([(8, 8), (11, 8)] + [(8, 8)] * 11)[1][0, 0])
        self.assertFalse(intent_path([(8, 8)] * 13 + [(11, 8)])[1][0, 0])

    def test_window_stops_at_death_reset_and_missing_observation(self):
        path = [(8, 8), (9, 8), (10, 8), (11, 8)]
        active = np.array([[True], [True], [False], [True]])
        self.assertFalse(intent_path(path, active=active)[1][0, 0])
        self.assertFalse(intent_path(path, match_steps=[0, 1, 0, 1])[1][0, 0])
        self.assertFalse(intent_path(path, steps=[0, 1, 3, 4])[1][0, 0])

    def test_quantization_keeps_masks_exact_and_signed_values(self):
        features = np.zeros((2, 25, 24, 24), dtype=np.float32)
        features[:, 21, 5, 7] = 1
        features[:, 9, 5, 7] = -0.75
        packed = pack_features(features)
        self.assertEqual(packed["feature_masks"].dtype, np.uint8)
        self.assertEqual(packed["feature_values"].dtype, np.float16)
        self.assertLess(float(packed["feature_values"].min()), 0)
        features[:, 21, 5, 7] = 0.3
        with self.assertRaisesRegex(ValueError, "binary"):
            pack_features(features)


class StreamTests(unittest.TestCase):
    def test_compatible_parser_migration_preserves_npz_and_backs_up_manifest(self):
        old_hash, new_hash = next(iter(COMPATIBLE_PARSER_TRANSITIONS))
        self.assertEqual(new_hash, dataset_manifest([], {})["parser_sha256"])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            npz = out / "1_p0.npz"
            npz.write_bytes(b"verified completed data")
            manifest = {"parser_sha256": old_hash, "replays": [{"output": str(npz)}],
                        "episodes": {"1": {"status": "parsed", "output_sha256": {npz.name: sha256(npz)}}}}
            path = out / "manifest.json"
            path.write_text(json.dumps(manifest))
            before = npz.read_bytes()
            migrate_compatible_parser(manifest, path, dataset_manifest([], {}))
            self.assertEqual(npz.read_bytes(), before)
            self.assertEqual(manifest["parser_sha256"], new_hash)
            self.assertEqual(manifest["replays"][0]["parser_sha256"], old_hash)
            backup = out / f"manifest.before_parser_{old_hash[:12]}.json"
            self.assertEqual(json.loads(backup.read_text())["parser_sha256"], old_hash)
            migrate_compatible_parser(manifest, path, dataset_manifest([], {}))
            self.assertEqual(len(manifest["parser_history"]), 1)
            npz.write_bytes(b"changed")
            manifest["parser_sha256"] = old_hash
            with self.assertRaisesRegex(ValueError, "changed NPZ"):
                migrate_compatible_parser(manifest, path, dataset_manifest([], {}))

    def test_unknown_parser_change_is_not_silently_migrated(self):
        manifest = {"parser_sha256": "unknown"}
        with self.assertRaisesRegex(ValueError, "incompatibly"):
            migrate_compatible_parser(manifest, Path("unused.json"), dataset_manifest([], {}))

    def test_offline_selection_filters_rank_and_missing_local_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            selection = {"competition_id": 86411, "episodes": {}}
            for game, rank in ((1, 1), (2, 25), (3, 101), (4, 2)):
                selection["episodes"][str(game)] = {
                    "episode": {"id": game, "winner_side": 0, "create_time": "2025-03-25"},
                    "sources": [{"rank": rank, "id": rank, "winner_side": 0}],
                    "download": {"sha256": "expected"},
                }
                if game != 4:
                    (raw / f"{game}.json").write_text("{}")
            path = raw / "selection.json"
            path.write_text(json.dumps(selection))
            selected = select_local_games(path, raw, 2)
            self.assertEqual(set(selected), {"1", "2"})
            self.assertEqual(selected["1"]["episode"]["expected_raw_sha256"], "expected")
            with self.assertRaisesRegex(ValueError, "unique"):
                select_local_games(path, raw, 3)

    def test_offline_cli_never_loads_credentials_or_requests_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp) / "raw", Path(tmp) / "out"
            raw.mkdir()
            episodes = {}
            for game in range(1, 6):
                (raw / f"{game}.json").write_text("{}")
                episodes[str(game)] = {
                    "episode": {"id": game, "winner_side": 0, "create_time": "2025-03-25"},
                    "sources": [{"rank": 1, "id": 1, "winner_side": 0}],
                }
            selection = Path(tmp) / "selection.json"
            selection.write_text(json.dumps({"competition_id": 86411, "episodes": episodes}))
            argv = ["stream_replays.py", "--offline", "--selection-only", "--target-games", "5",
                    "--reuse-dir", str(raw), "--selection-manifest", str(selection),
                    "--data-dir", str(out)]
            with (patch.object(sys, "argv", argv),
                  patch("scripts.stream_replays.load_cookie", side_effect=AssertionError("Cookie requested")),
                  patch("scripts.stream_replays.KaggleClient", side_effect=AssertionError("network requested"))):
                main()
            saved = json.loads((out / "manifest.json").read_text())
            self.assertEqual(saved["selection_config"]["mode"], "offline")
            self.assertEqual(len(saved["episodes"]), 5)
            (raw / "5.json").unlink()
            argv.remove("--selection-only")
            with (patch.object(sys, "argv", argv),
                  patch("scripts.stream_replays.load_cookie", side_effect=AssertionError("Cookie requested")),
                  patch("scripts.stream_replays.KaggleClient", side_effect=AssertionError("network requested")),
                  self.assertRaisesRegex(FileNotFoundError, "no download attempted")):
                main()

    def test_balanced_selection_deduplicates_and_refuses_shortfall(self):
        teams = [({"rank": 1}, [{"id": 1, "winner_side": 0}, {"id": 2, "winner_side": 0}]),
                 ({"rank": 2}, [{"id": 1, "winner_side": 0}, {"id": 3, "winner_side": 1}])]
        self.assertEqual(list(select_balanced_games(teams, 3)), ["1", "2", "3"])
        with self.assertRaisesRegex(ValueError, "unique"):
            select_balanced_games(teams, 4)

    def test_burn_requires_verified_npz_and_durable_manifest(self):
        class FakeParser:
            def parse(self, raw, out, **kwargs):
                path = out / f"{raw.stem}_p0.npz"
                valid = np.zeros((1, 16), dtype=bool)
                valid[0, 0] = True
                target = np.zeros((1, 16, 24, 24), dtype=np.uint8)
                target[0, 0, 3, 5] = 1
                positions = np.full((1, 16, 2), -1, dtype=np.int16)
                positions[0, 0] = (4, 8)
                np.savez_compressed(path, schema_version=np.uint8(3),
                                    **pack_features(np.zeros((1, 25, 24, 24))),
                                    intent_valid=valid, active=valid.copy(),
                                    intent_target=target, positions=positions)
                return [{"source": str(raw), "output": str(path), "valid_intent_labels": 1}]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            raw = out / "123.json"
            raw.write_text('{"steps": []}')
            entry = {"winner_side": 0, "source": {"rank": 1}}
            manifest = dataset_manifest([], {})
            manifest["episodes"] = {"123": entry}
            manifest_path = out / "manifest.json"
            with (patch("scripts.stream_replays.atomic_json", side_effect=OSError("disk full")),
                  self.assertRaises(OSError)):
                parse_batch([(raw, entry)], out, manifest, manifest_path, FakeParser())
            self.assertTrue(raw.exists())
            with (patch("scripts.stream_replays.verify_npz", side_effect=ValueError("bad ZIP")),
                  self.assertRaises(ValueError)):
                parse_batch([(raw, entry)], out, manifest, manifest_path, FakeParser())
            self.assertTrue(raw.exists())
            digest = sha256(raw)
            parse_batch([(raw, entry)], out, manifest, manifest_path, FakeParser(), remove_raw=False)
            self.assertTrue(raw.exists())
            parse_batch([(raw, entry)], out, manifest, manifest_path, FakeParser())
            self.assertFalse(raw.exists())
            self.assertEqual(json.loads(manifest_path.read_text())["source_sha256"]["123.json"], digest)
            with zipfile.ZipFile(out / "123_p0.npz") as archive:
                self.assertTrue(all(x.compress_type == zipfile.ZIP_DEFLATED for x in archive.infolist()))


if __name__ == "__main__":
    unittest.main()
