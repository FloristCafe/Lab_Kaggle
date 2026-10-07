import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from diagnose_replays import diagnose


def frame(step, match_step, mask):
    return {
        "steps": step,
        "match_steps": match_step,
        "units_mask": [mask, [False, False]],
        "units": {"position": [
            [[0, 0], [1, 0]], [[0, 0], [0, 0]]
        ]},
        "team_points": [0, 0],
    }


class ReplayDiagnosticTests(unittest.TestCase):
    def test_match_reset_does_not_count_as_unit_death(self):
        observations = [
            frame(0, 0, [False, False]),
            frame(1, 1, [True, True]),
            frame(2, 0, [True, True]),
            frame(3, 1, [True, False]),
            frame(4, 2, [False, False]),
            frame(5, 0, [False, False]),
        ]
        actions = [{"player_0": [[0, 0, 0], [0, 0, 0]]}] * 5
        with tempfile.TemporaryDirectory() as directory:
            replay = Path(directory) / "replay.json"
            replay.write_text(json.dumps({
                "observations": observations, "actions": actions
            }), encoding="utf-8")
            rows = diagnose(replay, "player_0")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["unit_losses"], 1)


if __name__ == "__main__":
    unittest.main()
