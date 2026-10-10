"""Downloader contract tests; no credentials or network requests required."""
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_kaggle_replays.py"
SPEC = importlib.util.spec_from_file_location("downloader", SCRIPT)
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


def team(n=1):
    return {"rank": n, "id": n, "name": f"team {n}", "submission_id": 100 + n}


def episode(n=10, own=None, side=0, rewards=(5, 0), date="2025-03-24T00:00:00Z"):
    own = own or team()
    agents = [{"teamId": own["id"], "submissionId": own["submission_id"], "reward": rewards[0]},
              {"teamId": 999, "submissionId": 999, "reward": rewards[1]}]
    agents[1 - side]["index"] = 1
    if side == 1:
        agents.reverse()
    return {"id": n, "state": "COMPLETED", "createTime": date, "agents": agents}


def replay(n=10, side=0):
    return {"name": "lux_ai_s3", "info": {"EpisodeId": n},
            "statuses": ["DONE", "DONE"], "rewards": [5, 0] if side == 0 else [0, 5],
            "steps": [[{"observation": {}}, {"observation": {}}] for _ in range(2)]}


class DownloaderTests(unittest.TestCase):
    def test_private_board_team_names_and_order(self):
        board = {"privateLeaderboard": [{"rank": 2, "teamId": 2, "submissionId": 102},
                                        {"rank": 1, "teamId": 1, "submissionId": 101}],
                 "teams": [{"teamId": 1, "teamName": "Flat Neurons"}]}
        rows = d.rows_from_leaderboard(board)
        self.assertEqual([t["id"] for t in rows], [1, 2])
        self.assertEqual(rows[0]["name"], "Flat Neurons")
        with self.assertRaises(d.KaggleApiError):
            d.rows_from_leaderboard({"publicLeaderboard": board["privateLeaderboard"]})

    def test_strict_wins_and_swapped_sides(self):
        self.assertEqual(d.winner_side(episode(), team()), 0)
        self.assertEqual(d.winner_side(episode(side=1), team()), 1)
        for rewards in [(0, 5), (3, 3), (None, 0), (float("nan"), 0)]:
            with self.subTest(rewards=rewards):
                self.assertIsNone(d.winner_side(episode(rewards=rewards), team()))
        ep = episode()
        ep["state"] = "RUNNING"
        self.assertIsNone(d.winner_side(ep, team()))
        with self.assertRaises(d.KaggleApiError):
            d.winner_side(episode(), team(2))

    def test_latest_creation_time_and_deduplication(self):
        old = episode(10)
        new = episode(11, date="2025-03-25T00:00:00Z")
        self.assertEqual([e["id"] for e in d.select_wins([old, new, new], team(), 2, 200)], [11, 10])

    def test_replay_rejects_wrong_game_identity_result_and_status(self):
        d.validate_replay(replay(), 10, 0)
        d.validate_replay(replay(side=1), 10, 1)
        for key, value in [("name", "lux_ai_2022"), ("info", {"EpisodeId": 11}),
                           ("statuses", ["ERROR", "DONE"]), ("rewards", [0, 5]),
                           ("steps", [[{"observation": {}}]])]:
            with self.subTest(key=key):
                payload = replay()
                payload[key] = value
                with self.assertRaises(d.KaggleApiError):
                    d.validate_replay(payload, 10, 0)

    def test_xsrf_header_priority_and_cross_host_redirect(self):
        client = d.KaggleClient("XSRF-TOKEN=correct%20value; CSRF-TOKEN=wrong; build-hash=build")
        self.assertEqual(client.headers["X-XSRF-TOKEN"], "correct value")
        req = Request("https://www.kaggle.com/replay", headers={
            "Cookie": "test=secret", "Authorization": "secret", "X-XSRF-TOKEN": "secret"})
        redirected = d.SafeRedirect().redirect_request(req, None, 302, "", {}, "https://storage.googleapis.com/replay")
        self.assertFalse(any(k.lower() in {"cookie", "authorization", "x-xsrf-token"}
                             for k in redirected.headers))
        with self.assertRaises(d.KaggleApiError):
            d.SafeRedirect().redirect_request(req, None, 302, "", {}, "http://www.kaggle.com/replay")

    def test_shared_rate_limiter_and_429_retry_after(self):
        clock = [0.0]
        sleeps = []
        requests = []
        response = io.BytesIO(b'{"ok": true}')
        another = io.BytesIO(b'{"ok": true}')
        error = HTTPError(d.EPISODES_URL, 429, "limited", {"Retry-After": "7"}, io.BytesIO())
        pending = [error, response, another]
        def sleep(seconds):
            sleeps.append(seconds)
            clock[0] += seconds
        def open_request(req, timeout):
            requests.append((clock[0], req.get_method()))
            result = pending.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        client = d.KaggleClient("", interval=3, retries=1)
        with patch.object(client.opener, "open", side_effect=open_request), \
             patch.object(d.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(d.time, "sleep", side_effect=sleep), redirect_stdout(io.StringIO()):
            client.request_json(d.EPISODES_URL, {"ids": []})
            client.request_json(d.REPLAY_URL.format(10))
        self.assertEqual(requests, [(0, "POST"), (7, "POST"), (10, "GET")])
        self.assertEqual(sleeps, [7, 3])

    def test_nonretryable_and_exhausted_http_errors(self):
        for status, retries, calls in [(401, 3, 1), (429, 0, 1)]:
            with self.subTest(status=status):
                client = d.KaggleClient("", retries=retries)
                error = HTTPError(d.EPISODES_URL, status, "", {}, io.BytesIO())
                with patch.object(client.opener, "open", side_effect=error) as opened:
                    with self.assertRaises(d.KaggleApiError):
                        client.request_json(d.EPISODES_URL, {})
                self.assertEqual(opened.call_count, calls)

    def test_invalid_download_does_not_write_replay(self):
        with tempfile.TemporaryDirectory() as root:
            args = SimpleNamespace(save_dir=Path(root) / "raw", max_downloads=0)
            manifest = {"episodes": {"10": {"sources": [{"winner_side": 0}]}}}
            counts = {"downloaded": 0, "existing": 0, "checked": set()}
            client = d.KaggleClient("")
            path = Path(root) / "selection.json"
            with patch.object(client, "request_json", return_value={"error": "oops"}), redirect_stdout(io.StringIO()):
                with self.assertRaises(d.KaggleApiError):
                    d.download_selected(client, args, manifest, path, counts)
            self.assertFalse((args.save_dir / "10.json").exists())
            self.assertEqual(d.read_json(path)["episodes"]["10"]["download"]["status"], "failed")

    def test_invalid_existing_replaced_and_valid_existing_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            args = SimpleNamespace(save_dir=Path(root), max_downloads=0)
            d.atomic_json(args.save_dir / "10.json", {"bad": "data"})
            client = d.KaggleClient("")
            manifest = {"episodes": {"10": {"sources": [{"winner_side": 0}]}}}
            with patch.object(client, "request_json", return_value=replay()) as request, redirect_stdout(io.StringIO()):
                for _ in range(2):
                    counts = {"downloaded": 0, "existing": 0, "checked": set()}
                    d.download_selected(client, args, manifest, Path(root) / "manifest.json", counts)
            self.assertEqual(request.call_count, 1)
            self.assertEqual(counts["existing"], 1)
            self.assertEqual(len(manifest["episodes"]["10"]["download"]["sha256"]), 64)

    def test_failed_second_team_preserves_first_and_resumes_from_cache(self):
        with tempfile.TemporaryDirectory() as root:
            args = SimpleNamespace(competition_id=86411, top_teams=2, wins_per_team=1,
                episode_scan_limit=200, save_dir=Path(root) / "raw_replays", cookie_file=None,
                request_interval=3, max_retries=0, max_downloads=0, dry_run=False, download_only=False)
            board = {"privateLeaderboard": [{"rank": n, "teamId": n, "submissionId": 100+n} for n in (1, 2)]}
            phase = [1]
            urls = []
            def request(client, url, payload=None):
                urls.append((url, copy.deepcopy(payload)))
                if url == d.LEADERBOARD_URL:
                    return board
                if url == d.EPISODES_URL:
                    n = payload["submissionId"] - 100
                    if n == 2 and phase[0] == 1:
                        raise d.KaggleApiError("HTTP 429 exhausted")
                    return {"episodes": [episode(10+n, own=team(n))]}
                return replay(int(url.split("/")[-2]))
            with patch.object(d, "parse_args", return_value=args), \
                 patch.object(d, "load_cookie", return_value="test=placeholder"), \
                 patch.object(d.KaggleClient, "request_json", new=request), redirect_stdout(io.StringIO()):
                with self.assertRaises(d.KaggleApiError):
                    d.main()
                manifest_path = Path(root) / "kaggle_downloads/86411/selection_top2_wins1_scan200.json"
                self.assertEqual(d.read_json(manifest_path)["episodes"]["11"]["download"]["status"], "validated")
                self.assertTrue((args.save_dir / "11.json").exists())
                phase[0] = 2
                urls.clear()
                self.assertEqual(d.main(), 0)
                self.assertEqual(urls, [(d.EPISODES_URL, {"ids": [], "submissionId": 102,
                    "successfulOnly": True, "includeInProgress": False}), (d.REPLAY_URL.format(12), None)])
                args.download_only = True
                urls.clear()
                self.assertEqual(d.main(), 0)
                self.assertEqual(urls, [])
            for path in (Path(root) / "kaggle_downloads").rglob("*.json"):
                self.assertNotIn("test=placeholder", path.read_text())


if __name__ == "__main__":
    unittest.main()
