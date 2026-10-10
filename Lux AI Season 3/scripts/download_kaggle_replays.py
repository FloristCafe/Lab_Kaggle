#!/usr/bin/env python3
"""Download latest strict wins from the Lux AI Season 3 private leaderboard.

All requests share a rate limiter and bounded retries. Team lists and selection
metadata are checkpointed on WSL ext4; credentials are never saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

COMPETITION_ID = 86411
SAVE_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_replays"
API = "https://www.kaggle.com/api/i/competitions."
LEADERBOARD_URL = API + "LeaderboardService/GetLeaderboard"
EPISODES_URL = API + "EpisodeService/ListEpisodes"
REPLAY_URL = "https://www.kaggle.com/competitions/episodes/{}/replay.json"


class KaggleApiError(RuntimeError):
    pass


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != "https":
            raise KaggleApiError("拒绝非 HTTPS 重定向。")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and urlparse(req.full_url).netloc != urlparse(newurl).netloc:
            for key in list(redirected.headers):
                if key.lower() in {"cookie", "authorization"} or key.lower().startswith("x-"):
                    del redirected.headers[key]
        return redirected


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise KaggleApiError(f"不能读取 JSON：{path.name}") from exc


def load_cookie(cookie_file: Path | None) -> str:
    value = (cookie_file.read_text(encoding="utf-8").strip() if cookie_file
             else os.environ.get("KAGGLE_COOKIE", "").strip())
    if not value or "=" not in value:
        raise KaggleApiError("请在当前终端设置 KAGGLE_COOKIE，或传入 --cookie-file。")
    return value


def retry_delay(value: str | None, attempt: int) -> float:
    if value:
        try:
            seconds = float(value)
            if math.isfinite(seconds):
                return max(0.0, seconds)
        except ValueError:
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                pass
    return min(60.0 * 2 ** attempt, 900.0) + random.uniform(0, 3)


class KaggleClient:
    def __init__(self, cookie: str, interval: float = 3.0, retries: int = 5):
        cookies = dict(part.strip().split("=", 1) for part in cookie.split(";") if "=" in part)
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 Lux-S3-ReplayDownloader/2.0",
            "Referer": "https://www.kaggle.com/competitions/lux-ai-season-3/leaderboard",
        }
        if cookie:
            self.headers["Cookie"] = cookie
        token = cookies.get("XSRF-TOKEN") or cookies.get("CSRF-TOKEN")
        if token:
            self.headers["X-XSRF-TOKEN"] = unquote(token)
        if cookies.get("build-hash"):
            self.headers["X-Kaggle-Build-Version"] = cookies["build-hash"]
        self.interval = interval
        self.retries = retries
        self.next_request = 0.0
        self.opener = build_opener(SafeRedirect())

    def request_json(self, url: str, payload: dict | None = None) -> Any:
        headers = dict(self.headers)
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers.update({"Content-Type": "application/json", "Origin": "https://www.kaggle.com"})
        for attempt in range(self.retries + 1):
            delay = max(0.0, self.next_request - time.monotonic())
            if delay:
                time.sleep(delay)
            self.next_request = time.monotonic() + self.interval
            try:
                request = Request(url, data=data, headers=headers, method="POST" if data else "GET")
                with self.opener.open(request, timeout=120) as response:
                    body = response.read()
                try:
                    result = json.loads(body)
                except ValueError as exc:
                    raise KaggleApiError("服务返回非 JSON，可能是登录页或接口已改变。") from exc
                if isinstance(result, dict) and result.get("error"):
                    raise KaggleApiError("服务在 JSON 中返回 error，未写入回放。")
                return result
            except HTTPError as exc:
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.retries:
                    wait = retry_delay(exc.headers.get("Retry-After"), attempt)
                    print(f"HTTP {exc.code}：等待 {wait:.1f} 秒后重试 ({attempt + 1}/{self.retries})", flush=True)
                    self.next_request = max(self.next_request, time.monotonic() + wait)
                    exc.close()
                    continue
                status = exc.code
                exc.close()
                hint = "Cookie 可能失效或访问被拒绝。" if status in {401, 403} else "名单与已完成下载已保存，可重新运行续传。"
                raise KaggleApiError(f"Kaggle HTTP {status}。{hint}") from exc
            except (URLError, TimeoutError, ConnectionError) as exc:
                if attempt < self.retries:
                    wait = min(5 * 2 ** attempt, 60)
                    print(f"网络异常：{wait} 秒后重试 ({attempt + 1}/{self.retries})", flush=True)
                    self.next_request = max(self.next_request, time.monotonic() + wait)
                    continue
                raise KaggleApiError("网络请求失败；已保存进度，可重新运行。") from exc
        raise AssertionError("unreachable")


def rows_from_leaderboard(payload: Any) -> list[dict]:
    # DEFAULT can include both boards. Explicitly select the private rows.
    rows = payload.get("privateLeaderboard") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise KaggleApiError("响应缺少 privateLeaderboard，不使用公榜替代。")
    names = {int(t["teamId"]): t.get("teamName", str(t["teamId"]))
             for t in payload.get("teams", [])}
    result = []
    for row in rows:
        if not all(key in row for key in ("rank", "teamId", "submissionId")):
            raise KaggleApiError("私榜行缺少 rank/teamId/submissionId。")
        result.append({"rank": int(row["rank"]), "id": int(row["teamId"]),
                       "name": names.get(int(row["teamId"]), str(row["teamId"])),
                       "submission_id": int(row["submissionId"])})
    return sorted(result, key=lambda t: (t["rank"], t["id"]))


def episode_rows(payload: Any) -> list[dict]:
    rows = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or any(not isinstance(ep, dict) or "id" not in ep for ep in rows):
        raise KaggleApiError("ListEpisodes 返回的 episodes 格式不匹配。")
    return rows


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def winner_side(ep: dict, team: dict) -> int | None:
    if ep.get("state") != "COMPLETED":
        return None
    agents = ep.get("agents", [])
    if not isinstance(agents, list) or len(agents) != 2:
        raise KaggleApiError(f"episode {ep.get('id')} 缺少两方 agents 元数据。")
    rewards = [numeric(a.get("reward")) for a in agents]
    if any(reward is None for reward in rewards):
        return None
    # Protobuf JSON omits a zero-valued index. Omitted index therefore means 0.
    sides = [int(a.get("index", 0)) for a in agents]
    if sorted(sides) != [0, 1]:
        raise KaggleApiError("agents index 无法对齐双方身份。")
    ours = [i for i, a in enumerate(agents)
            if a.get("submissionId") == team["submission_id"] and a.get("teamId") == team["id"]]
    if len(ours) != 1:
        raise KaggleApiError(f"episode {ep['id']} 的队伍/提交身份与排行榜不一致。")
    i = ours[0]
    return sides[i] if rewards[i] > rewards[1 - i] else None


def select_wins(rows: list[dict], team: dict, count: int, scan_limit: int) -> list[dict]:
    completed = [ep for ep in rows if ep.get("state") == "COMPLETED"]
    if any(not isinstance(ep.get("createTime"), str) for ep in completed):
        raise KaggleApiError("对局缺少 createTime，不能保证按最新对局筛选。")
    def date_key(ep):
        return datetime.fromisoformat(ep["createTime"].replace("Z", "+00:00")), int(ep["id"])
    try:
        ordered = sorted(completed, key=date_key, reverse=True)[:scan_limit]
    except ValueError as exc:
        raise KaggleApiError("createTime 格式错误。") from exc
    wins = []
    seen = set()
    for ep in ordered:
        side = winner_side(ep, team)
        if side is not None and int(ep["id"]) not in seen:
            wins.append({"id": int(ep["id"]), "winner_side": side, "create_time": ep["createTime"],
                         "end_time": ep.get("endTime"), "agents": ep["agents"]})
            seen.add(int(ep["id"]))
            if len(wins) == count:
                break
    return wins


def validate_replay(payload: Any, episode_id: int, side: int) -> None:
    if not isinstance(payload, dict) or payload.get("name") != "lux_ai_s3":
        raise KaggleApiError(f"episode {episode_id} 不是 Lux AI Season 3 回放。")
    info = payload.get("info")
    if not isinstance(info, dict) or info.get("EpisodeId") != episode_id:
        raise KaggleApiError(f"回放 EpisodeId 与目标 {episode_id} 不一致。")
    steps = payload.get("steps")
    if not isinstance(steps, list) or len(steps) < 2 or any(
        not isinstance(frame, list) or len(frame) != 2 or any(
            not isinstance(agent, dict) or "observation" not in agent for agent in frame
        ) for frame in steps
    ):
        raise KaggleApiError(f"episode {episode_id} 缺少有效双方逐帧观测。")
    rewards = payload.get("rewards")
    if payload.get("statuses") != ["DONE", "DONE"] or not isinstance(rewards, list) or len(rewards) != 2:
        raise KaggleApiError(f"episode {episode_id} 未正常完赛。")
    scores = [numeric(value) for value in rewards]
    if any(value is None for value in scores) or scores[side] <= scores[1 - side]:
        raise KaggleApiError(f"episode {episode_id} 回放未证实被选择的队伍获胜。")


def plan_teams(client: KaggleClient, args, checkpoint: Path) -> list[dict]:
    board_path = checkpoint / "leaderboard.json"
    if board_path.exists():
        board = read_json(board_path)
    else:
        board = client.request_json(LEADERBOARD_URL, {
            "competitionId": args.competition_id, "leaderboardMode": "LEADERBOARD_MODE_DEFAULT"})
        rows_from_leaderboard(board)
        atomic_json(board_path, board)
    return rows_from_leaderboard(board)[:args.top_teams]


def download_selected(client: KaggleClient, args, manifest: dict, manifest_path: Path, counts: dict) -> bool:
    for key, entry in sorted(manifest["episodes"].items(), key=lambda item: int(item[0]), reverse=True):
        episode_id = int(key)
        side = entry["sources"][0]["winner_side"]
        target = args.save_dir / f"{episode_id}.json"
        if episode_id in counts["checked"]:
            continue
        if target.exists():
            try:
                validate_replay(read_json(target), episode_id, side)
            except KaggleApiError:
                print(f"existing {target.name} 无效，将重新下载", flush=True)
            else:
                counts["existing"] += 1
                counts["checked"].add(episode_id)
                entry["download"] = {"status": "validated", "bytes": target.stat().st_size,
                                     "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
                atomic_json(manifest_path, manifest)
                print(f"skip validated {target.name}", flush=True)
                continue
        if args.max_downloads and counts["downloaded"] >= args.max_downloads:
            return False
        print(f"download {episode_id}", flush=True)
        try:
            payload = client.request_json(REPLAY_URL.format(episode_id))
            validate_replay(payload, episode_id, side)
            atomic_json(target, payload)
        except KaggleApiError:
            entry["download"] = {"status": "failed", "time": utc_now()}
            atomic_json(manifest_path, manifest)
            raise
        counts["downloaded"] += 1
        counts["checked"].add(episode_id)
        entry["download"] = {"status": "validated", "bytes": target.stat().st_size,
                             "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        atomic_json(manifest_path, manifest)
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--competition-id", type=int, default=COMPETITION_ID)
    p.add_argument("--top-teams", type=int, default=1)
    p.add_argument("--wins-per-team", type=int, default=5)
    p.add_argument("--episode-scan-limit", type=int, default=200)
    p.add_argument("--save-dir", type=Path, default=SAVE_DIR)
    p.add_argument("--cookie-file", type=Path)
    p.add_argument("--request-interval", type=float, default=3.0, help="所有网络请求的最小间隔秒数")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--max-downloads", type=int, default=0, help="本次新下载上限；0 表示不限")
    p.add_argument("--download-only", action="store_true", help="仅下载已保存清单，不再请求排行榜/对局列表")
    p.add_argument("--dry-run", action="store_true", help="只保存筛选清单，不下载回放")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.competition_id != COMPETITION_ID:
        raise KaggleApiError("本脚本只处理 Lux AI Season 3 (competitionId=86411)。")
    if min(args.top_teams, args.wins_per_team, args.episode_scan_limit) < 1:
        raise KaggleApiError("队伍数、胜局数、扫描数必须为正整数。")
    if args.episode_scan_limit < args.wins_per_team:
        raise KaggleApiError("扫描数不能小于每队胜局数。")
    if not math.isfinite(args.request_interval) or args.request_interval < 1.2 or min(args.max_retries, args.max_downloads) < 0:
        raise KaggleApiError("请求间隔至少 1.2 秒，重试数与下载上限不能为负。")
    if args.download_only and args.dry_run:
        raise KaggleApiError("--download-only 与 --dry-run 不能同时使用。")
    client = KaggleClient(load_cookie(args.cookie_file), args.request_interval, args.max_retries)
    checkpoint = args.save_dir.parent / "kaggle_downloads" / str(args.competition_id)
    manifest_path = checkpoint / f"selection_top{args.top_teams}_wins{args.wins_per_team}_scan{args.episode_scan_limit}.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {
        "schema_version": 2, "competition_id": args.competition_id,
        "created_at": utc_now(), "selection": {"top_teams": args.top_teams,
            "wins_per_team": args.wins_per_team, "scan_limit": args.episode_scan_limit,
            "order": "createTime descending", "filter": "strict win, matching team and submission"},
        "teams": {}, "episodes": {},
    }
    counts = {"downloaded": 0, "existing": 0, "checked": set()}
    try:
        if args.download_only:
            if not manifest["episodes"]:
                raise KaggleApiError("当前参数没有已保存清单；先运行普通模式或 --dry-run。")
            download_selected(client, args, manifest, manifest_path, counts)
        else:
            teams = plan_teams(client, args, checkpoint)
            print(f"selected teams: {len(teams)} (private leaderboard)", flush=True)
            for team in teams:
                cache_path = checkpoint / f"submission_{team['submission_id']}.json"
                if cache_path.exists():
                    payload = read_json(cache_path)
                else:
                    payload = client.request_json(EPISODES_URL, {"ids": [], "submissionId": team["submission_id"],
                        "successfulOnly": True, "includeInProgress": False})
                    episode_rows(payload)
                    atomic_json(cache_path, payload)
                rows = episode_rows(payload)
                wins = select_wins(rows, team, args.wins_per_team, args.episode_scan_limit)
                manifest["teams"][str(team["id"])] = {**team, "available_episodes": len(rows),
                    "selected_wins": len(wins), "episode_ids": [ep["id"] for ep in wins]}
                for ep in wins:
                    entry = manifest["episodes"].setdefault(str(ep["id"]), {"episode": ep, "sources": []})
                    source = {**team, "winner_side": ep["winner_side"]}
                    if source not in entry["sources"]:
                        entry["sources"].append(source)
                manifest["updated_at"] = utc_now()
                atomic_json(manifest_path, manifest)
                print(f"rank={team['rank']} {team['name']} team={team['id']}: verified wins={len(wins)}/{args.wins_per_team}", flush=True)
                if len(wins) < args.wins_per_team:
                    print("当前列表/扫描范围内胜局不足，保留实际数量，不用败局补足。", flush=True)
                if not args.dry_run and not download_selected(client, args, manifest, manifest_path, counts):
                    print("达到本次 --max-downloads 上限，可用相同命令续传。", flush=True)
                    break
    finally:
        print(f"unique selected: {len(manifest['episodes'])}; downloaded: {counts['downloaded']}; validated existing: {counts['existing']}", flush=True)
        print(f"checkpoint: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KaggleApiError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("已停止；已完成文件和清单保留，原命令可续传。", file=sys.stderr)
        raise SystemExit(130)
