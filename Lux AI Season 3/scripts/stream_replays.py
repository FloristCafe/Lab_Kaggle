"""Select top-100 strict wins, parse <=10 JSON at a time, verify NPZ, burn JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_kaggle_replays import (
    COMPETITION_ID,
    EPISODES_URL,
    REPLAY_URL,
    KaggleApiError,
    KaggleClient,
    atomic_json,
    episode_rows,
    load_cookie,
    plan_teams,
    read_json,
    select_wins,
    utc_now,
    validate_replay,
)
from scripts.replay_parser import (
    MASK_CHANNELS,
    SCHEMA_VERSION,
    VALUE_CHANNELS,
    OmniscientParser,
    dataset_manifest,
)

# This diagnostic-only fix leaves features, labels and candidate updates
# unchanged for every replay already accepted by the previous parser.
COMPATIBLE_PARSER_TRANSITIONS = {
    ("85b7d9863fbd2382476caeb8a8562b80c6842a2f4a5a1396e9a92ba89b29c91d",
     "ce209486b9d722fb605c193088e1459b447fa23238ac793b5e17d40be4de82f0"):
        "treat insufficient drift overlap as inconclusive, not conflicting",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def commit_manifest(path, payload):
    atomic_json(path, payload)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def migrate_compatible_parser(manifest, manifest_path, current):
    old_hash, new_hash = manifest.get("parser_sha256"), current["parser_sha256"]
    if old_hash == new_hash:
        return
    reason = COMPATIBLE_PARSER_TRANSITIONS.get((old_hash, new_hash))
    if reason is None:
        raise ValueError("parser changed incompatibly; choose a fresh data directory")
    for entry in manifest.get("episodes", {}).values():
        if entry.get("status") == "parsed":
            for name, expected in entry["output_sha256"].items():
                if sha256(manifest_path.parent / name) != expected:
                    raise ValueError(f"changed NPZ {name}; cannot migrate parser metadata")
    backup = manifest_path.with_name(f"manifest.before_parser_{old_hash[:12]}.json")
    if not backup.exists():
        commit_manifest(backup, manifest)
    for row in manifest["replays"]:
        row.setdefault("parser_sha256", old_hash)
    manifest.setdefault("parser_history", []).append({
        "from_sha256": old_hash, "to_sha256": new_hash,
        "reason": reason, "migrated_at": utc_now(),
        "preserved_npz_files": len(manifest["replays"]),
    })
    manifest["parser_sha256"] = new_hash
    manifest["drift_evidence_policy"] = current["drift_evidence_policy"]
    commit_manifest(manifest_path, manifest)
    print(f"compatible parser fix: preserved {len(manifest['replays'])} NPZ; manifest backed up", flush=True)


def select_balanced_games(team_wins, target):
    """Round-robin across ranks; deduplicate episodes before counting games."""
    selected = {}
    for offset in range(max((len(wins) for _, wins in team_wins), default=0)):
        for team, wins in team_wins:
            if offset >= len(wins):
                continue
            ep = wins[offset]
            key = str(ep["id"])
            if key not in selected and len(selected) < target:
                selected[key] = {"episode": ep, "source": team,
                                 "winner_side": ep["winner_side"]}
        if len(selected) == target:
            return selected
    raise ValueError(f"only {len(selected)} unique strict wins available; need {target}")


def select_games(client, args):
    checkpoint = ROOT / "data/kaggle_downloads" / str(COMPETITION_ID)
    teams = plan_teams(client, SimpleNamespace(competition_id=COMPETITION_ID,
                                             top_teams=args.top_teams), checkpoint)
    if len(teams) != args.top_teams:
        raise ValueError("leaderboard has fewer teams than requested")
    pools = []
    for team in teams:
        cache = checkpoint / f"submission_{team['submission_id']}.json"
        if cache.exists():
            payload = read_json(cache)
        else:
            payload = client.request_json(EPISODES_URL, {
                "ids": [], "submissionId": team["submission_id"],
                "successfulOnly": True, "includeInProgress": False})
            episode_rows(payload)
            atomic_json(cache, payload)
        wins = select_wins(episode_rows(payload), team, args.scan_limit, args.scan_limit)
        pools.append((team, wins))
        print(f"rank={team['rank']}: {len(wins)} strict wins available", flush=True)
    return select_balanced_games(pools, args.target_games)


def select_local_games(selection_path, raw_dir, target, top_teams=100):
    """Use only locally present, previously selected strict wins; never network."""
    selection = read_json(selection_path)
    if selection.get("competition_id") != COMPETITION_ID:
        raise ValueError("local selection manifest is not Lux AI Season 3")
    pools = {}
    for key, entry in selection.get("episodes", {}).items():
        if not (raw_dir / f"{int(key)}.json").is_file():
            continue
        episode = entry["episode"]
        if int(episode["id"]) != int(key):
            raise ValueError("local selection episode ID mismatch")
        for source in entry["sources"]:
            if not 1 <= int(source["rank"]) <= top_teams:
                continue
            side = int(source["winner_side"])
            if side not in (0, 1) or side != int(episode["winner_side"]):
                raise ValueError("local winner-side metadata mismatch")
            team = {k: v for k, v in source.items() if k != "winner_side"}
            team_key = (int(team["rank"]), int(team["id"]))
            if team_key not in pools:
                pools[team_key] = (team, [])
            pools[team_key][1].append({**episode,
                                      "expected_raw_sha256": entry.get("download", {}).get("sha256")})
    ordered = [pools[key] for key in sorted(pools)]
    for _, episodes in ordered:
        episodes.sort(key=lambda ep: (ep.get("create_time", ""), int(ep["id"])), reverse=True)
    return select_balanced_games(ordered, target)


def check_disk(data_dir, max_total_gib, reserve_gib):
    used = sum(p.stat().st_size for p in data_dir.rglob("*") if p.is_file())
    # Leave room for the next JSON and its temporary NPZ, not just final files.
    if used + 256 * 1024**2 > max_total_gib * 1024**3:
        raise RuntimeError("pipeline storage budget reached; no new download")
    volumes = [data_dir]
    if Path("/mnt/c").exists():
        volumes.append(Path("/mnt/c"))  # WSL ext4 free size alone is misleading.
    if any(shutil.disk_usage(p).free < reserve_gib * 1024**3 for p in volumes):
        raise RuntimeError("physical/free disk reserve reached; no new download")


def verify_npz(path):
    with np.load(path, allow_pickle=False) as data:
        if int(data["schema_version"]) != SCHEMA_VERSION:
            raise ValueError(f"{path}: wrong compact schema")
        for key in data.files:
            data[key]  # Inflate each ZIP member and verify its CRC before burning JSON.
        valid, active = data["intent_valid"], data["active"]
        target, positions = data["intent_target"], data["positions"]
        masks, values = data["feature_masks"], data["feature_values"]
        frames = len(valid)
        if (valid.shape != (frames, 16) or valid.dtype != np.bool_
                or active.shape != valid.shape or active.dtype != np.bool_
                or (valid & ~active).any()
                or target.shape != (frames, 16, 24, 24) or target.dtype != np.uint8
                or positions.shape != (frames, 16, 2)
                or ((positions[valid] < 0) | (positions[valid] >= 24)).any()):
            raise ValueError("invalid label/anchor schema")
        if not np.isin(target[valid], (0, 1)).all() or not (target[valid].sum((1, 2)) == 1).all():
            raise ValueError("valid targets must be one-hot")
        if (masks.dtype != np.uint8 or masks.shape != (frames, len(MASK_CHANNELS), 24, 24)
                or not np.isin(masks, (0, 1)).all()):
            raise ValueError("mask channels must remain binary uint8")
        if (values.dtype != np.float16 or values.shape != (frames, len(VALUE_CHANNELS), 24, 24)
                or not np.isfinite(values).all()):
            raise ValueError("numeric channels must remain finite float16")
        if "features" in data.files:
            raise ValueError("dense feature array must not be duplicated on disk")
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def burn_verified_raw(raw, entry, burn_existing=False):
    """Only remove the exact verified source, after the durable NPZ manifest."""
    if raw.exists():
        if sha256(raw) != entry["raw_sha256"]:
            raise ValueError("raw replay changed after parse; refusing deletion")
        os.remove(raw)
    reused = entry.get("reused_from")
    if burn_existing and reused and Path(reused).exists():
        original = Path(reused)
        if sha256(original) != entry["raw_sha256"]:
            raise ValueError("reused replay changed; refusing deletion")
        os.remove(original)


def parse_batch(batch, data_dir, manifest, manifest_path, parser, burn_existing=False,
                remove_raw=True):
    for raw, entry in batch:
        key = raw.stem
        raw_hash = sha256(raw)
        rows = parser.parse(raw, data_dir, overwrite=True, sides=(entry["winner_side"],))
        output_hashes = {}
        for row in rows:
            path = Path(row["output"])
            verify_npz(path)
            output_hashes[path.name] = sha256(path)
            row["source_sha256"] = raw_hash
            row["expert_rank"] = entry["source"]["rank"]
            row["parser_sha256"] = manifest["parser_sha256"]
        entry.update(status="parsed", raw_sha256=raw_hash,
                     output_sha256=output_hashes, parsed_at=utc_now())
        manifest["source_sha256"][raw.name] = raw_hash
        manifest["replays"] = [r for r in manifest["replays"]
                               if Path(r["source"]).stem != key] + rows
        manifest["updated_at"] = utc_now()
        commit_manifest(manifest_path, manifest)
        if remove_raw:
            burn_verified_raw(raw, entry, burn_existing)
        raw_status = "removed" if remove_raw else "retained"
        print(f"parsed {key}; valid units={sum(r['valid_intent_labels'] for r in rows)}; JSON {raw_status}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/processed_replays_unitcentric25")
    p.add_argument("--top-teams", type=int, default=100)
    p.add_argument("--target-games", type=int, default=400)
    p.add_argument("--scan-limit", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--cookie-file", type=Path)
    p.add_argument("--reuse-dir", type=Path, help="reuse existing JSON before network download")
    p.add_argument("--offline", action="store_true",
                   help="only existing local JSON and selection metadata; no Kaggle requests or Cookie")
    p.add_argument("--selection-manifest", type=Path,
                   help="existing downloader selection manifest, required for --offline")
    p.add_argument("--burn-existing", action="store_true", help="also delete selected reused JSON after verification")
    p.add_argument("--max-total-gib", type=float, default=40)
    p.add_argument("--reserve-gib", type=float, default=5)
    p.add_argument("--request-interval", type=float, default=3)
    p.add_argument("--selection-only", action="store_true")
    args = p.parse_args()
    if (not 1 <= args.top_teams <= 100 or not 5 <= args.target_games <= 500
            or not 1 <= args.batch_size <= 10 or args.scan_limit < 1
            or args.max_total_gib <= 0 or args.reserve_gib < 5
            or args.request_interval < 1.2):
        p.error("invalid team/game/batch/disk/rate limits")
    if args.burn_existing and args.reuse_dir is None:
        p.error("--burn-existing requires --reuse-dir")
    if args.offline and (args.reuse_dir is None or args.selection_manifest is None):
        p.error("--offline requires --reuse-dir and --selection-manifest")
    data_dir = args.data_dir.resolve()
    if data_dir.is_relative_to(Path("/mnt")):
        p.error("active NPZ and temporary JSON must be on WSL ext4")
    data_dir.mkdir(parents=True, exist_ok=True)
    spool = data_dir / ".raw-stream"
    spool.mkdir(exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    config = {"top_teams": args.top_teams, "target_games": args.target_games,
              "scan_limit": args.scan_limit, "views": "selected top-team winning side only"}
    if args.offline:
        config.update(mode="offline", raw_dir=str(args.reuse_dir.resolve()),
                      selection_manifest_sha256=sha256(args.selection_manifest))
    client = None

    def get_client():
        nonlocal client
        if args.offline:
            raise RuntimeError("offline mode forbids all Kaggle requests")
        if client is None:
            client = KaggleClient(load_cookie(args.cookie_file), args.request_interval)
        return client

    if manifest_path.exists():
        manifest = read_json(manifest_path)
        current = dataset_manifest([], {})
        if (manifest.get("selection_config") != config
                or manifest.get("schema_version") != SCHEMA_VERSION):
            p.error("manifest config/parser changed; choose a fresh data directory")
        migrate_compatible_parser(manifest, manifest_path, current)
    else:
        if list(data_dir.glob("*.npz")):
            p.error("untracked NPZ found; choose a fresh data directory")
        check_disk(data_dir, args.max_total_gib, args.reserve_gib)
        if args.offline:
            selected = select_local_games(args.selection_manifest, args.reuse_dir,
                                           args.target_games, args.top_teams)
        else:
            selected = select_games(get_client(), args)
        manifest = dataset_manifest([], {})
        manifest.update(selection_config=config, episodes=selected, created_at=utc_now())
        commit_manifest(manifest_path, manifest)
    if args.selection_only:
        print(f"selected {len(manifest['episodes'])} unique episodes; no replay download", flush=True)
        return
    parser = OmniscientParser(horizon=12)
    batch = []
    for key, entry in manifest["episodes"].items():
        raw = (args.reuse_dir if args.offline else spool) / f"{key}.json"
        if entry.get("status") == "parsed":
            for name, expected in entry["output_sha256"].items():
                if sha256(data_dir / name) != expected:
                    raise ValueError(f"changed/missing NPZ {name}; refusing to skip")
            if not args.offline or args.burn_existing:
                burn_verified_raw(raw, entry, args.burn_existing)
            continue
        check_disk(data_dir, args.max_total_gib, args.reserve_gib)
        if args.offline:
            if not raw.is_file():
                raise FileNotFoundError(f"offline replay missing: {raw}; no download attempted")
            expected = entry["episode"].get("expected_raw_sha256")
            if expected and sha256(raw) != expected:
                raise ValueError(f"{raw}: differs from previously downloaded replay")
            validate_replay(read_json(raw), int(key), entry["winner_side"])
            batch.append((raw, entry))
            if len(batch) == args.batch_size:
                parse_batch(batch, data_dir, manifest, manifest_path, parser,
                            remove_raw=args.burn_existing)
                batch.clear()
            continue
        reused = args.reuse_dir / raw.name if args.reuse_dir else None
        if not raw.exists():
            if reused and reused.is_file():
                validate_replay(read_json(reused), int(key), entry["winner_side"])
                shutil.copyfile(reused, raw)
                entry["reused_from"] = str(reused.resolve())
            else:
                payload = get_client().request_json(REPLAY_URL.format(key))
                validate_replay(payload, int(key), entry["winner_side"])
                atomic_json(raw, payload)
                del payload
        else:
            validate_replay(read_json(raw), int(key), entry["winner_side"])
            if reused and reused.is_file() and sha256(reused) == sha256(raw):
                entry["reused_from"] = str(reused.resolve())
        batch.append((raw, entry))
        if len(batch) == args.batch_size:
            parse_batch(batch, data_dir, manifest, manifest_path, parser, args.burn_existing)
            batch.clear()
    if batch:
        parse_batch(batch, data_dir, manifest, manifest_path, parser, args.burn_existing,
                    remove_raw=not args.offline or args.burn_existing)
    print(json.dumps({"parsed_games": sum(e.get("status") == "parsed" for e in manifest["episodes"].values()),
                      "npz_files": len(manifest["replays"]),
                      "valid_units": sum(r["valid_intent_labels"] for r in manifest["replays"]),
                      "manifest": str(manifest_path)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (KaggleApiError, OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}; rerun the same command to resume", file=sys.stderr)
        raise SystemExit(2)
