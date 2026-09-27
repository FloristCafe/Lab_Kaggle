"""Summarize completed Lux S3 replays without using hidden game state."""

import argparse
import csv
import json
import sys
from pathlib import Path


def summarize(path: Path, agent_name: str):
    replay = json.loads(path.read_text(encoding="utf-8"))
    observations = replay["observations"]
    players = replay["metadata"]["players"]
    matches = [
        player_id
        for player_id, agent_path in players.items()
        if Path(agent_path).parent.name == agent_name
    ]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one {agent_name} agent, found {len(matches)}")
    agent_id = matches[0]
    side = int(agent_id.removeprefix("player_"))
    other = 1 - side

    match_points = []
    for previous, current in zip(observations, observations[1:]):
        if current["match_steps"] < previous["match_steps"]:
            match_points.append(previous["team_points"])
    expected = int(replay["params"]["match_count_per_episode"])
    if len(match_points) != expected:
        raise ValueError(f"{path}: found {len(match_points)} of {expected} match scores")

    final_wins = observations[-1]["team_wins"]
    return {
        "seed": replay["metadata"]["seed"],
        "side": agent_id,
        "match_wins": final_wins[side],
        "other_match_wins": final_wins[other],
        "points": sum(points[side] for points in match_points),
        "other_points": sum(points[other] for points in match_points),
        "match_scores": " ".join(
            f"{points[side]}:{points[other]}" for points in match_points
        ),
        "replay": str(path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", nargs="+", type=Path)
    parser.add_argument("--agent", default="v0")
    args = parser.parse_args()
    rows = [summarize(path, args.agent) for path in args.replays]
    rows.sort(key=lambda row: (row["seed"], row["side"]))
    writer = csv.DictWriter(sys.stdout, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
