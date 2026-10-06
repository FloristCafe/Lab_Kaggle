"""Collect replay results into files that Codex can inspect directly."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize(path: Path, agent_name: str, requested_side: str | None = None) -> dict:
    replay = json.loads(path.read_text(encoding="utf-8"))
    players = replay["metadata"]["players"]
    agent_ids = [
        player_id
        for player_id, agent_path in players.items()
        if Path(agent_path).parent.name == agent_name
    ]
    if requested_side is not None:
        agent_id = requested_side
        if agent_id not in agent_ids:
            raise ValueError(f"{path}: {requested_side} is not a {agent_name} agent")
    elif len(agent_ids) == 1:
        agent_id = agent_ids[0]
    else:
        raise ValueError(f"{path}: expected one {agent_name} agent, found {agent_ids}")
    side = int(agent_id.removeprefix("player_"))
    other = 1 - side
    observations = replay["observations"]
    match_scores = []
    for previous, current in zip(observations, observations[1:]):
        if current["match_steps"] < previous["match_steps"]:
            points = previous["team_points"]
            match_scores.append((int(points[side]), int(points[other])))
    wins = observations[-1]["team_wins"]
    return {
        "seed": int(replay["metadata"]["seed"]),
        "agent_side": agent_id,
        "agent_match_wins": int(wins[side]),
        "opponent_match_wins": int(wins[other]),
        "agent_points": sum(score[0] for score in match_scores),
        "opponent_points": sum(score[1] for score in match_scores),
        "match_scores": " ".join(f"{left}:{right}" for left, right in match_scores),
        "replay": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, default=Path("replays"))
    parser.add_argument("--agent", default="v0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    paths = sorted(args.replay_dir.glob("*.json"))
    rows = []
    errors = []
    for path in paths:
        try:
            matching = [
                player_id
                for player_id, agent_path in json.loads(path.read_text(encoding="utf-8"))["metadata"]["players"].items()
                if Path(agent_path).parent.name == args.agent
            ]
            if len(matching) == 2:
                rows.extend(summarize(path, args.agent, side) for side in matching)
            else:
                rows.append(summarize(path, args.agent))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append({"replay": str(path), "error": str(error)})
    rows.sort(key=lambda row: (row["seed"], row["agent_side"], row["replay"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "latest_results.json").write_text(
        json.dumps({"agent": args.agent, "rows": rows, "errors": errors}, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "latest_results.csv").open("w", newline="", encoding="utf-8") as file:
        fields = list(rows[0]) if rows else ["replay", "error"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"# Latest replay results: `{args.agent}`", "", f"- Replays found: {len(paths)}", f"- Valid: {len(rows)}", f"- Errors: {len(errors)}", ""]
    if rows:
        lines += ["| Seed | Side | Wins | Opponent wins | Points | Opponent points | Match scores |", "|---:|---|---:|---:|---:|---:|---|"]
        for row in rows:
            lines.append(f"| {row['seed']} | {row['agent_side']} | {row['agent_match_wins']} | {row['opponent_match_wins']} | {row['agent_points']} | {row['opponent_points']} | {row['match_scores']} |")
    if errors:
        lines += ["", "## Errors", ""]
        lines += [f"- `{item['replay']}`: {item['error']}" for item in errors]
    (args.output_dir / "latest_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_dir / 'latest_results.md'}")


if __name__ == "__main__":
    main()
