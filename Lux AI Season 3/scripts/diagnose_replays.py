"""Post-game diagnostics for replay files.

This intentionally reads serialized full states only for analysis after a game;
the agent still receives the official partial observation during the match.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def match_ranges(observations):
    ends = [
        index
        for index, (before, after) in enumerate(zip(observations, observations[1:]))
        if after["match_steps"] < before["match_steps"]
    ]
    starts = [0] + [end + 1 for end in ends]
    return list(zip(starts, ends))


def diagnose(path: Path, agent_id: str) -> list[dict]:
    replay = json.loads(path.read_text(encoding="utf-8"))
    side = int(agent_id.removeprefix("player_"))
    observations = replay["observations"]
    actions = replay["actions"]
    rows = []
    for match_number, (start, end) in enumerate(match_ranges(observations), 1):
        moves = stationary = active_steps = active_units = 0
        unit_births = unit_losses = 0
        for frame in range(start, min(end, len(actions) - 1)):
            before = observations[frame]
            after = observations[frame + 1]
            mask_before = before["units_mask"][side]
            mask_after = after["units_mask"][side]
            unit_births += sum(not old and new for old, new in zip(mask_before, mask_after))
            unit_losses += sum(old and not new for old, new in zip(mask_before, mask_after))
            positions_before = before["units"]["position"][side]
            positions_after = after["units"]["position"][side]
            frame_active = sum(mask_before)
            active_units += frame_active
            if frame_active:
                active_steps += 1
            for unit_id, exists in enumerate(mask_before):
                if not exists:
                    continue
                action = actions[frame][agent_id][unit_id]
                moved = positions_before[unit_id] != positions_after[unit_id]
                if moved:
                    moves += 1
                if action[0] == 0:
                    stationary += 1
        points = observations[end]["team_points"]
        rows.append(
            {
                "match": match_number,
                "frames": end - start + 1,
                "points": int(points[side]),
                "opponent_points": int(points[1 - side]),
                "moves": moves,
                "stationary_actions": stationary,
                "move_rate": round(moves / max(active_units, 1), 4),
                "unit_births": unit_births,
                "unit_losses": unit_losses,
                "active_steps": active_steps,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("replays", nargs="+", type=Path)
    parser.add_argument("--agent-id", default="player_0")
    parser.add_argument("--output", type=Path, default=Path("outputs/replay_diagnostics.md"))
    args = parser.parse_args()
    lines = ["# Replay diagnostics", "", "These are post-game serialized-state diagnostics.", ""]
    for path in args.replays:
        rows = diagnose(path, args.agent_id)
        lines += [f"## `{path}`", "", "| Match | Points | Opponent | Moves | Stationary | Move rate | Births | Losses |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            lines.append("| {match} | {points} | {opponent_points} | {moves} | {stationary_actions} | {move_rate} | {unit_births} | {unit_losses} |".format(**row))
        lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
