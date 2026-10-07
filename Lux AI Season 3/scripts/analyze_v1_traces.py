"""Summarize v1 teacher decisions and paired replay outcomes after games."""

import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_replays import diagnose
from pack_v1_traces import read_frames


def summarize_trace(path, replay_dir):
    frames = read_frames(path)
    decisions = {key: np.stack([frame[key] for frame in frames]) for key in frames[0]}
    active = decisions["active"]
    action = decisions["action"][:, :, 0]
    target = decisions["target"]
    label_valid = decisions.get("label_valid")
    if label_valid is None:
        label_valid = np.zeros_like(active)
        selected_step, selected_unit = np.where(active & (target[:, :, 0] >= 0))
        selected_x, selected_y = target[selected_step, selected_unit].T
        label_valid[selected_step, selected_unit] = decisions["reachable"][
            selected_step, selected_unit, selected_x, selected_y
        ]
    features = decisions["global_features"]
    unit = decisions["unit_features"]
    valid = active & (target[:, :, 0] >= 0)
    timestep, unit_id = np.where(valid)
    tx, ty = target[timestep, unit_id].T
    probability = features[timestep, 12, tx, ty]
    current_x = np.rint(unit[timestep, unit_id, 0] * features.shape[2]).astype(int)
    current_y = np.rint(unit[timestep, unit_id, 1] * features.shape[3]).astype(int)
    target_distance = np.abs(tx - current_x) + np.abs(ty - current_y)
    energy = unit[:, :, 2] * 400
    moving = active & (action > 0) & (action < 5)
    idle = active & (action == 0)
    base_name = path.name.split("_player_")[0]
    replay = replay_dir / f"{base_name}.json"
    replay_data = json.loads(replay.read_text(encoding="utf-8"))
    move_cost = int(replay_data["params"]["unit_move_cost"])
    low_energy_idle = idle & (energy < move_cost - 0.1)
    at_target = idle & (target[:, :, 0] >= 0) & (
        target[:, :, 0] == np.rint(unit[:, :, 0] * features.shape[2])
    ) & (target[:, :, 1] == np.rint(unit[:, :, 1] * features.shape[3]))
    continued = active[1:] & active[:-1]
    changed = continued & np.any(target[1:] != target[:-1], axis=2)
    duplicate_steps = 0
    extra_target_slots = 0
    stacked_idle = 0
    unconfirmed_idle = 0
    unique_occupied_total = 0
    occupied_steps = 0
    for step in range(len(frames)):
        chosen = [tuple(tile) for tile in target[step, active[step]]]
        duplicate_steps += len(chosen) != len(set(chosen))
        extra_target_slots += len(chosen) - len(set(chosen))
        unit_ids = np.flatnonzero(active[step])
        occupied = [
            (int(round(unit[step, index, 0] * features.shape[2])),
             int(round(unit[step, index, 1] * features.shape[3])))
            for index in unit_ids
        ]
        if occupied:
            unique_occupied_total += len(set(occupied))
            occupied_steps += 1
        counts = {tile: occupied.count(tile) for tile in set(occupied)}
        for index, tile in zip(unit_ids, occupied):
            if idle[step, index]:
                stacked_idle += counts[tile] > 1
                chosen_tile = target[step, index]
                unconfirmed_idle += features[step, 12, chosen_tile[0], chosen_tile[1]] < 0.999
    same_position = np.all(unit[1:, :, :2] == unit[:-1, :, :2], axis=2)
    failed_move = moving[:-1] & continued & same_position
    identified = features[:, 12] >= 0.999
    match_starts = np.flatnonzero(decisions["match_step"] == 0)
    first_match_end = int(match_starts[1]) if len(match_starts) > 1 else len(frames)
    second_match_end = int(match_starts[2]) if len(match_starts) > 2 else len(frames)
    first_confirmed_frames = np.flatnonzero(identified.any(axis=(1, 2)))
    first_confirmed_index = int(first_confirmed_frames[0]) if len(first_confirmed_frames) else None
    first_match_tiles = set()
    for frame in range(first_match_end):
        for index in np.flatnonzero(active[frame]):
            first_match_tiles.add((
                int(round(unit[frame, index, 0] * features.shape[2])),
                int(round(unit[frame, index, 1] * features.shape[3])),
            ))
    side = f"player_{int(decisions['team_id'][0])}"
    matches = diagnose(replay, side)
    return {
        "name": base_name,
        "frames": len(frames),
        "active_actions": int(active.sum()),
        "move_percent": round(100 * moving.sum() / max(1, active.sum()), 1),
        "idle_percent": round(100 * idle.sum() / max(1, active.sum()), 1),
        "low_energy_idle_percent": round(100 * low_energy_idle.sum() / max(1, idle.sum()), 1),
        "at_target_idle_percent": round(100 * at_target.sum() / max(1, idle.sum()), 1),
        "target_change_percent": round(100 * changed.sum() / max(1, continued.sum()), 1),
        "attempted_move_stall_percent": round(100 * failed_move.sum() / max(1, moving[:-1].sum()), 1),
        "target_probability_mean": round(float(probability.mean()), 3),
        "target_probability_confirmed_percent": round(100 * (probability >= 0.999).mean(), 1),
        "target_distance_mean": round(float(target_distance.mean()), 1),
        "confirmed_tiles_final": int(identified[-1].sum()),
        "first_confirmed_step": (
            int(decisions["step"][first_confirmed_index])
            if first_confirmed_index is not None else "none"
        ),
        "first_confirmed_match": (
            int(np.searchsorted(match_starts, first_confirmed_index, side="right"))
            if first_confirmed_index is not None else "none"
        ),
        "first_two_confirmed": (
            "yes" if first_confirmed_index is not None
            and first_confirmed_index < second_match_end else "no"
        ),
        "first_match_visited_tiles": len(first_match_tiles),
        "unique_occupied_mean": round(unique_occupied_total / max(1, occupied_steps), 1),
        "duplicate_target_steps": duplicate_steps,
        "invalid_label_percent": round(100 * (active & ~label_valid).sum() / max(1, active.sum()), 2),
        "extra_target_slots": extra_target_slots,
        "stacked_idle_percent": round(100 * stacked_idle / max(1, idle.sum()), 1),
        "unconfirmed_idle_percent": round(100 * unconfirmed_idle / max(1, idle.sum()), 1),
        "match_scores": " ".join(f"{row['points']}:{row['opponent_points']}" for row in matches),
        "unit_losses": sum(row["unit_losses"] for row in matches),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--replay-dir", type=Path, default=Path("replays/v1_dev"))
    parser.add_argument("--output", type=Path, default=Path("outputs/v1_dev/trace_diagnostics.md"))
    args = parser.parse_args()
    rows = [summarize_trace(path, args.replay_dir) for path in args.traces]
    lines = [
        "# V1 trace diagnostics", "",
        "Percentages use active unit decisions. Low-energy and at-target idle percentages use idle decisions.",
        "Target changes use the same active unit in consecutive observations. These are diagnostics, not causal attributions.",
        "", "| Run | Frames | Decisions | Move % | Idle % | Low-energy idle % | At-target idle % | Target changes % | Stalled moves % | Mean target P | Confirmed target % | Mean target distance | Final confirmed tiles | First confirmed step | First confirmed match | Confirmed in M1-2 | Match 1 visited tiles | Duplicate-target steps | Unit losses | Match scores |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {frames} | {active_actions} | {move_percent} | {idle_percent} | "
            "{low_energy_idle_percent} | {at_target_idle_percent} | {target_change_percent} | "
            "{attempted_move_stall_percent} | {target_probability_mean} | "
            "{target_probability_confirmed_percent} | {target_distance_mean} | "
            "{confirmed_tiles_final} | {first_confirmed_step} | {first_confirmed_match} | "
            "{first_two_confirmed} | "
            "{first_match_visited_tiles} | {duplicate_target_steps} | {unit_losses} | "
            "{match_scores} |".format(**row)
        )
    lines += [
        "", "## Occupancy", "",
        "| Run | Mean unique occupied tiles | Stacked idle % | Idle on unconfirmed target % | Extra duplicate target slots | Invalid SL labels % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {unique_occupied_mean} | {stacked_idle_percent} | "
            "{unconfirmed_idle_percent} | {extra_target_slots} | {invalid_label_percent} |".format(**row)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
