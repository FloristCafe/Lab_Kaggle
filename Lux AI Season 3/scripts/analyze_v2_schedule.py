"""Rebuild v2 evaluation and physical movement diagnostics from saved artifacts."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def physical_diagnostics(path, side):
    replay = json.loads(path.read_text(encoding="utf-8"))
    observations, actions = replay["observations"], replay["actions"]
    player = f"player_{side}"
    new_overlap = spawn_overlap = stalled = active_moves = losses = 0
    examples = []
    stall_examples = []
    directions = {1: (0, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 0)}
    for index, (before, after) in enumerate(zip(observations, observations[1:])):
        if before["steps"] > 0 and before["match_steps"] == 0:
            continue
        old_mask, new_mask = before["units_mask"][side], after["units_mask"][side]
        old_pos = before["units"]["position"][side]
        new_pos = after["units"]["position"][side]
        action = actions[index][player]
        losses += sum(old and not new for old, new in zip(old_mask, new_mask))
        for unit_id, (old, new) in enumerate(zip(old_mask, new_mask)):
            if old and action[unit_id][0] in (1, 2, 3, 4):
                active_moves += 1
                if new and old_pos[unit_id] == new_pos[unit_id]:
                    stalled += 1
                    if len(stall_examples) < 5:
                        dx, dy = directions[action[unit_id][0]]
                        x, y = old_pos[unit_id]
                        target = (x + dx, y + dy)
                        stall_examples.append({
                            "step": before["steps"], "unit": unit_id,
                            "from": old_pos[unit_id], "target": target,
                            "energy": before["units"]["energy"][side][unit_id],
                            "target_tile_type": before["map_features"]["tile_type"][target[0]][target[1]],
                        })
        for left in range(len(old_mask)):
            if not new_mask[left]:
                continue
            for right in range(left + 1, len(old_mask)):
                if not new_mask[right] or new_pos[left] != new_pos[right]:
                    continue
                if old_mask[left] and old_mask[right]:
                    if old_pos[left] != old_pos[right]:
                        new_overlap += 1
                        if len(examples) < 3:
                            examples.append({"step": before["steps"],
                                             "units": [left, right],
                                             "positions_before": [old_pos[left], old_pos[right]],
                                             "position_after": new_pos[left],
                                             "actions": [action[left][0], action[right][0]]})
                else:
                    spawn_overlap += 1
    return {"new_friendly_overlap_pairs": new_overlap,
            "spawn_overlap_pairs": spawn_overlap,
            "attempted_moves": active_moves,
            "attempted_move_stalls": stalled,
            "unit_losses": losses, "overlap_examples": examples,
            "stall_examples": stall_examples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite")
    args = parser.parse_args()
    directory = ROOT / "outputs" / args.suite
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    runs = [json.loads(line) for line in (directory / "results.jsonl").read_text(
        encoding="utf-8").splitlines() if line]
    expected = {(seed, side) for seed in range(
        manifest["seed_start"], manifest["seed_start"] + manifest["seed_count"])
        for side in manifest["sides"]}
    if {(run["seed"], run["side"]) for run in runs} != expected:
        raise ValueError("schedule is incomplete or has duplicate fixtures")
    if any(run["return_code"] != 0 or "result" not in run for run in runs):
        raise ValueError("schedule has failed games")

    wins = 0
    match_wins = [0] * 5
    margin = 0
    totals = {key: 0 for key in (
        "invalid_action_logs", "contradictions", "planned_conflict_frames",
        "yield_decisions", "new_friendly_overlap_pairs", "spawn_overlap_pairs",
        "attempted_moves", "attempted_move_stalls", "unit_losses")}
    rows = []
    for run in sorted(runs, key=lambda item: (item["seed"], item["side"])):
        result = run["result"]
        scores = [tuple(map(int, pair.split(":")))
                  for pair in result["match_scores"].split()]
        if len(scores) != 5:
            raise ValueError("expected five match scores")
        game_win = result["agent_match_wins"] > result["opponent_match_wins"]
        wins += game_win
        for index, (ours, theirs) in enumerate(scores):
            match_wins[index] += ours > theirs
        margin += result["agent_points"] - result["opponent_points"]
        physical = physical_diagnostics(Path(run["replay"]), run["side"])
        metrics = run["metrics"] | physical
        for key in totals:
            totals[key] += metrics[key]
        rows.append((run, game_win, scores, metrics))

    lines = [
        "# V2 epoch isolation and reservation routing: development suite", "",
        "## Hypothesis", "",
        "A dynamic equation pool that never promotes early observations to permanent tile values, "
        "combined with rank-triggered yielding and reserved one-step routes, should avoid "
        "belief contradictions and planned friendly collisions while improving late-match scoring.",
        "", "## Code and schedule", "",
        f"- Agent SHA-256: `{manifest['agent_hashes']['v2']}`; environment: `{manifest['environment_revision']}`.",
        f"- Git revision: `{manifest['git_revision']}`; dirty worktree: `{manifest['git_dirty']}`; schedule SHA-256: `{manifest['schedule_sha256']}`.",
        f"- Seeds {manifest['seed_start']}–{manifest['seed_start'] + manifest['seed_count'] - 1}, both sides, v0 opponent; {len(runs)} five-match games.",
        "- Runner: project Python 3.11/JAX CPU; orchestration and MLflow: `/home/issue/ml-workspace/.venv/bin/python`.",
        "- MLflow experiment: `lux-s3-v2-epoch-router`; each row below has its run ID. All replay and trace paths are in `results.jsonl`.",
        "", "## Outcomes", "",
        "| Metric | Value |", "|---|---:|",
        f"| Five-match games won | {wins}/{len(runs)} |",
        f"| Match wins M1–M5 | {' / '.join(map(str, match_wins))} |",
        f"| Total point margin | {margin:+d} |",
        f"| Invalid-action log entries | {totals['invalid_action_logs']} |",
        f"| Static-solver contradictions | {totals['contradictions']} |",
        f"| Planned first-step friendly conflict frames | {totals['planned_conflict_frames']} |",
        f"| Actual new friendly overlap pairs, existing units | {totals['new_friendly_overlap_pairs']} |",
        f"| Spawn-related friendly overlap pairs | {totals['spawn_overlap_pairs']} |",
        f"| Attempted moves that stayed in place | {totals['attempted_move_stalls']}/{totals['attempted_moves']} |",
        f"| Unit losses (all causes) | {totals['unit_losses']} |",
        f"| Explicit yield decisions | {totals['yield_decisions']} |",
        "", "| Seed | Side | Game W | Match wins | Points | Invalid | Planned conflicts | New overlaps | Move stalls | Agent decision s | Max step s | MLflow run ID |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run, game_win, scores, metrics in rows:
        lines.append(
            f"| {run['seed']} | {run['side']} | {int(game_win)} | "
            f"{sum(ours > theirs for ours, theirs in scores)} | "
            f"{run['result']['agent_points']}:{run['result']['opponent_points']} | "
            f"{metrics['invalid_action_logs']} | {metrics['planned_conflict_frames']} | "
            f"{metrics['new_friendly_overlap_pairs']} | {metrics['attempted_move_stalls']} | "
            f"{metrics['agent_decision_seconds']:.1f} | "
            f"{metrics['max_decision_seconds']:.3f} | `{run['mlflow_run_id']}` |"
        )
    examples = [(run["seed"], run["side"], example)
                for run, _, _, metrics in rows
                for example in metrics["overlap_examples"]]
    lines += ["", "## Limits and next question", "",
              "Dynamic equations are retained for one observation only and never solved or inherited. "
              "The static pool starts after the final possible relic spawn window; this run does "
              "not validate early exact inference. The 0.5 unresolved map value remains a placeholder.",
              "Friendly overlap is legal in the official engine. Planned conflicts measure identical "
              "first-step destinations from distinct origins; actual new overlaps also reflect terrain "
              "changes, movement failures and simultaneous events. Spawn overlaps are reported separately. "
              "Move stalls are not automatically collisions, and unit losses are not assigned a cause.",
              "Decision seconds are wall-clock time inside `Agent.act`, excluding the JAX runner. "
              "Paired sides of one seed are correlated, and five seeds are a development suite. "
              "The final 50-seed holdout was not touched. Early match wins remain weak; the next "
              "isolated question is whether time-aware early evidence can improve M1–M2 without "
              "contaminating the static pool.",
              ""]
    if examples:
        lines += ["## New-overlap examples", ""]
        for seed, side, example in examples[:5]:
            lines.append(f"- Seed {seed}, side {side}, step {example['step']}: "
                         f"units {example['units']} from {example['positions_before']} "
                         f"to {example['position_after']} with actions {example['actions']}.")
    stalls = [(run["seed"], run["side"], example)
              for run, _, _, metrics in rows
              for example in metrics["stall_examples"]]
    if stalls:
        lines += ["", "## Move-stall examples", ""]
        for seed, side, example in stalls:
            lines.append(f"- Seed {seed}, side {side}, step {example['step']}, "
                         f"unit {example['unit']}: {example['from']} toward "
                         f"{example['target']}; energy {example['energy']}, "
                         f"target tile type {example['target_tile_type']} in full replay state.")
    path = directory / "research_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
