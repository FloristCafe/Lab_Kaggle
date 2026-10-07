"""Compare paired v1 suites using saved results, replays, and trace frames."""

import argparse
import json
from pathlib import Path

from analyze_v1_traces import summarize_trace


ROOT = Path(__file__).resolve().parents[1]


def load_suite(name):
    directory = ROOT / "outputs" / name
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text(
        encoding="utf-8"
    ).splitlines() if line]
    runs = {}
    for row in rows:
        key = (row["seed"], row["opponent"], row["side"])
        if key in runs or row["return_code"] != 0 or len(row["traces"]) != 1:
            raise ValueError(f"{name}: incomplete or duplicate run {key}")
        trace = summarize_trace(Path(row["traces"][0]), ROOT / "replays" / name)
        scores = [tuple(map(int, pair.split(":"))) for pair in
                  row["results"][0]["match_scores"].split()]
        if len(scores) != 5 or any(ours == theirs for ours, theirs in scores):
            raise ValueError(f"{name}: expected five non-tied match scores for {key}")
        runs[key] = {
            "game_win": row["results"][0]["agent_match_wins"] > 2,
            "match_wins": [ours > theirs for ours, theirs in scores],
            "point_margin": row["results"][0]["agent_points"]
            - row["results"][0]["opponent_points"],
            "first_confirmed": trace["first_confirmed_step"],
            "early_confirmed": trace["first_two_confirmed"] == "yes",
            "first_match_footprint": trace["first_match_visited_tiles"],
        }
    return manifest, runs


def summary(runs):
    values = list(runs.values())
    return {
        "games": sum(value["game_win"] for value in values),
        "matches": [sum(value["match_wins"][match] for value in values)
                    for match in range(5)],
        "point_margin": sum(value["point_margin"] for value in values),
        "first_confirmed_mean": sum(value["first_confirmed"] for value in values)
        / len(values),
        "early_confirmed": sum(value["early_confirmed"] for value in values),
        "footprint_mean": sum(value["first_match_footprint"] for value in values)
        / len(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_manifest, baseline = load_suite(args.baseline)
    candidate_manifest, candidate = load_suite(args.candidate)
    if baseline.keys() != candidate.keys():
        raise ValueError("suites do not have the same paired fixtures")
    if base_manifest["environment_revision"] != candidate_manifest["environment_revision"]:
        raise ValueError("suites use different environment revisions")
    before, after = summary(baseline), summary(candidate)
    count = len(baseline)
    lines = [
        "# V1 marginal information ablation", "",
        f"Baseline: `{args.baseline}` (`{base_manifest['agent_hashes']['v1']}`).  ",
        f"Candidate: `{args.candidate}` (`{candidate_manifest['agent_hashes']['v1']}`).  ",
        f"Environment: `{base_manifest['environment_revision']}`.  ",
        f"Paired fixtures: {count}; both sides of each seed are correlated. Results are descriptive.",
        "", "| Metric | Baseline | Candidate |", "|---|---:|---:|",
        f"| Five-match games won | {before['games']}/{count} | {after['games']}/{count} |",
        f"| Match wins M1-M5 | {' / '.join(map(str, before['matches']))} | {' / '.join(map(str, after['matches']))} |",
        f"| Match wins M1-M2 | {sum(before['matches'][:2])}/{2 * count} | {sum(after['matches'][:2])}/{2 * count} |",
        f"| Match wins M4-M5 | {sum(before['matches'][3:])}/{2 * count} | {sum(after['matches'][3:])}/{2 * count} |",
        f"| Total point margin | {before['point_margin']:+d} | {after['point_margin']:+d} |",
        f"| First confirmed step, mean | {before['first_confirmed_mean']:.1f} | {after['first_confirmed_mean']:.1f} |",
        f"| Confirmed in M1-M2 | {before['early_confirmed']}/{count} | {after['early_confirmed']}/{count} |",
        f"| Match 1 unique visited tiles, mean | {before['footprint_mean']:.1f} | {after['footprint_mean']:.1f} |",
        "", "| Seed | Side | Game W before/after | First confirmed step before/after | Match 1 visited tiles before/after |",
        "|---:|---:|---|---|---|",
    ]
    for seed, opponent, side in sorted(baseline):
        old = baseline[(seed, opponent, side)]
        new = candidate[(seed, opponent, side)]
        lines.append(
            f"| {seed} | {side} | {int(old['game_win'])}/{int(new['game_win'])} | "
            f"{old['first_confirmed']}/{new['first_confirmed']} | "
            f"{old['first_match_footprint']}/{new['first_match_footprint']} |"
        )
    lines += [
        "", "First confirmed step is the episode step at which the observation-only "
        "probability map first contains 1.0. Match 1 visited tiles counts unique physical "
        "coordinates occupied at least once by the team, not simultaneous occupancy.",
        "Early confirmations are absent by construction unless all occupied tiles score: "
        "the current solver defers exact equations until after the final relic spawn window.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
