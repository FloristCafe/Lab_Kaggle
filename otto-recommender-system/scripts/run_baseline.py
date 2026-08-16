from __future__ import annotations

import argparse
from pathlib import Path

from otto_recommender.candidates import candidates_from_recent_and_covisitation
from otto_recommender.io import read_events
from otto_recommender.submission import candidates_to_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple OTTO candidate baseline.")
    parser.add_argument("--events", required=True, help="Path to events: jsonl, parquet, or csv.")
    parser.add_argument("--output", default="artifacts/submissions/baseline_submission.csv")
    parser.add_argument("--n-sessions", type=int, default=None, help="Only applies to JSONL input.")
    parser.add_argument("--topk", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    read_kwargs = {}
    if args.n_sessions is not None:
        read_kwargs["n_sessions"] = args.n_sessions

    events = read_events(args.events, **read_kwargs)
    candidates = candidates_from_recent_and_covisitation(events, final_topk=args.topk)
    submission = candidates_to_submission(candidates)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Wrote {len(submission)} rows to {output_path}")


if __name__ == "__main__":
    main()
