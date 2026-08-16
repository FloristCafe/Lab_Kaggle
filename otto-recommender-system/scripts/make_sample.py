from __future__ import annotations

import argparse
from pathlib import Path

from otto_recommender.sampling import sample_jsonl_sessions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small OTTO JSONL sample.")
    parser.add_argument("--input", required=True, help="Path to train.jsonl or test.jsonl.")
    parser.add_argument("--output", default="data/sample/train_sample.jsonl")
    parser.add_argument("--n-sessions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = sample_jsonl_sessions(
        input_path=args.input,
        output_path=Path(args.output),
        n_sessions=args.n_sessions,
        seed=args.seed,
    )
    print(f"Wrote {written} sessions to {args.output}")


if __name__ == "__main__":
    main()
