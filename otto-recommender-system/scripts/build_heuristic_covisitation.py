from __future__ import annotations

import argparse
from pathlib import Path

from otto_recommender.heuristic_covisitation import build_all_rule_graphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stage-two heuristic co-visitation graphs.")
    parser.add_argument("--events", default="data/processed/local_train.parquet")
    parser.add_argument("--output-dir", default="artifacts/candidates/heuristic_covisitation")
    parser.add_argument("--stats-output", default="artifacts/reports/heuristic_covisitation_stats.csv")
    parser.add_argument("--n-buckets", type=int, default=16)
    parser.add_argument("--degree-alpha", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_all_rule_graphs(
        input_path=args.events,
        output_dir=args.output_dir,
        n_buckets=args.n_buckets,
        degree_alpha=args.degree_alpha,
    )
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats.write_csv(stats_path)
    print(f"stats_rows={stats.height:,} -> {stats_path}")
    print(f"graphs -> {args.output_dir}")


if __name__ == "__main__":
    main()
