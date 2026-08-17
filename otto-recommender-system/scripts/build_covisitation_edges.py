from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from otto_recommender.polars_graph import build_pruned_edge_parts, merge_pruned_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact Polars COO co-visitation edges.")
    parser.add_argument("--events", default="data/processed/local_train.parquet")
    parser.add_argument("--parts-dir", default="artifacts/candidates/edge_parts")
    parser.add_argument("--output", default="artifacts/candidates/top20_edges.parquet")
    parser.add_argument("--stats-output", default="artifacts/reports/edge_build_stats.csv")
    parser.add_argument("--n-buckets", type=int, default=16)
    parser.add_argument("--max-events-per-session", type=int, default=30)
    parser.add_argument("--topk-per-chunk", type=int, default=80)
    parser.add_argument("--final-topk-per-source", type=int, default=20)
    return parser.parse_args()


def run_build(args: argparse.Namespace) -> None:
    parts_dir = Path(args.parts_dir)
    stats = build_pruned_edge_parts(
        input_path=args.events,
        output_dir=parts_dir,
        n_buckets=args.n_buckets,
        max_events_per_session=args.max_events_per_session,
        topk_per_chunk=args.topk_per_chunk,
    )
    edge_paths = sorted(parts_dir.glob("edges_part_*.parquet"))
    final_edges = merge_pruned_edges(
        edge_paths=edge_paths,
        output_path=args.output,
        final_topk_per_source=args.final_topk_per_source,
    )
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats.write_csv(stats_path)
    print(f"edge_parts={len(edge_paths)} -> {parts_dir}")
    print(f"final_edges={final_edges.height:,} -> {args.output}")
    print(f"stats -> {stats_path}")


def main() -> None:
    run_build(parse_args())


if __name__ == "__main__":
    main()
