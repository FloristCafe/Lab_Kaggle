from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from otto_recommender.interaction_features import write_interaction_features_candidate_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build interaction features with strict candidate chunking.")
    parser.add_argument("--events", default="data/processed/local_train.parquet")
    parser.add_argument("--candidates-dir", default="artifacts/candidates/heuristic_candidates_top100_parts")
    parser.add_argument("--graphs-dir", default="artifacts/candidates/heuristic_covisitation")
    parser.add_argument("--output-dir", default="artifacts/features/interaction_features_candidate_chunks")
    parser.add_argument("--stats-output", default="artifacts/reports/interaction_features_candidate_chunks_stats.csv")
    parser.add_argument("--n-buckets", type=int, default=16)
    parser.add_argument("--bucket-start", type=int, default=0)
    parser.add_argument("--bucket-end", type=int, default=None)
    parser.add_argument("--candidate-chunks", type=int, default=10)
    parser.add_argument("--max-seed-items", type=int, default=20)
    parser.add_argument("--delta-t-fill", type=int, default=9_999_999)
    return parser.parse_args()


def graph_paths(graphs_dir: str | Path) -> list[Path]:
    graphs_dir = Path(graphs_dir)
    return [
        graphs_dir / "click_to_click_top20.parquet",
        graphs_dir / "cart_order_to_cart_order_top20.parquet",
        graphs_dir / "click_to_cart_order_top20.parquet",
    ]


def main() -> None:
    args = parse_args()
    end = args.n_buckets if args.bucket_end is None else args.bucket_end
    graphs = graph_paths(args.graphs_dir)
    missing = [str(path) for path in graphs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing graph files: {missing}")

    all_stats: list[pl.DataFrame] = []
    for bucket in range(args.bucket_start, end):
        candidates_path = Path(args.candidates_dir) / f"candidates_part_{bucket:03d}.parquet"
        if not candidates_path.exists():
            raise FileNotFoundError(f"Missing candidates part: {candidates_path}")
        bucket_output_dir = Path(args.output_dir) / f"bucket_{bucket:03d}"
        stats = write_interaction_features_candidate_chunks(
            candidates_path=candidates_path,
            events_path=args.events,
            graph_paths=graphs,
            output_dir=bucket_output_dir,
            n_candidate_chunks=args.candidate_chunks,
            max_seed_items=args.max_seed_items,
            delta_t_fill=args.delta_t_fill,
        ).with_columns(pl.lit(bucket).alias("bucket"))
        all_stats.append(stats)
        rows = stats.select(pl.col("rows").sum()).item()
        print(f"bucket={bucket} chunks={stats.height} rows={rows:,} -> {bucket_output_dir}")

    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(all_stats, how="vertical").select(
        "bucket",
        "chunk",
        "row_start",
        "row_end",
        "rows",
        "path",
    ).write_csv(stats_path)
    print(f"stats -> {stats_path}")


if __name__ == "__main__":
    main()

