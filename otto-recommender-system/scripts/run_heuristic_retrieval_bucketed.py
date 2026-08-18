from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from otto_recommender.heuristic_covisitation import (
    evaluate_heuristic_recommendations,
    popular_fallback_items,
    recommend_from_heuristic_graphs,
)
from otto_recommender.schema import SESSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Top-N heuristic candidates by session bucket.")
    parser.add_argument("--events", default="data/processed/local_train.parquet")
    parser.add_argument("--graphs-dir", default="artifacts/candidates/heuristic_covisitation")
    parser.add_argument("--labels", default="data/processed/local_valid_labels.parquet")
    parser.add_argument("--output-dir", default="artifacts/candidates/heuristic_candidates_top100_parts")
    parser.add_argument("--stats-output", default="artifacts/reports/heuristic_candidates_top100_stats.csv")
    parser.add_argument("--metrics-output", default="artifacts/reports/heuristic_candidates_top100_metrics.csv")
    parser.add_argument("--n-buckets", type=int, default=16)
    parser.add_argument("--candidate-topk", type=int, default=100)
    parser.add_argument("--min-candidates", type=int, default=50)
    parser.add_argument("--eval-topk", type=int, default=20)
    parser.add_argument("--max-seed-items", type=int, default=20)
    parser.add_argument("--disable-popular-fallback", action="store_true")
    return parser.parse_args()


def graph_paths(graphs_dir: str | Path) -> list[Path]:
    graphs_dir = Path(graphs_dir)
    return [
        graphs_dir / "click_to_click_top20.parquet",
        graphs_dir / "cart_order_to_cart_order_top20.parquet",
        graphs_dir / "click_to_cart_order_top20.parquet",
    ]


def candidate_count_summary(part_paths: list[Path]) -> pl.DataFrame:
    counts = (
        pl.scan_parquet([str(path) for path in part_paths])
        .group_by(SESSION)
        .agg(pl.len().alias("n_candidates"))
        .collect()
    )
    return counts.select(
        pl.len().alias("sessions"),
        pl.min("n_candidates").alias("min_candidates"),
        pl.mean("n_candidates").alias("mean_candidates"),
        pl.median("n_candidates").alias("median_candidates"),
        pl.max("n_candidates").alias("max_candidates"),
        (pl.col("n_candidates") >= 50).sum().alias("sessions_ge_50"),
        (pl.col("n_candidates") == 100).sum().alias("sessions_eq_100"),
    )


def main() -> None:
    args = parse_args()
    paths = graph_paths(args.graphs_dir)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing graph files: {missing}")

    edge_frames = [pl.read_parquet(path) for path in paths]
    all_events = pl.scan_parquet(args.events)
    popular_fallback = None
    if not args.disable_popular_fallback:
        popular_fallback = popular_fallback_items(all_events.collect(), topk=args.candidate_topk)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    part_paths: list[Path] = []
    bucket_stats: list[dict[str, int | str]] = []

    for bucket in range(args.n_buckets):
        events = all_events.filter((pl.col(SESSION) % args.n_buckets) == bucket).collect()
        candidates = recommend_from_heuristic_graphs(
            events=events,
            edge_frames=edge_frames,
            topk=args.candidate_topk,
            min_candidates=args.min_candidates,
            max_seed_items=args.max_seed_items,
            popular_fallback=popular_fallback,
        )
        part_path = output_dir / f"candidates_part_{bucket:03d}.parquet"
        candidates.write_parquet(part_path)
        part_paths.append(part_path)
        bucket_stats.append(
            {
                "bucket": bucket,
                "events": events.height,
                "candidate_rows": candidates.height,
                "path": str(part_path),
            }
        )
        print(f"bucket={bucket} events={events.height:,} candidates={candidates.height:,} -> {part_path}")

    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats = pl.DataFrame(bucket_stats)
    summary = candidate_count_summary(part_paths)
    stats.write_csv(stats_path)
    summary.write_csv(stats_path.with_name(stats_path.stem + "_summary.csv"))
    print(summary)

    labels_path = Path(args.labels)
    if labels_path.exists():
        recommendations = (
            pl.scan_parquet([str(path) for path in part_paths])
            .filter(pl.col("rank") <= args.eval_topk)
            .collect()
        )
        metrics = evaluate_heuristic_recommendations(
            labels=pl.read_parquet(labels_path),
            recommendations=recommendations,
            k=args.eval_topk,
        )
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.write_csv(metrics_path)
        weighted = metrics.select("weighted_recall").item(0, 0)
        print(f"weighted_recall@{args.eval_topk}={weighted:.6f} -> {metrics_path}")


if __name__ == "__main__":
    main()
