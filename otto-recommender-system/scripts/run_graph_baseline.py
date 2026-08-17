from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from otto_recommender.polars_graph import evaluate_recall_at_k, recommend_from_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run graph retrieval baseline from compact COO edges.")
    parser.add_argument("--events", default="data/processed/local_train.parquet")
    parser.add_argument("--edges", default="artifacts/candidates/top20_edges.parquet")
    parser.add_argument("--labels", default="data/processed/local_valid_labels.parquet")
    parser.add_argument("--recommendations-output", default="artifacts/candidates/local_recommendations.parquet")
    parser.add_argument("--metrics-output", default="artifacts/reports/local_graph_metrics.csv")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--max-seed-items", type=int, default=20)
    return parser.parse_args()


def run_baseline(args: argparse.Namespace) -> None:
    events = pl.read_parquet(args.events)
    edges = pl.read_parquet(args.edges)
    recommendations = recommend_from_edges(
        events=events,
        edges=edges,
        topk=args.topk,
        max_seed_items=args.max_seed_items,
    )
    rec_path = Path(args.recommendations_output)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    recommendations.write_parquet(rec_path)
    print(f"recommendations={recommendations.height:,} -> {rec_path}")

    labels_path = Path(args.labels)
    if labels_path.exists():
        labels = pl.read_parquet(labels_path)
        metrics = evaluate_recall_at_k(labels, recommendations, k=args.topk)
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.write_csv(metrics_path)
        weighted = metrics.select("weighted_recall").item(0, 0)
        print(f"weighted_recall@{args.topk}={weighted:.6f} -> {metrics_path}")


def main() -> None:
    run_baseline(parse_args())


if __name__ == "__main__":
    main()
