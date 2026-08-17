from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from otto_recommender.heuristic_covisitation import (
    evaluate_heuristic_recommendations,
    recommend_from_heuristic_graphs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate stage-two heuristic graphs into Top-20 candidates.")
    parser.add_argument("--events", default="data/processed/local_train.parquet")
    parser.add_argument("--graphs-dir", default="artifacts/candidates/heuristic_covisitation")
    parser.add_argument("--labels", default="data/processed/local_valid_labels.parquet")
    parser.add_argument("--recommendations-output", default="artifacts/candidates/heuristic_recommendations.parquet")
    parser.add_argument("--metrics-output", default="artifacts/reports/heuristic_graph_metrics.csv")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--max-seed-items", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_paths = [
        Path(args.graphs_dir) / "click_to_click_top20.parquet",
        Path(args.graphs_dir) / "cart_order_to_cart_order_top20.parquet",
        Path(args.graphs_dir) / "click_to_cart_order_top20.parquet",
    ]
    missing = [str(path) for path in graph_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing graph files: {missing}")

    events = pl.read_parquet(args.events)
    edge_frames = [pl.read_parquet(path) for path in graph_paths]
    recommendations = recommend_from_heuristic_graphs(
        events=events,
        edge_frames=edge_frames,
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
        metrics = evaluate_heuristic_recommendations(labels, recommendations, k=args.topk)
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.write_csv(metrics_path)
        weighted = metrics.select("weighted_recall").item(0, 0)
        print(f"weighted_recall@{args.topk}={weighted:.6f} -> {metrics_path}")


if __name__ == "__main__":
    main()
