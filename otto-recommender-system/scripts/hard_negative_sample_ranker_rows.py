from __future__ import annotations

import argparse
from pathlib import Path

from otto_recommender.hard_negative_sampling import HardNegativeConfig, write_hard_negative_parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard-negative sample sparse cart/order ranker rows.")
    parser.add_argument("--input-dir", default="artifacts/ranker/time_split/labeled_features_parts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--target-col", required=True, choices=("target_cart", "target_order"))
    parser.add_argument("--graph-col", required=True)
    parser.add_argument("--item-features", default=None)
    parser.add_argument("--hard-click-neg-per-pos", type=float, default=8.0)
    parser.add_argument("--hard-graph-neg-per-pos", type=float, default=8.0)
    parser.add_argument("--random-neg-per-pos", type=float, default=4.0)
    parser.add_argument("--min-local-clicks", type=int, default=2)
    parser.add_argument("--graph-quantile", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pattern", default="*.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(Path(args.input_dir).glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No labeled parts matched {Path(args.input_dir) / args.pattern}")
    config = HardNegativeConfig(
        target_col=args.target_col,
        graph_col=args.graph_col,
        hard_click_neg_per_pos=args.hard_click_neg_per_pos,
        hard_graph_neg_per_pos=args.hard_graph_neg_per_pos,
        random_neg_per_pos=args.random_neg_per_pos,
        min_local_clicks=args.min_local_clicks,
        graph_quantile=args.graph_quantile,
        seed=args.seed,
    )
    stats = write_hard_negative_parts(
        input_paths=paths,
        output_dir=args.output_dir,
        stats_output=args.stats_output,
        config=config,
        item_features_path=args.item_features,
    )
    print(stats.tail(1))
    print(f"sampled_parts -> {args.output_dir}")
    print(f"stats -> {args.stats_output}")


if __name__ == "__main__":
    main()
