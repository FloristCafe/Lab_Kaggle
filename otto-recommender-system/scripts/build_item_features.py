from __future__ import annotations

import argparse
from pathlib import Path

from otto_recommender.feature_engineering import build_item_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build item_features.parquet with low-memory lazy Polars.")
    parser.add_argument("--input", default="data/test.parquet")
    parser.add_argument("--output", default="artifacts/features/item_features.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_item_features(args.input, args.output)
    print(f"item_features -> {Path(args.output)}")


if __name__ == "__main__":
    main()

