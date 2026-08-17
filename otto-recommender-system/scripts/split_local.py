from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from otto_recommender.polars_graph import split_train_valid_tail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split local OTTO parquet into train events and validation labels.")
    parser.add_argument("--input", default="data/test.parquet")
    parser.add_argument("--train-output", default="data/processed/local_train.parquet")
    parser.add_argument("--labels-output", default="data/processed/local_valid_labels.parquet")
    parser.add_argument("--valid-events-per-session", type=int, default=1)
    parser.add_argument("--min-session-length", type=int, default=2)
    return parser.parse_args()


def run_split(args: argparse.Namespace) -> None:
    events = pl.read_parquet(args.input)
    train, labels = split_train_valid_tail(
        events,
        valid_events_per_session=args.valid_events_per_session,
        min_session_length=args.min_session_length,
    )
    train_path = Path(args.train_output)
    labels_path = Path(args.labels_output)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    train.write_parquet(train_path)
    labels.write_parquet(labels_path)
    print(f"train_events={train.height:,} -> {train_path}")
    print(f"valid_label_rows={labels.height:,} -> {labels_path}")


def main() -> None:
    run_split(parse_args())


if __name__ == "__main__":
    main()
