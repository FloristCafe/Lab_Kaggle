from __future__ import annotations

import argparse
import gc
from pathlib import Path

import polars as pl

from otto_recommender.schema import AID, SESSION, TS, TYPE

MS_PER_DAY = 86_400_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict global time-axis split for ranker labels.")
    parser.add_argument("--input", default="data/test.parquet")
    parser.add_argument("--output-dir", default="data/processed/time_split")
    parser.add_argument("--valid-days", type=float, default=1.0)
    parser.add_argument("--cutoff-ts", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scan = pl.scan_parquet(args.input).select(
        pl.col(SESSION).cast(pl.Int32),
        pl.col(AID).cast(pl.Int32),
        pl.col(TS).cast(pl.Int64),
        pl.col(TYPE).cast(pl.Int8),
    )
    summary = scan.select(
        pl.len().alias("rows"),
        pl.col(SESSION).n_unique().alias("sessions"),
        pl.min(TS).alias("min_ts"),
        pl.max(TS).alias("max_ts"),
    ).collect()
    max_ts = int(summary.select("max_ts").item())
    cutoff_ts = args.cutoff_ts if args.cutoff_ts is not None else int(max_ts - args.valid_days * MS_PER_DAY)

    feature_events = scan.filter(pl.col(TS) < cutoff_ts)
    label_events = scan.filter(pl.col(TS) >= cutoff_ts)

    train_path = output_dir / "train_events.parquet"
    label_events_path = output_dir / "label_events.parquet"
    labels_path = output_dir / "valid_labels.parquet"
    targets_path = output_dir / "label_targets.parquet"
    report_path = output_dir / "split_report.csv"

    feature_events.sink_parquet(train_path)
    label_events.sink_parquet(label_events_path)

    (
        label_events.group_by([SESSION, TYPE])
        .agg(pl.col(AID).unique(maintain_order=True).alias("ground_truth"))
        .sort([SESSION, TYPE])
        .sink_parquet(labels_path)
    )

    (
        label_events.group_by([SESSION, AID])
        .agg(
            (pl.col(TYPE) == 0).any().cast(pl.Int8).alias("target_click"),
            (pl.col(TYPE) == 1).any().cast(pl.Int8).alias("target_cart"),
            (pl.col(TYPE) == 2).any().cast(pl.Int8).alias("target_order"),
        )
        .sink_parquet(targets_path)
    )

    train_summary = pl.scan_parquet(train_path).select(
        pl.len().alias("feature_rows"),
        pl.col(SESSION).n_unique().alias("feature_sessions"),
        pl.min(TS).alias("feature_min_ts"),
        pl.max(TS).alias("feature_max_ts"),
    ).collect()
    label_summary = pl.scan_parquet(label_events_path).select(
        pl.len().alias("label_event_rows"),
        pl.col(SESSION).n_unique().alias("label_sessions"),
        pl.min(TS).alias("label_min_ts"),
        pl.max(TS).alias("label_max_ts"),
    ).collect()
    target_summary = pl.scan_parquet(targets_path).select(
        pl.len().alias("label_target_rows"),
        pl.sum("target_click").alias("target_click_rows"),
        pl.sum("target_cart").alias("target_cart_rows"),
        pl.sum("target_order").alias("target_order_rows"),
    ).collect()
    boundary = (
        pl.scan_parquet(train_path)
        .group_by(SESSION)
        .agg(pl.max(TS).alias("feature_max_ts"))
        .join(
            pl.scan_parquet(label_events_path).group_by(SESSION).agg(pl.min(TS).alias("label_min_ts")),
            on=SESSION,
            how="inner",
        )
        .select(
            pl.len().alias("sessions_with_both_sides"),
            (pl.col("feature_max_ts") < pl.col("label_min_ts")).sum().alias("strict_ok_sessions"),
            (pl.col("feature_max_ts") >= pl.col("label_min_ts")).sum().alias("boundary_violations"),
        )
        .collect()
    )
    report = pl.concat(
        [
            summary.with_columns(pl.lit(cutoff_ts).alias("cutoff_ts")),
            train_summary,
            label_summary,
            target_summary,
            boundary,
        ],
        how="horizontal",
    )
    report.write_csv(report_path)

    print(f"cutoff_ts={cutoff_ts}")
    print(f"feature_events -> {train_path}")
    print(f"label_events -> {label_events_path}")
    print(f"valid_labels -> {labels_path}")
    print(f"label_targets -> {targets_path}")
    print(f"report -> {report_path}")
    print(report)

    del report, train_summary, label_summary, target_summary, boundary
    gc.collect()


if __name__ == "__main__":
    main()
