from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from otto_recommender.schema import AID, SESSION, TS, TYPE

TARGET_COLUMNS = {0: "target_click", 1: "target_cart", 2: "target_order"}


def split_validation_window(
    validation: pl.LazyFrame, prefix_fraction: float
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Split sessions at timestamp boundaries, dropping sessions without a time split."""
    ranked = validation.sort([SESSION, TS]).with_columns(
        pl.col(TS).rank(method="ordinal").over(SESSION).cast(pl.UInt32).alias("_time_rank"),
        pl.len().over(SESSION).cast(pl.UInt32).alias("_session_length"),
    )
    session_lengths = ranked.group_by(SESSION).agg(
        pl.max("_session_length"), pl.col(TS).n_unique().cast(pl.UInt32).alias("_timestamp_count")
    )
    target_lengths = session_lengths.filter(pl.col("_session_length") >= 2).with_columns(
        (pl.col("_session_length") * prefix_fraction)
        .floor()
        .clip(lower_bound=1)
        .clip(upper_bound=pl.col("_session_length") - 1)
        .cast(pl.UInt32)
        .alias("_target_prefix_length")
    )
    boundaries = (
        ranked.group_by([SESSION, TS])
        .agg(pl.max("_time_rank").alias("_cumulative_length"))
        .sort([SESSION, TS])
        .with_columns(
            pl.col("_cumulative_length").cast(pl.UInt32)
        )
        .join(target_lengths, on=SESSION, how="inner")
        .filter(
            (pl.col("_timestamp_count") >= 2)
            & (pl.col("_cumulative_length") < pl.col("_session_length"))
        )
        .with_columns(
            (pl.col("_cumulative_length") - pl.col("_target_prefix_length")).abs().alias("_distance")
        )
        .sort([SESSION, "_distance", "_cumulative_length"])
        .group_by(SESSION, maintain_order=True)
        .first()
        .select(SESSION, pl.col(TS).alias("_boundary_ts"))
    )
    split_events = validation.join(boundaries, on=SESSION, how="inner")
    prefix = split_events.filter(pl.col(TS) <= pl.col("_boundary_ts")).select(SESSION, AID, TS, TYPE)
    suffix = split_events.filter(pl.col(TS) > pl.col("_boundary_ts")).select(SESSION, AID, TS, TYPE)
    return prefix, suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split validation events into per-session prefix and suffix.")
    parser.add_argument("--input", default="data/test.parquet")
    parser.add_argument("--output-dir", default="data/processed/session_tail_split")
    parser.add_argument("--valid-days", type=float, default=1.0)
    parser.add_argument("--cutoff-ts", type=int, default=None)
    parser.add_argument("--prefix-fraction", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.5 <= args.prefix_fraction < 1.0:
        raise ValueError("--prefix-fraction must be in [0.5, 1.0)")
    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = pl.scan_parquet(args.input).select(
        pl.col(SESSION).cast(pl.Int32),
        pl.col(AID).cast(pl.Int32),
        pl.col(TS).cast(pl.Int64),
        pl.col(TYPE).cast(pl.Int8),
    )
    max_ts = events.select(pl.max(TS)).collect().item()
    cutoff = args.cutoff_ts
    if cutoff is None:
        cutoff = int(max_ts - args.valid_days * 86_400_000)

    history_path = output_dir / "history_plus_valid_prefix.parquet"
    prefix_path = output_dir / "valid_prefix.parquet"
    suffix_path = output_dir / "valid_suffix.parquet"
    targets_path = output_dir / "valid_targets.parquet"
    report_path = output_dir / "split_report.csv"

    global_history = events.filter(pl.col(TS) < cutoff)
    validation = events.filter(pl.col(TS) >= cutoff)
    validation_session_stats = validation.group_by(SESSION).agg(
        pl.len().alias("_length"), pl.col(TS).n_unique().alias("_timestamp_count")
    )
    dropped_tied_sessions = validation_session_stats.filter(
        (pl.col("_length") >= 2) & (pl.col("_timestamp_count") < 2)
    ).select(pl.len()).collect().item()
    prefix, suffix = split_validation_window(validation, args.prefix_fraction)

    prefix.sink_parquet(prefix_path)
    suffix.sink_parquet(suffix_path)
    pl.concat([global_history, pl.scan_parquet(prefix_path)], how="vertical").sink_parquet(history_path)
    (
        pl.scan_parquet(suffix_path)
        .group_by([SESSION, AID])
        .agg(*[(pl.col(TYPE) == event_type).any().cast(pl.Int8).alias(name) for event_type, name in TARGET_COLUMNS.items()])
        .sink_parquet(targets_path)
    )

    report = pl.concat(
        [
            pl.DataFrame(
                {
                    "cutoff_ts": [cutoff],
                    "prefix_fraction": [args.prefix_fraction],
                    "dropped_sessions_without_time_split": [dropped_tied_sessions],
                }
            ),
            pl.scan_parquet(prefix_path).select(
                pl.len().alias("prefix_rows"), pl.col(SESSION).n_unique().alias("prefix_sessions")
            ).collect(),
            pl.scan_parquet(suffix_path).select(
                pl.len().alias("suffix_rows"), pl.col(SESSION).n_unique().alias("suffix_sessions"),
                (pl.col(TYPE) == 2).sum().alias("suffix_order_events"),
            ).collect(),
            pl.scan_parquet(targets_path).select(
                pl.len().alias("suffix_target_pairs"),
                pl.sum("target_order").alias("suffix_order_pairs"),
            ).collect(),
            pl.scan_parquet(targets_path)
            .filter(pl.col("target_order") == 1)
            .select(pl.col(SESSION).n_unique().alias("suffix_order_sessions"))
            .collect(),
            pl.DataFrame({"elapsed_seconds": [time.perf_counter() - started]}),
        ],
        how="horizontal_extend",
    )
    report.write_csv(report_path)
    print(f"cutoff_ts={cutoff}")
    print(f"prefix -> {prefix_path}")
    print(f"suffix -> {suffix_path}")
    print(f"history_plus_valid_prefix -> {history_path}")
    print(f"targets -> {targets_path}")
    print(f"report -> {report_path}")
    print(report)


if __name__ == "__main__":
    main()
