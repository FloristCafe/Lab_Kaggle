"""Low-memory feature engineering for OTTO item and session tables."""

from __future__ import annotations

import gc
from pathlib import Path

import polars as pl

from .schema import AID, SESSION, TS, TYPE

CLICK_TYPE = 0
CART_TYPE = 1
ORDER_TYPE = 2
SECONDS_PER_HOUR = 3_600
SECONDS_PER_DAY = 86_400
CONVERSION_PRIOR_STRENGTH = 50.0


def _lazy_events(input_path: str | Path) -> pl.LazyFrame:
    """Read OTTO parquet lazily with stable dtypes."""
    return pl.scan_parquet(input_path).select(
        pl.col(SESSION).cast(pl.Int32),
        pl.col(AID).cast(pl.Int32),
        pl.col(TS).cast(pl.Int64),
        pl.col(TYPE).cast(pl.Int8),
    )


def _clean_sink(target: pl.LazyFrame, output_path: str | Path) -> None:
    """Persist a lazy plan and trigger explicit GC after the write."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target.sink_parquet(output_path)
    gc.collect()


def build_item_features(input_path: str | Path, output_path: str | Path) -> None:
    """Create item_features.parquet with absolute counts, conversion rate, and recent activity."""
    events = _lazy_events(input_path)

    item_counts = events.group_by(AID).agg(
        pl.len().cast(pl.UInt32).alias("total_interactions"),
        (pl.col(TYPE) == CLICK_TYPE).sum().cast(pl.UInt32).alias("click_count"),
        (pl.col(TYPE) == CART_TYPE).sum().cast(pl.UInt32).alias("cart_count"),
        (pl.col(TYPE) == ORDER_TYPE).sum().cast(pl.UInt32).alias("order_count"),
    )

    global_stats = events.select(
        pl.len().cast(pl.Float32).alias("_total_interactions"),
        (pl.col(TYPE) == CLICK_TYPE).sum().cast(pl.Float32).alias("_click_count"),
        (pl.col(TYPE) == CART_TYPE).sum().cast(pl.Float32).alias("_cart_count"),
        (pl.col(TYPE) == ORDER_TYPE).sum().cast(pl.Float32).alias("_order_count"),
    ).collect()
    global_click_count = max(global_stats.item(0, "_click_count"), 1.0)
    global_conversion_rate = (
        global_stats.item(0, "_cart_count") + global_stats.item(0, "_order_count")
    ) / global_click_count

    max_ts = events.select(pl.max(TS).alias("_max_ts")).collect().item(0, 0)
    recent_24h = (
        events.filter(pl.col(TS) >= pl.lit(max_ts - SECONDS_PER_DAY * 1_000))
        .group_by(AID)
        .agg(pl.len().cast(pl.UInt32).alias("recent_24h_interactions"))
    )

    item_features = (
        item_counts.join(recent_24h, on=AID, how="left")
        .with_columns(
            pl.col("recent_24h_interactions").fill_null(0).cast(pl.UInt32),
            (
                (
                    pl.col("cart_count").cast(pl.Float32)
                    + pl.col("order_count").cast(pl.Float32)
                    + pl.lit(CONVERSION_PRIOR_STRENGTH * global_conversion_rate, dtype=pl.Float32)
                )
                / (pl.col("click_count").cast(pl.Float32) + pl.lit(CONVERSION_PRIOR_STRENGTH, dtype=pl.Float32))
            )
            .cast(pl.Float32)
            .alias("conversion_rate"),
        )
        .select(
            pl.col(AID).cast(pl.Int32),
            pl.col("total_interactions").cast(pl.UInt32),
            pl.col("click_count").cast(pl.UInt32),
            pl.col("cart_count").cast(pl.UInt32),
            pl.col("order_count").cast(pl.UInt32),
            pl.col("recent_24h_interactions").cast(pl.UInt32),
            pl.col("conversion_rate").cast(pl.Float32),
        )
        .sort(AID)
    )
    _clean_sink(item_features, output_path)
    del events, global_stats, item_counts, max_ts, recent_24h, item_features
    gc.collect()


def build_user_features(input_path: str | Path, output_path: str | Path) -> None:
    """Create user_features.parquet with session-level activity and window-shopping flag."""
    events = _lazy_events(input_path)

    session_features = (
        events.group_by(SESSION)
        .agg(
            pl.len().cast(pl.UInt32).alias("session_length"),
            pl.col(AID).n_unique().cast(pl.UInt32).alias("unique_items"),
            pl.min(TS).cast(pl.Int64).alias("first_ts"),
            pl.max(TS).cast(pl.Int64).alias("last_ts"),
            (pl.col(TYPE) == CART_TYPE).sum().cast(pl.UInt32).alias("cart_count"),
            (pl.col(TYPE) == ORDER_TYPE).sum().cast(pl.UInt32).alias("order_count"),
        )
        .with_columns(
            (pl.col("last_ts") - pl.col("first_ts")).cast(pl.Int64).alias("duration"),
            (
                ((pl.col("session_length") >= 50) & (pl.col("cart_count") == 0) & (pl.col("order_count") == 0))
                .cast(pl.Int8)
                .alias("is_window_shopping")
            ),
        )
        .select(
            pl.col(SESSION).cast(pl.Int32),
            pl.col("session_length").cast(pl.UInt32),
            pl.col("unique_items").cast(pl.UInt32),
            pl.col("first_ts").cast(pl.Int64),
            pl.col("last_ts").cast(pl.Int64),
            pl.col("duration").cast(pl.Int64),
            pl.col("cart_count").cast(pl.UInt32),
            pl.col("order_count").cast(pl.UInt32),
            pl.col("is_window_shopping").cast(pl.Int8),
        )
        .sort(SESSION)
    )
    _clean_sink(session_features, output_path)
    del events, session_features
    gc.collect()
