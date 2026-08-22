"""Hard negative mining for sparse conversion rankers."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

from .schema import AID, SESSION

HASH_DENOMINATOR = 1_000_000

ITEM_FEATURE_COLUMNS = (
    "total_interactions",
    "click_count",
    "cart_count",
    "order_count",
    "recent_24h_interactions",
    "conversion_rate",
    "item_cart_conversion_rate",
    "item_buy_conversion_rate",
    "item_cart_to_order_rate",
    "item_funnel_dropoff_rate",
)


@dataclass(frozen=True)
class HardNegativeConfig:
    """Sampling quotas expressed as negatives per positive."""

    target_col: str
    graph_col: str
    hard_click_neg_per_pos: float = 8.0
    hard_graph_neg_per_pos: float = 8.0
    random_neg_per_pos: float = 4.0
    min_local_clicks: int = 2
    graph_quantile: float = 0.80
    seed: int = 42


def _scan_paths(paths: Iterable[str | Path]) -> pl.LazyFrame:
    return pl.scan_parquet([str(path) for path in paths])


def graph_weight_threshold(
    paths: Iterable[str | Path],
    target_col: str,
    graph_col: str,
    graph_quantile: float,
) -> float:
    """Estimate the hard graph threshold from non-converted rows with positive graph weight."""
    value = (
        _scan_paths(paths)
        .filter((pl.col(target_col) == 0) & (pl.col(graph_col) > 0))
        .select(pl.col(graph_col).quantile(graph_quantile).alias("_threshold"))
        .collect()
        .item()
    )
    return float(value or 0.0)


def with_hard_negative_bucket(
    frame: pl.LazyFrame,
    config: HardNegativeConfig,
    graph_threshold: float,
) -> pl.LazyFrame:
    """Assign mutually exclusive positive, graph-hard, click-hard, and random buckets."""
    graph_hard_expr = (pl.col(config.graph_col) > 0) & (pl.col(config.graph_col) >= graph_threshold)
    click_hard_expr = pl.col("local_click_count") >= config.min_local_clicks
    return frame.with_columns(
        pl.when(pl.col(config.target_col) == 1)
        .then(pl.lit("positive"))
        .when(graph_hard_expr)
        .then(pl.lit("hard_graph"))
        .when(click_hard_expr)
        .then(pl.lit("hard_click"))
        .otherwise(pl.lit("random"))
        .alias("negative_bucket")
    )


def bucket_counts(
    paths: Iterable[str | Path],
    config: HardNegativeConfig,
    graph_threshold: float,
) -> pl.DataFrame:
    """Count rows per sampling bucket before downsampling."""
    return (
        with_hard_negative_bucket(_scan_paths(paths), config, graph_threshold)
        .group_by("negative_bucket")
        .agg(pl.len().alias("available_rows"))
        .collect()
    )


def sampling_thresholds(counts: pl.DataFrame, config: HardNegativeConfig) -> dict[str, int]:
    """Translate per-bucket quotas into deterministic hash thresholds."""
    count_by_bucket = {
        row["negative_bucket"]: int(row["available_rows"])
        for row in counts.select("negative_bucket", "available_rows").iter_rows(named=True)
    }
    positives = count_by_bucket.get("positive", 0)
    if positives <= 0:
        raise ValueError(f"No positives found for {config.target_col}")

    quotas = {
        "hard_click": int(positives * config.hard_click_neg_per_pos),
        "hard_graph": int(positives * config.hard_graph_neg_per_pos),
        "random": int(positives * config.random_neg_per_pos),
    }
    thresholds: dict[str, int] = {}
    for bucket, quota in quotas.items():
        available = count_by_bucket.get(bucket, 0)
        fraction = min(1.0, quota / max(available, 1))
        thresholds[bucket] = int(fraction * HASH_DENOMINATOR)
    return thresholds


def _item_feature_scan(item_features_path: str | Path) -> pl.LazyFrame:
    available = pl.scan_parquet(item_features_path).collect_schema().names()
    columns = [AID, *[col for col in ITEM_FEATURE_COLUMNS if col in available]]
    return pl.scan_parquet(item_features_path).select(
        pl.col(AID).cast(pl.Int32),
        *[pl.col(col) for col in columns if col != AID],
    )


def sampled_hard_negative_frame(
    path: str | Path,
    config: HardNegativeConfig,
    graph_threshold: float,
    thresholds: dict[str, int],
    item_features_path: str | Path | None = None,
) -> pl.LazyFrame:
    """Build one lazily sampled part, optionally enriched with global item features."""
    frame = with_hard_negative_bucket(pl.scan_parquet(path), config, graph_threshold).with_columns(
        pl.struct([pl.col(SESSION), pl.col(AID), pl.col("negative_bucket")])
        .hash(seed=config.seed)
        .mod(HASH_DENOMINATOR)
        .alias("_sample_hash")
    )
    keep = (
        (pl.col("negative_bucket") == "positive")
        | ((pl.col("negative_bucket") == "hard_click") & (pl.col("_sample_hash") < thresholds["hard_click"]))
        | ((pl.col("negative_bucket") == "hard_graph") & (pl.col("_sample_hash") < thresholds["hard_graph"]))
        | ((pl.col("negative_bucket") == "random") & (pl.col("_sample_hash") < thresholds["random"]))
    )
    sampled = frame.filter(keep).drop("_sample_hash")
    if item_features_path is not None:
        sampled = sampled.join(_item_feature_scan(item_features_path), on=AID, how="left")
    return sampled


def write_hard_negative_parts(
    input_paths: Iterable[str | Path],
    output_dir: str | Path,
    stats_output: str | Path,
    config: HardNegativeConfig,
    item_features_path: str | Path | None = None,
) -> pl.DataFrame:
    """Write hard-negative sampled parts and return per-part audit statistics."""
    paths = [Path(path) for path in input_paths]
    if not paths:
        raise FileNotFoundError("No input parquet parts were provided.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_output = Path(stats_output)
    stats_output.parent.mkdir(parents=True, exist_ok=True)

    graph_threshold = graph_weight_threshold(paths, config.target_col, config.graph_col, config.graph_quantile)
    counts = bucket_counts(paths, config, graph_threshold)
    thresholds = sampling_thresholds(counts, config)

    stats: list[dict[str, float | int | str]] = []
    for path in paths:
        output_path = output_dir / path.name
        sampled = sampled_hard_negative_frame(
            path=path,
            config=config,
            graph_threshold=graph_threshold,
            thresholds=thresholds,
            item_features_path=item_features_path,
        )
        sampled.sink_parquet(output_path)
        part_stats = (
            pl.scan_parquet(output_path)
            .group_by("negative_bucket")
            .agg(pl.len().alias("rows"))
            .collect()
        )
        row = {
            "part": path.name,
            "graph_threshold": graph_threshold,
            "threshold_hard_click": thresholds["hard_click"],
            "threshold_hard_graph": thresholds["hard_graph"],
            "threshold_random": thresholds["random"],
            "path": str(output_path),
        }
        for bucket in ("positive", "hard_click", "hard_graph", "random"):
            bucket_rows = part_stats.filter(pl.col("negative_bucket") == bucket)
            row[f"{bucket}_rows"] = int(bucket_rows.select("rows").item()) if bucket_rows.height else 0
        row["rows"] = sum(int(row[f"{bucket}_rows"]) for bucket in ("positive", "hard_click", "hard_graph", "random"))
        row["negative_rows"] = int(row["rows"]) - int(row["positive_rows"])
        stats.append(row)
        del sampled, part_stats
        gc.collect()

    stats_frame = pl.DataFrame(stats)
    totals = stats_frame.select(
        pl.lit("__TOTAL__").alias("part"),
        pl.first("graph_threshold").alias("graph_threshold"),
        pl.first("threshold_hard_click").alias("threshold_hard_click"),
        pl.first("threshold_hard_graph").alias("threshold_hard_graph"),
        pl.first("threshold_random").alias("threshold_random"),
        pl.lit("").alias("path"),
        pl.sum("positive_rows").alias("positive_rows"),
        pl.sum("hard_click_rows").alias("hard_click_rows"),
        pl.sum("hard_graph_rows").alias("hard_graph_rows"),
        pl.sum("random_rows").alias("random_rows"),
        pl.sum("rows").alias("rows"),
        pl.sum("negative_rows").alias("negative_rows"),
    ).with_columns((pl.col("negative_rows") / pl.col("positive_rows")).alias("neg_pos_ratio"))
    result = pl.concat([stats_frame.with_columns((pl.col("negative_rows") / pl.col("positive_rows")).alias("neg_pos_ratio")), totals], how="diagonal")
    result.write_csv(stats_output)
    return result
