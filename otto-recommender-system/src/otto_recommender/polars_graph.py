"""Polars-first graph utilities for OTTO co-visitation retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl

from .schema import AID, SESSION, TS, TYPE

AID_X = "aid_x"
AID_Y = "aid_y"
WEIGHT = "weight"
RANK = "rank"
GROUND_TRUTH = "ground_truth"


def cast_events(events: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Normalize OTTO event dtypes for compact graph operations."""
    return events.select(
        pl.col(SESSION).cast(pl.Int32),
        pl.col(AID).cast(pl.Int32),
        pl.col(TS).cast(pl.Int64),
        pl.col(TYPE).cast(pl.Int8),
    )


def event_weight_expr(type_col: str = TYPE) -> pl.Expr:
    """Map OTTO event type ids to retrieval weights."""
    return (
        pl.when(pl.col(type_col) == 0)
        .then(1.0)
        .when(pl.col(type_col) == 1)
        .then(6.0)
        .when(pl.col(type_col) == 2)
        .then(3.0)
        .otherwise(1.0)
        .cast(pl.Float32)
    )


def prune_topk(
    frame: pl.DataFrame,
    group_col: str,
    score_col: str,
    topk: int,
    rank_col: str = RANK,
) -> pl.DataFrame:
    """Keep the highest-scoring rows per group."""
    if frame.is_empty():
        return frame.with_columns(pl.lit(None, dtype=pl.UInt32).alias(rank_col)).drop(rank_col)
    return (
        frame.with_columns(
            pl.col(score_col)
            .rank(method="ordinal", descending=True)
            .over(group_col)
            .cast(pl.UInt32)
            .alias(rank_col)
        )
        .filter(pl.col(rank_col) <= topk)
        .drop(rank_col)
    )


def split_train_valid_tail(
    events: pl.DataFrame,
    valid_events_per_session: int = 1,
    min_session_length: int = 2,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Hold out each session's final events as local validation labels."""
    events = cast_events(events)
    marked = (
        events.sort([SESSION, TS])
        .with_columns(
            pl.len().over(SESSION).cast(pl.Int32).alias("_session_len"),
            pl.cum_count(AID).over(SESSION).cast(pl.Int32).alias("_pos"),
        )
        .with_columns(
            (
                (pl.col("_session_len") >= min_session_length)
                & (pl.col("_pos") > pl.col("_session_len") - valid_events_per_session)
            ).alias("_is_valid")
        )
    )
    train = marked.filter(~pl.col("_is_valid")).drop("_session_len", "_pos", "_is_valid")
    labels = (
        marked.filter(pl.col("_is_valid"))
        .group_by([SESSION, TYPE])
        .agg(pl.col(AID).cast(pl.Int32).unique(maintain_order=True).alias(GROUND_TRUTH))
        .sort([SESSION, TYPE])
    )
    return train, labels


def build_covisitation_edges(
    events: pl.DataFrame,
    max_events_per_session: int = 30,
) -> pl.DataFrame:
    """Build directed COO-style co-visitation edges: aid_x -> aid_y."""
    events = cast_events(events)
    recent = (
        events.sort([SESSION, TS])
        .with_columns(
            pl.len().over(SESSION).cast(pl.Int32).alias("_session_len"),
            pl.cum_count(AID).over(SESSION).cast(pl.Int32).alias("_pos"),
        )
        .filter(pl.col("_pos") > (pl.col("_session_len") - max_events_per_session).clip(0))
        .drop("_session_len", "_pos")
    )

    left = recent.select(
        pl.col(SESSION),
        pl.col(AID).alias(AID_X),
        pl.col(TS).alias("ts_x"),
    )
    right = recent.select(
        pl.col(SESSION),
        pl.col(AID).alias(AID_Y),
        pl.col(TS).alias("ts_y"),
        pl.col(TYPE).alias("type_y"),
    )
    return (
        left.join(right, on=SESSION, how="inner")
        .filter((pl.col(AID_X) != pl.col(AID_Y)) & (pl.col("ts_x") < pl.col("ts_y")))
        .with_columns(
            (
                event_weight_expr("type_y")
                / (1.0 + ((pl.col("ts_y") - pl.col("ts_x")).cast(pl.Float32) / 86_400_000.0))
            )
            .cast(pl.Float32)
            .alias(WEIGHT)
        )
        .group_by([AID_X, AID_Y])
        .agg(pl.col(WEIGHT).sum().cast(pl.Float32).alias(WEIGHT))
        .select(
            pl.col(AID_X).cast(pl.Int32),
            pl.col(AID_Y).cast(pl.Int32),
            pl.col(WEIGHT).cast(pl.Float32),
        )
    )


def build_pruned_edge_parts(
    input_path: str | Path,
    output_dir: str | Path,
    n_buckets: int = 16,
    max_events_per_session: int = 30,
    topk_per_chunk: int = 80,
) -> pl.DataFrame:
    """Build pruned edge-list chunks by session hash bucket."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: list[dict[str, int | str]] = []
    for bucket in range(n_buckets):
        events = (
            cast_events(pl.scan_parquet(input_path))
            .filter((pl.col(SESSION) % n_buckets) == bucket)
            .collect()
        )
        edges = build_covisitation_edges(
            events,
            max_events_per_session=max_events_per_session,
        )
        pruned = prune_topk(edges, AID_X, WEIGHT, topk_per_chunk)
        part_path = output_dir / f"edges_part_{bucket:03d}.parquet"
        pruned.write_parquet(part_path)
        stats.append(
            {
                "bucket": bucket,
                "events": events.height,
                "edges_before_prune": edges.height,
                "edges_after_prune": pruned.height,
                "path": str(part_path),
            }
        )
    return pl.DataFrame(stats)


def merge_pruned_edges(
    edge_paths: Iterable[str | Path],
    output_path: str | Path,
    final_topk_per_source: int = 20,
) -> pl.DataFrame:
    """Merge pruned edge chunks, aggregate weights, and prune again."""
    paths = [str(Path(path)) for path in edge_paths]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged = (
        pl.scan_parquet(paths)
        .group_by([AID_X, AID_Y])
        .agg(pl.col(WEIGHT).sum().cast(pl.Float32).alias(WEIGHT))
        .collect()
    )
    final_edges = prune_topk(merged, AID_X, WEIGHT, final_topk_per_source).sort(
        [AID_X, WEIGHT],
        descending=[False, True],
    )
    final_edges.write_parquet(output_path)
    return final_edges


def recommend_from_edges(
    events: pl.DataFrame,
    edges: pl.DataFrame,
    topk: int = 20,
    max_seed_items: int = 20,
    recent_weight: float = 50.0,
) -> pl.DataFrame:
    """Create per-session recommendations with DataFrame joins instead of nested dicts."""
    events = cast_events(events)
    edges = edges.select(
        pl.col(AID_X).cast(pl.Int32),
        pl.col(AID_Y).cast(pl.Int32),
        pl.col(WEIGHT).cast(pl.Float32),
    )
    seeds = (
        events.sort([SESSION, TS], descending=[False, True])
        .group_by(SESSION, maintain_order=True)
        .head(max_seed_items)
        .with_columns(pl.cum_count(AID).over(SESSION).cast(pl.UInt32).alias("_seed_rank"))
    )
    recent_candidates = seeds.select(
        pl.col(SESSION),
        pl.col(AID).alias(AID),
        (pl.lit(recent_weight) / pl.col("_seed_rank").cast(pl.Float32)).alias("score"),
    )
    graph_candidates = (
        seeds.select(pl.col(SESSION), pl.col(AID).alias(AID_X))
        .join(edges, on=AID_X, how="inner")
        .select(
            pl.col(SESSION),
            pl.col(AID_Y).alias(AID),
            pl.col(WEIGHT).alias("score"),
        )
    )
    candidates = (
        pl.concat([recent_candidates, graph_candidates], how="vertical")
        .group_by([SESSION, AID])
        .agg(pl.col("score").sum().cast(pl.Float32).alias("score"))
        .with_columns(
            pl.col("score")
            .rank(method="ordinal", descending=True)
            .over(SESSION)
            .cast(pl.UInt32)
            .alias(RANK)
        )
        .filter(pl.col(RANK) <= topk)
        .sort([SESSION, RANK])
    )
    return candidates.select(
        pl.col(SESSION).cast(pl.Int32),
        pl.col(AID).cast(pl.Int32),
        pl.col("score").cast(pl.Float32),
        pl.col(RANK).cast(pl.UInt32),
    )


def evaluate_recall_at_k(
    labels: pl.DataFrame,
    recommendations: pl.DataFrame,
    k: int = 20,
) -> pl.DataFrame:
    """Compute per-type and weighted Recall@K from Polars recommendation rows."""
    labels_exp = labels.explode(GROUND_TRUTH).rename({GROUND_TRUTH: AID})
    recs_top = recommendations.filter(pl.col(RANK) <= k).select(SESSION, AID).unique()
    totals = labels_exp.group_by(TYPE).agg(pl.len().alias("total"))
    hits = (
        labels_exp.join(recs_top, on=[SESSION, AID], how="inner")
        .group_by(TYPE)
        .agg(pl.len().alias("hits"))
    )
    scores = (
        totals.join(hits, on=TYPE, how="left")
        .with_columns(pl.col("hits").fill_null(0))
        .with_columns((pl.col("hits") / pl.col("total")).cast(pl.Float32).alias("recall"))
        .with_columns(
            pl.when(pl.col(TYPE) == 0)
            .then(0.10)
            .when(pl.col(TYPE) == 1)
            .then(0.30)
            .when(pl.col(TYPE) == 2)
            .then(0.60)
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("metric_weight")
        )
        .sort(TYPE)
    )
    weighted = (
        scores.select(
            ((pl.col("recall") * pl.col("metric_weight")).sum() / pl.col("metric_weight").sum())
            .cast(pl.Float32)
            .alias("weighted_recall")
        )
        .item()
    )
    return scores.with_columns(pl.lit(weighted).cast(pl.Float32).alias("weighted_recall"))
