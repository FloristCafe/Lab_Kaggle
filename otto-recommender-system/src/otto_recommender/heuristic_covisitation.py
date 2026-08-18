"""Heuristic multi-rule co-visitation graphs for OTTO retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

from .polars_graph import AID_X, AID_Y, RANK, WEIGHT, cast_events, evaluate_recall_at_k, prune_topk
from .schema import AID, SESSION, TS, TYPE

GRAPH = "graph"
SCORE = "score"
ITEM_FREQ = "item_freq"
ITEM_FREQ_X = "item_freq_x"
ITEM_FREQ_Y = "item_freq_y"

SECONDS_PER_DAY = 86_400.0
MS_PER_SECOND = 1_000.0


@dataclass(frozen=True)
class CoVisitationRule:
    """Configuration for one heuristic co-visitation graph."""

    name: str
    source_types: tuple[int, ...]
    target_types: tuple[int, ...]
    max_time_delta_seconds: float
    max_events_per_session: int
    topk_per_chunk: int
    final_topk_per_source: int
    decay_c_seconds: float
    degree_alpha: float = 0.5
    graph_weight: float = 1.0


CLICK_TO_CLICK = CoVisitationRule(
    name="click_to_click",
    source_types=(0,),
    target_types=(0,),
    max_time_delta_seconds=24 * SECONDS_PER_DAY / 24,
    max_events_per_session=30,
    topk_per_chunk=80,
    final_topk_per_source=20,
    decay_c_seconds=6 * 60 * 60,
    degree_alpha=0.5,
    graph_weight=1.0,
)

CART_ORDER_TO_CART_ORDER = CoVisitationRule(
    name="cart_order_to_cart_order",
    source_types=(1, 2),
    target_types=(1, 2),
    max_time_delta_seconds=14 * SECONDS_PER_DAY,
    max_events_per_session=30,
    topk_per_chunk=80,
    final_topk_per_source=20,
    decay_c_seconds=7 * SECONDS_PER_DAY,
    degree_alpha=0.5,
    graph_weight=1.5,
)

CLICK_TO_CART_ORDER = CoVisitationRule(
    name="click_to_cart_order",
    source_types=(0,),
    target_types=(1, 2),
    max_time_delta_seconds=7 * SECONDS_PER_DAY,
    max_events_per_session=40,
    topk_per_chunk=100,
    final_topk_per_source=20,
    decay_c_seconds=2 * SECONDS_PER_DAY,
    degree_alpha=0.5,
    graph_weight=2.0,
)

DEFAULT_RULES = (
    CLICK_TO_CLICK,
    CART_ORDER_TO_CART_ORDER,
    CLICK_TO_CART_ORDER,
)


def target_value_expr(type_col: str = "type_y") -> pl.Expr:
    """Assign commercial target value to OTTO type ids."""
    return (
        pl.when(pl.col(type_col) == 0)
        .then(1.0)
        .when(pl.col(type_col) == 1)
        .then(3.0)
        .when(pl.col(type_col) == 2)
        .then(6.0)
        .otherwise(1.0)
        .cast(pl.Float32)
    )


def time_decay_expr(rule: CoVisitationRule, dt_seconds_col: str = "dt_seconds") -> pl.Expr:
    """Return a Rust-vectorized rational time decay expression."""
    dt = pl.col(dt_seconds_col).cast(pl.Float32)
    c = pl.lit(rule.decay_c_seconds, dtype=pl.Float32)
    return (c / (dt + c)).cast(pl.Float32)


def degree_penalty_expr(rule: CoVisitationRule) -> pl.Expr:
    """Penalize hub items using global interaction frequency."""
    nx = pl.col(ITEM_FREQ_X).cast(pl.Float32).clip(1.0)
    ny = pl.col(ITEM_FREQ_Y).cast(pl.Float32).clip(1.0)
    return (1.0 / ((nx * ny) ** rule.degree_alpha)).cast(pl.Float32)


def attach_degree_penalty(
    pairs: pl.DataFrame,
    frequencies: pl.DataFrame,
    rule: CoVisitationRule,
) -> pl.DataFrame:
    """Hash-join global item frequencies after pair generation."""
    return (
        pairs.join(frequencies, left_on=AID_X, right_on=AID, how="left")
        .rename({ITEM_FREQ: ITEM_FREQ_X})
        .join(frequencies, left_on=AID_Y, right_on=AID, how="left")
        .rename({ITEM_FREQ: ITEM_FREQ_Y})
        .with_columns(
            pl.col(ITEM_FREQ_X).fill_null(1),
            pl.col(ITEM_FREQ_Y).fill_null(1),
        )
        .with_columns(degree_penalty_expr(rule).alias("_degree_penalty"))
    )


def item_frequency(events: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Count global item interaction frequency for degree penalty."""
    frequency_frame = (
        events.select(pl.col(AID).cast(pl.Int32))
        .group_by(AID)
        .agg(pl.len().cast(pl.UInt32).alias(ITEM_FREQ))
    )
    if isinstance(frequency_frame, pl.LazyFrame):
        return frequency_frame.collect()
    return frequency_frame


def recent_session_events(events: pl.DataFrame, max_events_per_session: int) -> pl.DataFrame:
    """Keep only the most recent events per session before pair generation."""
    return (
        cast_events(events)
        .sort([SESSION, TS])
        .with_columns(
            pl.len().over(SESSION).cast(pl.Int32).alias("_session_len"),
            pl.cum_count(AID).over(SESSION).cast(pl.Int32).alias("_pos"),
        )
        .filter(pl.col("_pos") > (pl.col("_session_len") - max_events_per_session).clip(0))
        .drop("_session_len", "_pos")
    )


def build_rule_edges(
    events: pl.DataFrame,
    rule: CoVisitationRule,
    frequencies: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build directed COO edges for one heuristic rule."""
    recent = recent_session_events(events, rule.max_events_per_session)
    frequencies = item_frequency(events) if frequencies is None else frequencies
    sources = (
        recent.filter(pl.col(TYPE).is_in(rule.source_types))
        .select(
            pl.col(SESSION),
            pl.col(AID).alias(AID_X),
            pl.col(TS).alias("ts_x"),
            pl.col(TYPE).alias("type_x"),
        )
    )
    targets = (
        recent.filter(pl.col(TYPE).is_in(rule.target_types))
        .select(
            pl.col(SESSION),
            pl.col(AID).alias(AID_Y),
            pl.col(TS).alias("ts_y"),
            pl.col(TYPE).alias("type_y"),
        )
    )
    if sources.is_empty() or targets.is_empty():
        return pl.DataFrame(
            schema={
                AID_X: pl.Int32,
                AID_Y: pl.Int32,
                WEIGHT: pl.Float32,
                GRAPH: pl.Utf8,
            }
        )

    pairs = (
        sources.join(targets, on=SESSION, how="inner")
        .with_columns(((pl.col("ts_y") - pl.col("ts_x")) / MS_PER_SECOND).alias("dt_seconds"))
        .filter(
            (pl.col(AID_X) != pl.col(AID_Y))
            & (pl.col("dt_seconds") > 0)
            & (pl.col("dt_seconds") <= rule.max_time_delta_seconds)
        )
    )
    pairs = (
        attach_degree_penalty(pairs, frequencies, rule)
        .with_columns(
            (
                target_value_expr("type_y")
                * time_decay_expr(rule, "dt_seconds")
                * pl.col("_degree_penalty")
                * pl.lit(rule.graph_weight, dtype=pl.Float32)
            )
            .cast(pl.Float32)
            .alias(WEIGHT)
        )
    )
    if pairs.is_empty():
        return pl.DataFrame(
            schema={
                AID_X: pl.Int32,
                AID_Y: pl.Int32,
                WEIGHT: pl.Float32,
                GRAPH: pl.Utf8,
            }
        )
    return (
        pairs.group_by([AID_X, AID_Y])
        .agg(pl.col(WEIGHT).sum().cast(pl.Float32).alias(WEIGHT))
        .select(
            pl.col(AID_X).cast(pl.Int32),
            pl.col(AID_Y).cast(pl.Int32),
            pl.col(WEIGHT).cast(pl.Float32),
            pl.lit(rule.name).alias(GRAPH),
        )
    )


def build_pruned_rule_parts(
    input_path: str | Path,
    output_dir: str | Path,
    rule: CoVisitationRule,
    n_buckets: int,
    frequencies: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build chunked and pruned edge parts for one rule."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frequencies = item_frequency(pl.scan_parquet(input_path)) if frequencies is None else frequencies

    stats: list[dict[str, int | str]] = []
    for bucket in range(n_buckets):
        events = (
            cast_events(pl.scan_parquet(input_path))
            .filter((pl.col(SESSION) % n_buckets) == bucket)
            .collect()
        )
        edges = build_rule_edges(events, rule, frequencies=frequencies)
        pruned = prune_topk(edges, AID_X, WEIGHT, rule.topk_per_chunk)
        part_path = output_dir / rule.name / f"edges_part_{bucket:03d}.parquet"
        part_path.parent.mkdir(parents=True, exist_ok=True)
        pruned.write_parquet(part_path)
        stats.append(
            {
                "graph": rule.name,
                "bucket": bucket,
                "events": events.height,
                "edges_before_prune": edges.height,
                "edges_after_prune": pruned.height,
                "path": str(part_path),
            }
        )
    return pl.DataFrame(stats)


def merge_rule_parts(
    part_paths: Iterable[str | Path],
    output_path: str | Path,
    final_topk_per_source: int,
) -> pl.DataFrame:
    """Merge and re-prune one rule's edge parts."""
    paths = [str(Path(path)) for path in part_paths]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not paths:
        raise ValueError(f"No edge parts found for {output_path}")
    merged = (
        pl.scan_parquet(paths)
        .group_by([AID_X, AID_Y, GRAPH])
        .agg(pl.col(WEIGHT).sum().cast(pl.Float32).alias(WEIGHT))
        .collect()
    )
    final_edges = prune_topk(merged, AID_X, WEIGHT, final_topk_per_source).sort(
        [AID_X, WEIGHT],
        descending=[False, True],
    )
    final_edges.write_parquet(output_path)
    return final_edges


def build_all_rule_graphs(
    input_path: str | Path,
    output_dir: str | Path,
    n_buckets: int = 16,
    rules: Iterable[CoVisitationRule] = DEFAULT_RULES,
    degree_alpha: float | None = None,
) -> pl.DataFrame:
    """Build all heuristic graphs and return build statistics."""
    output_dir = Path(output_dir)
    input_path = Path(input_path)
    frequencies = item_frequency(pl.scan_parquet(input_path))
    stats: list[pl.DataFrame] = []
    for rule in rules:
        if degree_alpha is not None:
            rule = CoVisitationRule(
                name=rule.name,
                source_types=rule.source_types,
                target_types=rule.target_types,
                max_time_delta_seconds=rule.max_time_delta_seconds,
                max_events_per_session=rule.max_events_per_session,
                topk_per_chunk=rule.topk_per_chunk,
                final_topk_per_source=rule.final_topk_per_source,
                decay_c_seconds=rule.decay_c_seconds,
                degree_alpha=degree_alpha,
                graph_weight=rule.graph_weight,
            )
        parts_dir = output_dir / "edge_parts"
        rule_stats = build_pruned_rule_parts(
            input_path=input_path,
            output_dir=parts_dir,
            rule=rule,
            n_buckets=n_buckets,
            frequencies=frequencies,
        )
        part_paths = sorted((parts_dir / rule.name).glob("edges_part_*.parquet"))
        merge_rule_parts(
            part_paths=part_paths,
            output_path=output_dir / f"{rule.name}_top20.parquet",
            final_topk_per_source=rule.final_topk_per_source,
        )
        stats.append(rule_stats)
    return pl.concat(stats, how="vertical")


def recommend_from_heuristic_graphs(
    events: pl.DataFrame,
    edge_frames: Iterable[pl.DataFrame],
    topk: int = 20,
    min_candidates: int = 20,
    max_seed_items: int = 20,
    recent_weight: float = 60.0,
    popular_fallback: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Aggregate recent-item and multi-rule graph retrieval into session candidates."""
    edges = pl.concat(list(edge_frames), how="vertical").select(
        pl.col(AID_X).cast(pl.Int32),
        pl.col(AID_Y).cast(pl.Int32),
        pl.col(WEIGHT).cast(pl.Float32),
        pl.col(GRAPH),
    )
    seeds = (
        cast_events(events)
        .sort([SESSION, TS], descending=[False, True])
        .group_by(SESSION, maintain_order=True)
        .head(max_seed_items)
        .with_columns(pl.cum_count(AID).over(SESSION).cast(pl.UInt32).alias("_seed_rank"))
    )
    recent_candidates = seeds.select(
        pl.col(SESSION),
        pl.col(AID),
        (pl.lit(recent_weight) / pl.col("_seed_rank").cast(pl.Float32)).alias(SCORE),
        pl.lit("recent").alias(GRAPH),
    )
    graph_candidates = (
        seeds.select(pl.col(SESSION), pl.col(AID).alias(AID_X))
        .join(edges, on=AID_X, how="inner")
        .select(
            pl.col(SESSION),
            pl.col(AID_Y).alias(AID),
            pl.col(WEIGHT).alias(SCORE),
            pl.col(GRAPH),
        )
    )
    candidate_scores = (
        pl.concat([recent_candidates, graph_candidates], how="vertical")
        .group_by([SESSION, AID])
        .agg(pl.col(SCORE).sum().cast(pl.Float32).alias(SCORE))
    )
    if popular_fallback is not None and min_candidates > 0:
        sparse_sessions = (
            candidate_scores.with_columns(
                pl.col(SCORE)
                .rank(method="ordinal", descending=True)
                .over(SESSION)
                .cast(pl.UInt32)
                .alias(RANK)
            )
            .filter(pl.col(RANK) <= topk)
            .group_by(SESSION)
            .agg(pl.len().alias("_candidate_count"))
            .filter(pl.col("_candidate_count") < min_candidates)
            .select(SESSION)
        )
        if not sparse_sessions.is_empty():
            fallback = popular_fallback.head(topk).with_columns(
                (pl.lit(recent_weight * 0.01) / pl.col(RANK).cast(pl.Float32)).alias(SCORE),
                pl.lit("popular_fallback").alias(GRAPH),
            )
            fallback_candidates = sparse_sessions.join(fallback, how="cross").select(SESSION, AID, SCORE)
            candidate_scores = (
                pl.concat([candidate_scores, fallback_candidates], how="vertical")
                .group_by([SESSION, AID])
                .agg(pl.col(SCORE).sum().cast(pl.Float32).alias(SCORE))
            )

    return (
        candidate_scores
        .with_columns(
            pl.col(SCORE)
            .rank(method="ordinal", descending=True)
            .over(SESSION)
            .cast(pl.UInt32)
            .alias(RANK)
        )
        .filter(pl.col(RANK) <= topk)
        .sort([SESSION, RANK])
        .select(
            pl.col(SESSION).cast(pl.Int32),
            pl.col(AID).cast(pl.Int32),
            pl.col(SCORE).cast(pl.Float32),
            pl.col(RANK).cast(pl.UInt32),
        )
    )


def popular_fallback_items(events: pl.DataFrame, topk: int = 100) -> pl.DataFrame:
    """Return globally popular items for filling sparse candidate pools."""
    return (
        cast_events(events)
        .with_columns(
            pl.when(pl.col(TYPE) == 0)
            .then(1.0)
            .when(pl.col(TYPE) == 1)
            .then(3.0)
            .when(pl.col(TYPE) == 2)
            .then(6.0)
            .otherwise(1.0)
            .cast(pl.Float32)
            .alias(SCORE)
        )
        .group_by(AID)
        .agg(pl.col(SCORE).sum().cast(pl.Float32).alias(SCORE))
        .with_columns(
            pl.col(SCORE)
            .rank(method="ordinal", descending=True)
            .cast(pl.UInt32)
            .alias(RANK)
        )
        .filter(pl.col(RANK) <= topk)
        .sort(RANK)
        .select(pl.col(AID).cast(pl.Int32), pl.col(SCORE), pl.col(RANK))
    )


def evaluate_heuristic_recommendations(
    labels: pl.DataFrame,
    recommendations: pl.DataFrame,
    k: int = 20,
) -> pl.DataFrame:
    """Thin wrapper to keep the stage-two script readable."""
    return evaluate_recall_at_k(labels, recommendations.rename({SCORE: "score"}), k=k)
