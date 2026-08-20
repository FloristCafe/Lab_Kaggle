"""Candidate-level interaction features for ranker training."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterable

import polars as pl

from .heuristic_covisitation import GRAPH, SCORE
from .polars_graph import AID_X, AID_Y, RANK, WEIGHT, cast_events
from .schema import AID, SESSION, TS, TYPE

CLICK_TYPE = 0
CART_TYPE = 1
ORDER_TYPE = 2
DEFAULT_DELTA_T_FILL = 9_999_999

GRAPH_COLUMNS = {
    "click_to_click": "graph_w_click_to_click",
    "cart_order_to_cart_order": "graph_w_cart_order_to_cart_order",
    "click_to_cart_order": "graph_w_click_to_cart_order",
}


def local_item_stats(events: pl.DataFrame) -> pl.DataFrame:
    """Count session-local interactions for each candidate item."""
    return (
        cast_events(events)
        .group_by([SESSION, AID])
        .agg(
            pl.len().cast(pl.UInt16).alias("local_interaction_count"),
            (pl.col(TYPE) == CLICK_TYPE).sum().cast(pl.UInt16).alias("local_click_count"),
            (pl.col(TYPE) == CART_TYPE).sum().cast(pl.UInt16).alias("local_cart_count"),
            (pl.col(TYPE) == ORDER_TYPE).sum().cast(pl.UInt16).alias("local_order_count"),
            pl.max(TS).cast(pl.Int64).alias("item_last_ts"),
        )
    )


def session_last_ts(events: pl.DataFrame) -> pl.DataFrame:
    """Return the last active timestamp for each session."""
    return cast_events(events).group_by(SESSION).agg(pl.max(TS).cast(pl.Int64).alias("session_last_ts"))


def recent_seed_items(events: pl.DataFrame, max_seed_items: int = 20) -> pl.DataFrame:
    """Return recent session items used to propagate graph weights."""
    return (
        cast_events(events)
        .sort([SESSION, TS], descending=[False, True])
        .group_by(SESSION, maintain_order=True)
        .head(max_seed_items)
        .select(pl.col(SESSION), pl.col(AID).alias(AID_X))
        .unique()
    )


def graph_signal_features(
    events: pl.DataFrame,
    edge_frames: Iterable[pl.DataFrame],
    max_seed_items: int = 20,
) -> pl.DataFrame:
    """Propagate raw graph weights from recent seed items to candidate items."""
    edges = pl.concat(list(edge_frames), how="vertical").select(
        pl.col(AID_X).cast(pl.Int32),
        pl.col(AID_Y).cast(pl.Int32),
        pl.col(GRAPH),
        pl.col(WEIGHT).cast(pl.Float32),
    )
    seeds = recent_seed_items(events, max_seed_items=max_seed_items)
    if seeds.is_empty() or edges.is_empty():
        return pl.DataFrame(schema={SESSION: pl.Int32, AID: pl.Int32})

    joined = (
        seeds.join(edges, on=AID_X, how="inner")
        .select(
            pl.col(SESSION),
            pl.col(AID_Y).alias(AID),
            pl.col(GRAPH),
            pl.col(WEIGHT),
        )
    )
    features = (
        joined.group_by([SESSION, AID])
        .agg(
            pl.col(WEIGHT).sum().cast(pl.Float32).alias("graph_weight_sum"),
            pl.col(WEIGHT).max().cast(pl.Float32).alias("graph_weight_max"),
            pl.col(GRAPH).n_unique().cast(pl.UInt8).alias("graph_source_count"),
            *[
                pl.when(pl.col(GRAPH) == graph_name)
                .then(pl.col(WEIGHT))
                .otherwise(0.0)
                .sum()
                .cast(pl.Float32)
                .alias(column_name)
                for graph_name, column_name in GRAPH_COLUMNS.items()
            ],
        )
        .with_columns(
            pl.when(
                (pl.col("graph_w_click_to_cart_order") >= pl.col("graph_w_click_to_click"))
                & (pl.col("graph_w_click_to_cart_order") >= pl.col("graph_w_cart_order_to_cart_order"))
                & (pl.col("graph_w_click_to_cart_order") > 0)
            )
            .then(3)
            .when(
                (pl.col("graph_w_cart_order_to_cart_order") >= pl.col("graph_w_click_to_click"))
                & (pl.col("graph_w_cart_order_to_cart_order") > 0)
            )
            .then(2)
            .when(pl.col("graph_w_click_to_click") > 0)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("dominant_graph_source")
        )
    )
    return features


def graph_signal_features_for_candidates(
    candidates: pl.DataFrame,
    session_events: pl.DataFrame,
    edge_frames: Iterable[pl.DataFrame],
    max_seed_items: int = 20,
) -> pl.DataFrame:
    """Propagate graph weights, then keep only current chunk candidate pairs."""
    candidate_keys = candidates.select(SESSION, AID).unique()
    graph_stats = graph_signal_features(
        events=session_events,
        edge_frames=edge_frames,
        max_seed_items=max_seed_items,
    )
    if graph_stats.is_empty():
        return graph_stats
    return graph_stats.join(candidate_keys, on=[SESSION, AID], how="inner")


def build_interaction_features_for_bucket(
    candidates: pl.DataFrame,
    events: pl.DataFrame,
    edge_frames: Iterable[pl.DataFrame],
    max_seed_items: int = 20,
    delta_t_fill: int = DEFAULT_DELTA_T_FILL,
) -> pl.DataFrame:
    """Join candidate rows with local session-item and graph features."""
    candidates = candidates.select(
        pl.col(SESSION).cast(pl.Int32),
        pl.col(AID).cast(pl.Int32),
        pl.col(SCORE).cast(pl.Float32).alias("candidate_score"),
        pl.col(RANK).cast(pl.UInt32).alias("candidate_rank"),
    )
    local_stats = local_item_stats(events)
    session_stats = session_last_ts(events)
    graph_stats = graph_signal_features(events, edge_frames=edge_frames, max_seed_items=max_seed_items)

    features = (
        candidates.join(local_stats, on=[SESSION, AID], how="left")
        .join(session_stats, on=SESSION, how="left")
        .join(graph_stats, on=[SESSION, AID], how="left")
        .with_columns(
            pl.col("local_interaction_count").fill_null(0).cast(pl.UInt16),
            pl.col("local_click_count").fill_null(0).cast(pl.UInt16),
            pl.col("local_cart_count").fill_null(0).cast(pl.UInt16),
            pl.col("local_order_count").fill_null(0).cast(pl.UInt16),
            pl.col("graph_weight_sum").fill_null(0.0).cast(pl.Float32),
            pl.col("graph_weight_max").fill_null(0.0).cast(pl.Float32),
            pl.col("graph_source_count").fill_null(0).cast(pl.UInt8),
            pl.col("graph_w_click_to_click").fill_null(0.0).cast(pl.Float32),
            pl.col("graph_w_cart_order_to_cart_order").fill_null(0.0).cast(pl.Float32),
            pl.col("graph_w_click_to_cart_order").fill_null(0.0).cast(pl.Float32),
            pl.col("dominant_graph_source").fill_null(0).cast(pl.Int8),
        )
        .with_columns(
            (pl.col("local_interaction_count") > 0).cast(pl.Int8).alias("is_repeated_item"),
            (pl.col("session_last_ts") - pl.col("item_last_ts")).cast(pl.Int64).alias("delta_t_nan"),
        )
        .with_columns(
            pl.col("delta_t_nan").fill_null(delta_t_fill).cast(pl.Int64).alias("delta_t_filled"),
        )
        .select(
            pl.col(SESSION),
            pl.col(AID),
            pl.col("candidate_score"),
            pl.col("candidate_rank"),
            pl.col("local_interaction_count"),
            pl.col("local_click_count"),
            pl.col("local_cart_count"),
            pl.col("local_order_count"),
            pl.col("is_repeated_item"),
            pl.col("session_last_ts"),
            pl.col("item_last_ts"),
            pl.col("delta_t_nan"),
            pl.col("delta_t_filled"),
            pl.col("graph_weight_sum"),
            pl.col("graph_weight_max"),
            pl.col("graph_source_count"),
            pl.col("graph_w_click_to_click"),
            pl.col("graph_w_cart_order_to_cart_order"),
            pl.col("graph_w_click_to_cart_order"),
            pl.col("graph_weight_sum").alias("graph_weight_total"),
            pl.col("dominant_graph_source"),
        )
        .sort([SESSION, "candidate_rank"])
    )

    del local_stats, session_stats, graph_stats
    gc.collect()
    return features


def local_item_stats_for_candidates(events_path: str | Path, candidates: pl.DataFrame) -> pl.DataFrame:
    """Scan train parquet and keep only current chunk's session-item records."""
    keys = candidates.select(SESSION, AID).unique()
    return (
        keys.lazy()
        .join(cast_events(pl.scan_parquet(events_path)), on=[SESSION, AID], how="left")
        .filter(pl.col(TS).is_not_null())
        .group_by([SESSION, AID])
        .agg(
            pl.len().cast(pl.UInt16).alias("local_interaction_count"),
            (pl.col(TYPE) == CLICK_TYPE).sum().cast(pl.UInt16).alias("local_click_count"),
            (pl.col(TYPE) == CART_TYPE).sum().cast(pl.UInt16).alias("local_cart_count"),
            (pl.col(TYPE) == ORDER_TYPE).sum().cast(pl.UInt16).alias("local_order_count"),
            pl.max(TS).cast(pl.Int64).alias("item_last_ts"),
        )
        .collect()
    )


def session_events_for_candidates(events_path: str | Path, candidates: pl.DataFrame) -> pl.DataFrame:
    """Scan train parquet and keep only sessions present in the current chunk."""
    sessions = candidates.select(SESSION).unique()
    return (
        sessions.lazy()
        .join(cast_events(pl.scan_parquet(events_path)), on=SESSION, how="left")
        .filter(pl.col(TS).is_not_null())
        .collect()
    )


def build_interaction_features_for_candidate_chunk(
    candidates: pl.DataFrame,
    events_path: str | Path,
    edge_frames: Iterable[pl.DataFrame],
    max_seed_items: int = 20,
    delta_t_fill: int = DEFAULT_DELTA_T_FILL,
) -> pl.DataFrame:
    """Build interaction features for one candidate chunk using lazy local scans."""
    candidates = candidates.select(
        pl.col(SESSION).cast(pl.Int32),
        pl.col(AID).cast(pl.Int32),
        pl.col(SCORE).cast(pl.Float32).alias("candidate_score"),
        pl.col(RANK).cast(pl.UInt32).alias("candidate_rank"),
    )
    local_stats = local_item_stats_for_candidates(events_path, candidates)
    session_events = session_events_for_candidates(events_path, candidates)
    session_stats = session_last_ts(session_events)
    graph_stats = graph_signal_features_for_candidates(
        candidates=candidates,
        session_events=session_events,
        edge_frames=edge_frames,
        max_seed_items=max_seed_items,
    )
    features = (
        candidates.join(local_stats, on=[SESSION, AID], how="left")
        .join(session_stats, on=SESSION, how="left")
        .join(graph_stats, on=[SESSION, AID], how="left")
        .with_columns(
            pl.col("local_interaction_count").fill_null(0).cast(pl.UInt16),
            pl.col("local_click_count").fill_null(0).cast(pl.UInt16),
            pl.col("local_cart_count").fill_null(0).cast(pl.UInt16),
            pl.col("local_order_count").fill_null(0).cast(pl.UInt16),
            pl.col("graph_weight_sum").fill_null(0.0).cast(pl.Float32),
            pl.col("graph_weight_max").fill_null(0.0).cast(pl.Float32),
            pl.col("graph_source_count").fill_null(0).cast(pl.UInt8),
            pl.col("graph_w_click_to_click").fill_null(0.0).cast(pl.Float32),
            pl.col("graph_w_cart_order_to_cart_order").fill_null(0.0).cast(pl.Float32),
            pl.col("graph_w_click_to_cart_order").fill_null(0.0).cast(pl.Float32),
            pl.col("dominant_graph_source").fill_null(0).cast(pl.Int8),
        )
        .with_columns(
            (pl.col("local_interaction_count") > 0).cast(pl.Int8).alias("is_repeated_item"),
            (pl.col("session_last_ts") - pl.col("item_last_ts")).cast(pl.Int64).alias("delta_t_nan"),
        )
        .with_columns(
            pl.col("delta_t_nan").fill_null(delta_t_fill).cast(pl.Int64).alias("delta_t_filled"),
            pl.col("graph_weight_sum").alias("graph_weight_total"),
        )
        .select(
            pl.col(SESSION),
            pl.col(AID),
            pl.col("candidate_score"),
            pl.col("candidate_rank"),
            pl.col("local_interaction_count"),
            pl.col("local_click_count"),
            pl.col("local_cart_count"),
            pl.col("local_order_count"),
            pl.col("is_repeated_item"),
            pl.col("session_last_ts"),
            pl.col("item_last_ts"),
            pl.col("delta_t_nan"),
            pl.col("delta_t_filled"),
            pl.col("graph_weight_sum"),
            pl.col("graph_weight_max"),
            pl.col("graph_source_count"),
            pl.col("graph_w_click_to_click"),
            pl.col("graph_w_cart_order_to_cart_order"),
            pl.col("graph_w_click_to_cart_order"),
            pl.col("graph_weight_total"),
            pl.col("dominant_graph_source"),
        )
        .sort([SESSION, "candidate_rank"])
    )
    del local_stats, session_events, session_stats, graph_stats
    gc.collect()
    return features


def write_interaction_features_candidate_chunks(
    candidates_path: str | Path,
    events_path: str | Path,
    graph_paths: Iterable[str | Path],
    output_dir: str | Path,
    n_candidate_chunks: int = 10,
    max_seed_items: int = 20,
    delta_t_fill: int = DEFAULT_DELTA_T_FILL,
) -> pl.DataFrame:
    """Split one candidate part into chunks and write each feature chunk immediately."""
    candidates_path = Path(candidates_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = pl.scan_parquet(candidates_path).select(pl.len().alias("n")).collect().item()
    chunk_size = max((total_rows + n_candidate_chunks - 1) // n_candidate_chunks, 1)
    edge_frames = [pl.read_parquet(path) for path in graph_paths]
    stats: list[dict[str, int | str]] = []

    for chunk_id in range(n_candidate_chunks):
        start = chunk_id * chunk_size
        if start >= total_rows:
            break
        end = min(start + chunk_size, total_rows)
        candidates = (
            pl.scan_parquet(candidates_path)
            .with_row_index("_row_nr")
            .filter((pl.col("_row_nr") >= start) & (pl.col("_row_nr") < end))
            .drop("_row_nr")
            .collect()
        )
        features = build_interaction_features_for_candidate_chunk(
            candidates=candidates,
            events_path=events_path,
            edge_frames=edge_frames,
            max_seed_items=max_seed_items,
            delta_t_fill=delta_t_fill,
        )
        output_path = output_dir / f"interaction_features_chunk_{chunk_id:03d}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.lazy().sink_parquet(output_path)
        stats.append(
            {
                "chunk": chunk_id,
                "row_start": start,
                "row_end": end,
                "rows": features.height,
                "path": str(output_path),
            }
        )
        del candidates, features
        gc.collect()

    del edge_frames
    gc.collect()
    return pl.DataFrame(stats)


def write_interaction_features_part(
    candidates_path: str | Path,
    events_path: str | Path,
    graph_paths: Iterable[str | Path],
    output_path: str | Path,
    n_buckets: int,
    bucket: int,
    max_seed_items: int = 20,
    delta_t_fill: int = DEFAULT_DELTA_T_FILL,
) -> int:
    """Build and write one bucket of interaction features."""
    candidates = pl.read_parquet(candidates_path)
    events = (
        pl.scan_parquet(events_path)
        .filter((pl.col(SESSION) % n_buckets) == bucket)
        .collect()
    )
    edge_frames = [pl.read_parquet(path) for path in graph_paths]
    features = build_interaction_features_for_bucket(
        candidates=candidates,
        events=events,
        edge_frames=edge_frames,
        max_seed_items=max_seed_items,
        delta_t_fill=delta_t_fill,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(output_path)
    rows = features.height
    del candidates, events, edge_frames, features
    gc.collect()
    return rows
