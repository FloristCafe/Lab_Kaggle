"""Vectorized ranking metrics for large candidate tables."""

from __future__ import annotations

import polars as pl


def ndcg_at_k_polars(
    candidates: pl.LazyFrame | pl.DataFrame,
    targets: pl.LazyFrame | pl.DataFrame,
    score_col: str = "score",
    k: int = 20,
    target_col: str = "target_order",
) -> pl.DataFrame:
    """Return per-session binary NDCG using Polars windows and aggregations only."""
    cand = candidates.lazy() if isinstance(candidates, pl.DataFrame) else candidates
    truth = targets.lazy() if isinstance(targets, pl.DataFrame) else targets
    ranked = (
        cand.with_columns(
            pl.col(score_col).rank(method="ordinal", descending=True).over("session").alias("rank")
        )
        .filter(pl.col("rank") <= k)
        .select("session", "aid", "rank")
    )
    relevant = truth.filter(pl.col(target_col) == 1).select("session", "aid")
    session_sizes = relevant.group_by("session").agg(pl.len().alias("n_relevant"))
    discounts = pl.DataFrame({"rank": list(range(1, k + 1))}).lazy().with_columns(
        (1.0 / (pl.col("rank").cast(pl.Float64) + 1).log(base=2)).alias("discount")
    )
    dcg = (
        ranked.join(relevant, on=["session", "aid"], how="inner")
        .with_columns((1.0 / (pl.col("rank").cast(pl.Float64) + 1).log(base=2)).alias("gain"))
        .group_by("session").agg(pl.sum("gain").alias("dcg"))
    )
    idcg = (
        session_sizes.with_columns(pl.col("n_relevant").clip(upper_bound=k).alias("n"))
        .join(discounts, how="cross")
        .filter(pl.col("rank") <= pl.col("n"))
        .group_by("session").agg(pl.sum("discount").alias("idcg"))
    )
    return (
        session_sizes.select("session")
        .join(dcg, on="session", how="left")
        .join(idcg, on="session", how="left")
        .with_columns(pl.col("dcg").fill_null(0.0))
        .with_columns((pl.col("dcg") / pl.col("idcg")).alias("ndcg"))
    )


def mean_ndcg_at_k_polars(
    candidates: pl.LazyFrame | pl.DataFrame,
    targets: pl.LazyFrame | pl.DataFrame,
    score_col: str = "score",
    k: int = 20,
    target_col: str = "target_order",
) -> float:
    """Compute the mean session NDCG; sessions with no hit contribute zero."""
    result = ndcg_at_k_polars(candidates, targets, score_col=score_col, k=k, target_col=target_col).select(pl.mean("ndcg"))
    return float(result.collect().item())
