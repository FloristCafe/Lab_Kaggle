"""Validation metrics for OTTO-style predictions."""

from __future__ import annotations

from collections.abc import Iterable
import math

import pandas as pd

from .schema import LABELS, SESSION, TYPE


def normalize_labels(value: str | Iterable[int]) -> list[int]:
    """Convert a Kaggle label string or iterable into a list of item ids."""
    if isinstance(value, str):
        if not value.strip():
            return []
        return [int(item) for item in value.split()]
    return [int(item) for item in value]


def recall_at_k(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    k: int = 20,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted Recall@K for rows with session, type, and labels."""
    weights = weights or {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
    required = {SESSION, TYPE, "ground_truth"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"labels missing columns: {sorted(missing)}")
    pred_required = {SESSION, TYPE, LABELS}
    pred_missing = pred_required - set(predictions.columns)
    if pred_missing:
        raise ValueError(f"predictions missing columns: {sorted(pred_missing)}")

    pred_lookup = {
        (int(row[SESSION]), row[TYPE]): normalize_labels(row[LABELS])[:k]
        for _, row in predictions.iterrows()
    }

    total_weight = 0.0
    weighted_recall = 0.0
    for target_type, group in labels.groupby(TYPE):
        hits = 0
        total = 0
        for _, row in group.iterrows():
            truth = set(normalize_labels(row["ground_truth"]))
            if not truth:
                continue
            pred = set(pred_lookup.get((int(row[SESSION]), target_type), []))
            hits += len(truth & pred)
            total += min(len(truth), k)
        if total == 0:
            continue
        weight = weights.get(target_type, 0.0)
        weighted_recall += weight * hits / total
        total_weight += weight
    return weighted_recall / total_weight if total_weight else 0.0


def binary_dcg_at_ranks(ranks: Iterable[int], k: int = 20) -> float:
    """Compute binary DCG from 1-based hit ranks."""
    return sum(1.0 / math.log2(int(rank) + 1.0) for rank in ranks if 1 <= int(rank) <= k)


def binary_idcg_at_k(n_relevant: int, k: int = 20) -> float:
    """Best possible binary DCG for n relevant items."""
    return sum(1.0 / math.log2(rank + 1.0) for rank in range(1, min(int(n_relevant), k) + 1))


def ndcg_at_k_from_ranked_hits(
    labels: pd.DataFrame,
    ranked_hits: pd.DataFrame,
    k: int = 20,
) -> dict[str, float | int]:
    """Compute strict binary NDCG@K from labels and hit ranks.

    `labels` must contain `session` and `ground_truth`; `ranked_hits` must contain
    `session` and 1-based `rank` for predicted items that hit the ground truth.
    Sessions with labels but no hit contribute zero DCG.
    """
    if labels.empty:
        return {
            "label_sessions": 0,
            "hit_sessions": 0,
            "dcg": 0.0,
            "idcg": 0.0,
            "ndcg_at_k": 0.0,
        }
    label_sizes = {
        int(row[SESSION]): len(normalize_labels(row["ground_truth"]))
        for _, row in labels.iterrows()
    }
    idcg = sum(binary_idcg_at_k(size, k=k) for size in label_sizes.values())
    if ranked_hits.empty:
        dcg = 0.0
        hit_sessions = 0
    else:
        hits_by_session = ranked_hits.groupby(SESSION)["rank"].apply(list)
        dcg = sum(binary_dcg_at_ranks(ranks, k=k) for ranks in hits_by_session)
        hit_sessions = int(len(hits_by_session))
    return {
        "label_sessions": int(len(label_sizes)),
        "hit_sessions": hit_sessions,
        "dcg": float(dcg),
        "idcg": float(idcg),
        "ndcg_at_k": float(dcg / idcg) if idcg > 0 else 0.0,
    }
