"""Validation metrics for OTTO-style predictions."""

from __future__ import annotations

from collections.abc import Iterable

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

