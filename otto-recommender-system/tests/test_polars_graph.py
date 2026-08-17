from __future__ import annotations

import polars as pl

from otto_recommender.polars_graph import (
    AID_X,
    AID_Y,
    WEIGHT,
    build_covisitation_edges,
    evaluate_recall_at_k,
    recommend_from_edges,
    split_train_valid_tail,
)


def toy_numeric_events() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "session": [1, 1, 1, 2, 2, 2],
            "aid": [10, 11, 12, 10, 13, 12],
            "ts": [1, 2, 3, 1, 2, 3],
            "type": [0, 0, 1, 0, 0, 2],
        }
    )


def test_split_train_valid_tail() -> None:
    train, labels = split_train_valid_tail(toy_numeric_events())

    assert train.height == 4
    assert labels.height == 2
    assert labels.select(pl.col("ground_truth").list.len().sum()).item() == 2


def test_build_edges_and_recommend() -> None:
    train, labels = split_train_valid_tail(toy_numeric_events())
    edges = build_covisitation_edges(train, max_events_per_session=10)

    assert {AID_X, AID_Y, WEIGHT}.issubset(edges.columns)
    assert edges.height > 0

    recs = recommend_from_edges(train, edges, topk=3)
    metrics = evaluate_recall_at_k(labels, recs, k=3)

    assert recs.height > 0
    assert metrics.select("weighted_recall").item(0, 0) >= 0.0
