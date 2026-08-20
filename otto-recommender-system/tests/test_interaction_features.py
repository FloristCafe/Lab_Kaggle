from __future__ import annotations

import polars as pl

from otto_recommender.interaction_features import build_interaction_features_for_bucket


def test_interaction_features_nan_and_filled_delta() -> None:
    candidates = pl.DataFrame(
        {
            "session": [1, 1],
            "aid": [10, 20],
            "score": [1.0, 0.5],
            "rank": [1, 2],
        }
    )
    events = pl.DataFrame(
        {
            "session": [1, 1],
            "aid": [10, 11],
            "ts": [1_000, 5_000],
            "type": [0, 1],
        }
    )
    edges = pl.DataFrame(
        {
            "aid_x": [11],
            "aid_y": [20],
            "graph": ["click_to_cart_order"],
            "weight": [0.75],
        }
    )

    features = build_interaction_features_for_bucket(
        candidates=candidates,
        events=events,
        edge_frames=[edges],
        delta_t_fill=9_999_999,
    )

    repeated = features.filter(pl.col("aid") == 10)
    new_item = features.filter(pl.col("aid") == 20)

    assert repeated.select("is_repeated_item").item() == 1
    assert repeated.select("local_click_count").item() == 1
    assert repeated.select("delta_t_nan").item() == 4_000
    assert new_item.select("is_repeated_item").item() == 0
    assert new_item.select("delta_t_nan").item() is None
    assert new_item.select("delta_t_filled").item() == 9_999_999
    assert new_item.select("graph_w_click_to_cart_order").item() == 0.75
    assert new_item.select("graph_weight_total").item() == 0.75
    assert new_item.select("dominant_graph_source").item() == 3
