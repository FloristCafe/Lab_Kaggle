from __future__ import annotations

import math

import polars as pl

from otto_recommender.heuristic_covisitation import (
    AID_X,
    AID_Y,
    CLICK_TO_CART_ORDER,
    CLICK_TO_CLICK,
    CoVisitationRule,
    GRAPH,
    ITEM_FREQ,
    WEIGHT,
    attach_degree_penalty,
    build_rule_edges,
    item_frequency,
    popular_fallback_items,
    recommend_from_heuristic_graphs,
)


def toy_events_ms() -> pl.DataFrame:
    hour = 3_600_000
    return pl.DataFrame(
        {
            "session": [1, 1, 1, 1, 2, 2, 2],
            "aid": [10, 11, 12, 13, 10, 14, 15],
            "ts": [0, hour, 2 * hour, 3 * hour, 0, hour, 2 * hour],
            "type": [0, 0, 1, 2, 0, 1, 2],
        }
    )


def test_click_to_click_uses_time_decay() -> None:
    edges = build_rule_edges(toy_events_ms(), CLICK_TO_CLICK)

    assert {AID_X, AID_Y, WEIGHT, GRAPH}.issubset(edges.columns)
    assert edges.filter((pl.col(AID_X) == 10) & (pl.col(AID_Y) == 11)).height == 1
    assert edges.select(pl.col(WEIGHT).max()).item() < 1.0


def test_degree_penalty_reduces_hub_edges() -> None:
    events = toy_events_ms()
    penalized = build_rule_edges(events, CLICK_TO_CLICK)
    no_penalty_rule = CoVisitationRule(
        name=CLICK_TO_CLICK.name,
        source_types=CLICK_TO_CLICK.source_types,
        target_types=CLICK_TO_CLICK.target_types,
        max_time_delta_seconds=CLICK_TO_CLICK.max_time_delta_seconds,
        max_events_per_session=CLICK_TO_CLICK.max_events_per_session,
        topk_per_chunk=CLICK_TO_CLICK.topk_per_chunk,
        final_topk_per_source=CLICK_TO_CLICK.final_topk_per_source,
        decay_c_seconds=CLICK_TO_CLICK.decay_c_seconds,
        degree_alpha=0.0,
        graph_weight=CLICK_TO_CLICK.graph_weight,
    )
    unpenalized = build_rule_edges(events, no_penalty_rule)

    edge_filter = (pl.col(AID_X) == 10) & (pl.col(AID_Y) == 11)
    assert penalized.filter(edge_filter).select(WEIGHT).item() < unpenalized.filter(edge_filter).select(WEIGHT).item()


def test_item_frequency_reads_aid_counts() -> None:
    freq = item_frequency(toy_events_ms())

    assert {"aid", ITEM_FREQ}.issubset(freq.columns)
    assert freq.filter(pl.col("aid") == 10).select(ITEM_FREQ).item() == 2


def test_attach_degree_penalty_after_pair_generation() -> None:
    pairs = pl.DataFrame(
        {
            "session": [1],
            "aid_x": [10],
            "aid_y": [11],
            "ts_x": [0],
            "ts_y": [3_600_000],
            "type_x": [0],
            "type_y": [0],
            "dt_seconds": [3600.0],
        }
    )
    freq = pl.DataFrame({"aid": [10, 11], ITEM_FREQ: [4, 9]})
    joined = attach_degree_penalty(pairs, freq, CLICK_TO_CLICK)

    assert math.isclose(joined.select("_degree_penalty").item(), 1.0 / 6.0, rel_tol=1e-6)


def test_click_to_cart_order_filters_source_and_target_types() -> None:
    edges = build_rule_edges(toy_events_ms(), CLICK_TO_CART_ORDER)

    assert edges.filter((pl.col(AID_Y) == 11)).height == 0
    assert edges.filter(pl.col(AID_Y).is_in([12, 13, 14, 15])).height > 0


def test_recommend_from_three_graph_shape() -> None:
    events = toy_events_ms()
    edge_frames = [
        build_rule_edges(events, CLICK_TO_CLICK),
        build_rule_edges(events, CLICK_TO_CART_ORDER),
    ]
    recs = recommend_from_heuristic_graphs(events, edge_frames, topk=5)

    assert {"session", "aid", "score", "rank"}.issubset(recs.columns)
    assert recs.group_by("session").agg(pl.len().alias("n")).select(pl.col("n").max()).item() <= 5


def test_recommend_can_fill_with_popular_fallback() -> None:
    events = toy_events_ms()
    fallback = popular_fallback_items(events, topk=5)
    recs = recommend_from_heuristic_graphs(
        events,
        edge_frames=[build_rule_edges(events, CLICK_TO_CLICK)],
        topk=5,
        min_candidates=5,
        popular_fallback=fallback,
    )

    counts = recs.group_by("session").agg(pl.len().alias("n"))
    assert counts.select(pl.col("n").min()).item() == 5
