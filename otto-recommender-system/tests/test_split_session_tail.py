from __future__ import annotations

import polars as pl

from otto_recommender.schema import AID, SESSION, TS, TYPE
from scripts.split_session_tail import split_validation_window


def test_split_validation_window_drops_short_sessions_and_keeps_both_sides() -> None:
    events = pl.DataFrame(
        {
            SESSION: [1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5],
            AID: list(range(14)),
            TS: [0, 1, 2, 3, 4, 5, 6, 7, 10, 10, 11, 12, 20, 20],
            TYPE: [0] * 14,
        }
    )

    prefix, suffix = split_validation_window(events.lazy(), prefix_fraction=0.8)
    prefix, suffix = prefix.collect(), suffix.collect()

    assert prefix.group_by(SESSION).len().sort(SESSION).rows() == [(2, 1), (3, 4), (4, 3)]
    assert suffix.group_by(SESSION).len().sort(SESSION).rows() == [(2, 1), (3, 1), (4, 1)]
    boundaries = prefix.group_by(SESSION).agg(pl.max(TS).alias("prefix_max"))
    suffix_boundaries = suffix.group_by(SESSION).agg(pl.min(TS).alias("suffix_min"))
    assert boundaries.join(suffix_boundaries, on=SESSION).filter(pl.col("prefix_max") >= pl.col("suffix_min")).is_empty()
