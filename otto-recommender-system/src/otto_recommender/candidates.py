"""Candidate generation baselines for OTTO."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

import pandas as pd

from .schema import AID, EVENT_TYPE_WEIGHTS, SESSION, TS, TYPE


def _dedupe_keep_order(items: Iterable[int], limit: int) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for item in items:
        item = int(item)
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
        if len(result) >= limit:
            break
    return result


def popular_items(events: pd.DataFrame, topk: int = 100) -> list[int]:
    """Return globally popular aids, weighted by OTTO event type."""
    weighted = events[[AID, TYPE]].copy()
    weighted["weight"] = weighted[TYPE].map(EVENT_TYPE_WEIGHTS).fillna(1.0)
    scores = weighted.groupby(AID, sort=False)["weight"].sum().sort_values(ascending=False)
    return [int(aid) for aid in scores.head(topk).index]


def recent_items_by_session(events: pd.DataFrame, topk: int = 20) -> dict[int, list[int]]:
    """Return each session's most recent unique aids."""
    ordered = events.sort_values([SESSION, TS], ascending=[True, False])
    result: dict[int, list[int]] = {}
    for session, group in ordered.groupby(SESSION, sort=False):
        result[int(session)] = _dedupe_keep_order(group[AID].tolist(), topk)
    return result


def co_visitation_map(
    events: pd.DataFrame,
    max_events_per_session: int = 30,
    topk: int = 40,
) -> dict[int, list[int]]:
    """Build a simple item-to-item co-visitation map from local events."""
    pairs: dict[int, Counter[int]] = defaultdict(Counter)
    ordered = events.sort_values([SESSION, TS])
    for _, group in ordered.groupby(SESSION, sort=False):
        aids = group[AID].tail(max_events_per_session).astype(int).tolist()
        unique_aids = _dedupe_keep_order(reversed(aids), max_events_per_session)
        for i, aid in enumerate(unique_aids):
            for other in unique_aids[i + 1 :]:
                if aid == other:
                    continue
                pairs[aid][other] += 1
                pairs[other][aid] += 1
    return {
        aid: [other for other, _ in counter.most_common(topk)]
        for aid, counter in pairs.items()
    }


def candidates_from_recent_and_covisitation(
    events: pd.DataFrame,
    session_ids: Iterable[int] | None = None,
    final_topk: int = 20,
    covisitation_topk: int = 40,
) -> dict[int, list[int]]:
    """Generate candidates by mixing recent session items, co-visitation, and popularity."""
    recent = recent_items_by_session(events, topk=final_topk)
    covisitation = co_visitation_map(events, topk=covisitation_topk)
    fallback = popular_items(events, topk=final_topk)
    if session_ids is None:
        session_ids = recent.keys()

    candidates: dict[int, list[int]] = {}
    for session in session_ids:
        session = int(session)
        seed_items = recent.get(session, [])
        expanded: list[int] = []
        for aid in seed_items:
            expanded.extend(covisitation.get(aid, []))
        candidates[session] = _dedupe_keep_order([*seed_items, *expanded, *fallback], final_topk)
    return candidates

