"""Submission formatting helpers."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .schema import EVENT_TYPES, LABELS, SESSION, SESSION_TYPE, TYPE


def candidates_to_submission(
    candidates: dict[int, list[int]],
    event_types: Iterable[str] = EVENT_TYPES,
) -> pd.DataFrame:
    """Format per-session candidates as Kaggle's session_type/labels rows."""
    rows: list[tuple[str, str]] = []
    for session, aids in sorted(candidates.items()):
        label_string = " ".join(str(int(aid)) for aid in aids)
        for event_type in event_types:
            rows.append((f"{int(session)}_{event_type}", label_string))
    return pd.DataFrame(rows, columns=[SESSION_TYPE, LABELS])


def predictions_for_metric(
    candidates: dict[int, list[int]],
    event_types: Iterable[str] = EVENT_TYPES,
) -> pd.DataFrame:
    """Format candidates as rows with session/type/labels for local metrics."""
    rows: list[tuple[int, str, list[int]]] = []
    for session, aids in sorted(candidates.items()):
        for event_type in event_types:
            rows.append((int(session), event_type, aids))
    return pd.DataFrame(rows, columns=[SESSION, TYPE, LABELS])

