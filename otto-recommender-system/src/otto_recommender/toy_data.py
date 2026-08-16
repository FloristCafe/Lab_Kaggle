"""Small deterministic data for local pipeline checks."""

from __future__ import annotations

import pandas as pd

from .schema import AID, SESSION, TS, TYPE


def make_toy_events() -> pd.DataFrame:
    """Return a tiny OTTO-like event table that needs no Kaggle data."""
    rows = [
        (1, 101, 1_660_000_000_000, "clicks"),
        (1, 102, 1_660_000_010_000, "clicks"),
        (1, 103, 1_660_000_020_000, "carts"),
        (1, 104, 1_660_000_030_000, "orders"),
        (2, 101, 1_660_000_000_000, "clicks"),
        (2, 105, 1_660_000_010_000, "clicks"),
        (2, 103, 1_660_000_020_000, "carts"),
        (3, 106, 1_660_000_000_000, "clicks"),
        (3, 101, 1_660_000_010_000, "clicks"),
        (3, 104, 1_660_000_020_000, "orders"),
        (4, 107, 1_660_000_000_000, "clicks"),
        (4, 102, 1_660_000_010_000, "clicks"),
        (4, 103, 1_660_000_020_000, "carts"),
    ]
    return pd.DataFrame(rows, columns=[SESSION, AID, TS, TYPE])


def make_toy_labels() -> pd.DataFrame:
    """Return tiny labels in the same shape as local validation labels."""
    return pd.DataFrame(
        [
            (1, "clicks", [104]),
            (2, "carts", [103]),
            (3, "orders", [104]),
        ],
        columns=[SESSION, TYPE, "ground_truth"],
    )

