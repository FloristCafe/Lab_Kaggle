from __future__ import annotations

from pathlib import Path

import polars as pl

from otto_recommender.feature_engineering import build_item_features, build_user_features


def toy_events() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "session": [1, 1, 1, 2, 2, 3, 3, 3],
            "aid": [10, 10, 10, 11, 12, 12, 12, 12],
            "ts": [1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 8_000],
            "type": [0, 0, 0, 0, 0, 1, 1, 2],
        }
    )


def test_build_item_features(tmp_path: Path) -> None:
    input_path = tmp_path / "toy.parquet"
    output_path = tmp_path / "item_features.parquet"
    toy_events().write_parquet(input_path)

    build_item_features(input_path, output_path)
    df = pl.read_parquet(output_path)

    assert {"aid", "total_interactions", "click_count", "cart_count", "order_count", "recent_24h_interactions", "conversion_rate"}.issubset(df.columns)
    assert df.filter(pl.col("aid") == 10).select("total_interactions").item() == 3
    assert df.filter(pl.col("aid") == 11).select("conversion_rate").item() < 1.0


def test_build_user_features(tmp_path: Path) -> None:
    input_path = tmp_path / "toy.parquet"
    output_path = tmp_path / "user_features.parquet"
    toy_events().write_parquet(input_path)

    build_user_features(input_path, output_path)
    df = pl.read_parquet(output_path)

    assert {"session", "session_length", "unique_items", "duration", "cart_count", "order_count", "is_window_shopping"}.issubset(df.columns)
    assert df.filter(pl.col("session") == 1).select("session_length").item() == 3
    assert df.filter(pl.col("session") == 2).select("duration").item() == 1_000
