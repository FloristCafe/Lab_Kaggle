from __future__ import annotations

from pathlib import Path

import polars as pl

from otto_recommender.hard_negative_sampling import HardNegativeConfig, sampling_plan, write_hard_negative_parts


def test_hard_negative_sampling_preserves_positives_and_buckets(tmp_path: Path) -> None:
    input_path = tmp_path / "features.parquet"
    output_dir = tmp_path / "sampled"
    stats_path = tmp_path / "stats.csv"
    pl.DataFrame(
        {
            "session": [1, 1, 1, 2, 2, 2, 3, 3],
            "aid": [10, 11, 12, 10, 13, 14, 15, 16],
            "local_click_count": [0, 3, 1, 2, 0, 1, 4, 0],
            "graph_w_cart_order_to_cart_order": [0.0, 0.1, 0.9, 0.0, 0.8, 0.0, 0.2, 0.0],
            "target_order": [1, 0, 0, 1, 0, 0, 0, 0],
        }
    ).write_parquet(input_path)

    stats = write_hard_negative_parts(
        input_paths=[input_path],
        output_dir=output_dir,
        stats_output=stats_path,
        config=HardNegativeConfig(
            target_col="target_order",
            graph_col="graph_w_cart_order_to_cart_order",
            hard_click_neg_per_pos=1,
            hard_graph_neg_per_pos=1,
            random_neg_per_pos=1,
            min_local_clicks=2,
            graph_quantile=0.5,
        ),
    )

    sampled = pl.read_parquet(output_dir / input_path.name)
    assert sampled.filter(pl.col("target_order") == 1).height == 2
    assert {"positive", "hard_click", "hard_graph", "random"}.intersection(
        set(sampled.select("negative_bucket").to_series().to_list())
    )
    assert stats.filter(pl.col("part") == "__TOTAL__").select("positive_rows").item() == 2


def test_sampling_plan_falls_back_to_random_when_hard_click_is_exhausted() -> None:
    counts = pl.DataFrame(
        {
            "negative_bucket": ["positive", "hard_click", "hard_graph", "random"],
            "available_rows": [10, 2, 100, 1_000],
        }
    )
    _, plan = sampling_plan(
        counts,
        HardNegativeConfig(
            target_col="target_order",
            graph_col="graph_w_cart_order_to_cart_order",
            hard_click_neg_per_pos=8,
            hard_graph_neg_per_pos=10,
            random_neg_per_pos=2,
        ),
    )

    hard_click = plan.filter(pl.col("bucket") == "hard_click")
    random = plan.filter(pl.col("bucket") == "random")
    all_negatives = plan.filter(pl.col("bucket") == "__ALL_NEGATIVES__")

    assert hard_click.select("requested_rows").item() == 80
    assert hard_click.select("planned_rows").item() == 2
    assert random.select("requested_rows").item() == 20
    assert random.select("planned_rows").item() == 98
    assert all_negatives.select("direct_shortfall_rows").item() == 0
