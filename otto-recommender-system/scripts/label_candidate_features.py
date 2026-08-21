from __future__ import annotations

import argparse
import gc
from pathlib import Path

import polars as pl

from otto_recommender.schema import AID, SESSION

TARGET_COLUMNS = ("target_click", "target_cart", "target_order")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Left-join strict time labels onto candidate feature parts.")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--targets", default="data/processed/time_split/label_targets.parquet")
    parser.add_argument("--output-dir", default="artifacts/ranker/time_split/labeled_features_parts")
    parser.add_argument("--stats-output", default="artifacts/reports/time_split_labeled_features_stats.csv")
    parser.add_argument("--pattern", default="*.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(features_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No feature parts matched {features_dir / args.pattern}")

    targets = pl.scan_parquet(args.targets).select(
        pl.col(SESSION).cast(pl.Int32),
        pl.col(AID).cast(pl.Int32),
        *[pl.col(col).cast(pl.Int8) for col in TARGET_COLUMNS],
    )
    stats: list[dict[str, int | str]] = []
    for path in paths:
        output_path = output_dir / path.name
        labeled = (
            pl.scan_parquet(path)
            .join(targets, on=[SESSION, AID], how="left")
            .with_columns([pl.col(col).fill_null(0).cast(pl.Int8) for col in TARGET_COLUMNS])
        )
        labeled.sink_parquet(output_path)
        part_stats = pl.scan_parquet(output_path).select(
            pl.len().alias("rows"),
            *[pl.sum(col).alias(col) for col in TARGET_COLUMNS],
        ).collect()
        stats.append(
            {
                "part": path.name,
                "rows": int(part_stats.select("rows").item()),
                "target_click": int(part_stats.select("target_click").item()),
                "target_cart": int(part_stats.select("target_cart").item()),
                "target_order": int(part_stats.select("target_order").item()),
                "path": str(output_path),
            }
        )
        print(
            f"{path.name}: rows={stats[-1]['rows']:,} "
            f"click={stats[-1]['target_click']:,} cart={stats[-1]['target_cart']:,} "
            f"order={stats[-1]['target_order']:,} -> {output_path}"
        )
        del labeled, part_stats
        gc.collect()

    stats_frame = pl.DataFrame(stats)
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_frame.write_csv(stats_path)
    total = stats_frame.select(
        pl.sum("rows").alias("rows"),
        pl.sum("target_click").alias("target_click"),
        pl.sum("target_cart").alias("target_cart"),
        pl.sum("target_order").alias("target_order"),
    )
    print(f"stats -> {stats_path}")
    print(total)


if __name__ == "__main__":
    main()
