from __future__ import annotations

import argparse
import gc
from pathlib import Path

import polars as pl

from otto_recommender.schema import AID, SESSION

HASH_DENOMINATOR = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downsample negative ranker rows while preserving positives.")
    parser.add_argument("--input-dir", default="artifacts/ranker/time_split/labeled_features_parts")
    parser.add_argument("--output-dir", default="artifacts/ranker/time_split/click_downsampled_1to20_parts")
    parser.add_argument("--stats-output", default="artifacts/reports/time_split_click_downsample_stats.csv")
    parser.add_argument("--target-col", default="target_click")
    parser.add_argument("--neg-per-pos", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pattern", default="*.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(input_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No labeled parts matched {input_dir / args.pattern}")

    global_stats = (
        pl.scan_parquet([str(path) for path in paths])
        .select(
            pl.len().alias("rows"),
            pl.sum(args.target_col).alias("positive_rows"),
        )
        .with_columns((pl.col("rows") - pl.col("positive_rows")).alias("negative_rows"))
        .collect()
    )
    positives = int(global_stats.select("positive_rows").item())
    negatives = int(global_stats.select("negative_rows").item())
    if positives <= 0:
        raise ValueError(f"No positives found for {args.target_col}")
    sample_fraction = min(1.0, args.neg_per_pos * positives / max(negatives, 1))
    hash_threshold = int(sample_fraction * HASH_DENOMINATOR)

    stats: list[dict[str, int | float | str]] = []
    for path in paths:
        output_path = output_dir / path.name
        sampled = (
            pl.scan_parquet(path)
            .with_columns(
                (
                    pl.struct([pl.col(SESSION), pl.col(AID)])
                    .hash(seed=args.seed)
                    .mod(HASH_DENOMINATOR)
                ).alias("_sample_hash")
            )
            .filter((pl.col(args.target_col) == 1) | (pl.col("_sample_hash") < hash_threshold))
            .drop("_sample_hash")
        )
        sampled.sink_parquet(output_path)
        part_stats = pl.scan_parquet(output_path).select(
            pl.len().alias("rows"),
            pl.sum(args.target_col).alias("positive_rows"),
        ).with_columns((pl.col("rows") - pl.col("positive_rows")).alias("negative_rows")).collect()
        stats.append(
            {
                "part": path.name,
                "rows": int(part_stats.select("rows").item()),
                "positive_rows": int(part_stats.select("positive_rows").item()),
                "negative_rows": int(part_stats.select("negative_rows").item()),
                "sample_fraction": sample_fraction,
                "path": str(output_path),
            }
        )
        print(
            f"{path.name}: rows={stats[-1]['rows']:,} pos={stats[-1]['positive_rows']:,} "
            f"neg={stats[-1]['negative_rows']:,} -> {output_path}"
        )
        del sampled, part_stats
        gc.collect()

    stats_frame = pl.DataFrame(stats)
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_frame.write_csv(stats_path)
    selected = stats_frame.select(
        pl.sum("rows").alias("rows"),
        pl.sum("positive_rows").alias("positive_rows"),
        pl.sum("negative_rows").alias("negative_rows"),
    ).with_columns(
        (pl.col("negative_rows") / pl.col("positive_rows")).alias("neg_pos_ratio"),
        pl.lit(sample_fraction).alias("negative_sample_fraction"),
    )
    print(f"stats -> {stats_path}")
    print(selected)


if __name__ == "__main__":
    main()
