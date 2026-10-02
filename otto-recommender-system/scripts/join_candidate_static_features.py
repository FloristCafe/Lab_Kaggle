from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import polars as pl

from otto_recommender.schema import AID, SESSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lazily join item and session features to candidate parts.")
    parser.add_argument("--candidates-dir", required=True)
    parser.add_argument("--item-features", required=True)
    parser.add_argument("--user-features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--pattern", default="candidates_part_*.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates_dir = Path(args.candidates_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(candidates_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No candidates matched {candidates_dir / args.pattern}")

    item_features = pl.scan_parquet(args.item_features)
    user_features = pl.scan_parquet(args.user_features).rename(
        {
            "session_length": "session_length",
            "unique_items": "session_unique_items",
            "first_ts": "session_first_ts",
            "last_ts": "session_last_ts",
            "duration": "session_duration",
            "cart_count": "session_cart_count",
            "order_count": "session_order_count",
            "is_window_shopping": "session_is_window_shopping",
        }
    )
    rows: list[dict[str, int | float | str]] = []
    for path in paths:
        started = time.perf_counter()
        output_path = output_dir / path.name
        joined = (
            pl.scan_parquet(path)
            .join(item_features, on=AID, how="left")
            .join(user_features, on=SESSION, how="left")
        )
        joined.sink_parquet(output_path)
        stats = pl.scan_parquet(output_path).select(
            pl.len().alias("rows"),
            pl.col("total_interactions").is_null().sum().alias("missing_item_features"),
            pl.col("session_length").is_null().sum().alias("missing_user_features"),
        ).collect()
        row = {
            "part": path.name,
            "rows": int(stats["rows"][0]),
            "missing_item_features": int(stats["missing_item_features"][0]),
            "missing_user_features": int(stats["missing_user_features"][0]),
            "elapsed_seconds": time.perf_counter() - started,
            "path": str(output_path),
        }
        rows.append(row)
        print(
            f"{path.name}: rows={row['rows']:,} missing_item={row['missing_item_features']:,} "
            f"missing_user={row['missing_user_features']:,} elapsed={row['elapsed_seconds']:.2f}s"
        )
        del joined, stats
        gc.collect()

    report = pl.DataFrame(rows)
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    report.write_csv(stats_path)
    print("totals:")
    print(report.select(
        pl.sum("rows").alias("rows"),
        pl.sum("missing_item_features").alias("missing_item_features"),
        pl.sum("missing_user_features").alias("missing_user_features"),
        pl.sum("elapsed_seconds").alias("chunk_elapsed_seconds"),
    ))
    print(f"stats -> {stats_path}")


if __name__ == "__main__":
    main()
