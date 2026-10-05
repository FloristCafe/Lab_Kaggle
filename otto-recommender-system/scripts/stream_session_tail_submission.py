"""Stream LightGBM ranking over session-disjoint candidate Parquet parts.

The candidate files keep raw OTTO aids.  The SASRec vocabulary shift is internal
to its embedding; this script subtracts ``aid_offset`` only when formatting the
final Kaggle labels.
"""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import lightgbm as lgb
import polars as pl


EVENT_MODELS = ("clicks", "carts", "orders")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream three LightGBM rankers into a Kaggle submission.")
    parser.add_argument("--candidates-dir", required=True)
    parser.add_argument("--click-model", required=True)
    parser.add_argument("--cart-model", required=True)
    parser.add_argument("--order-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pattern", default="*.parquet")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--aid-offset", type=int, default=1)
    return parser.parse_args()


def score_part(frame: pl.DataFrame, model: lgb.Booster, topk: int) -> pl.DataFrame:
    feature_names = model.feature_name()
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        # Compatibility for checkpoints trained before feature pruning.
        frame = frame.with_columns(*(pl.lit(0.0).alias(name) for name in missing))
    scores = model.predict(frame.select(feature_names).to_pandas())
    return (
        frame.select("session", "aid")
        .with_columns(pl.Series("prediction", scores).cast(pl.Float32))
        .with_columns(
            pl.col("prediction")
            .rank(method="ordinal", descending=True)
            .over("session")
            .alias("prediction_rank")
        )
        .filter(pl.col("prediction_rank") <= topk)
        .sort(["session", "prediction_rank"])
    )


def labels_by_session(ranked: pl.DataFrame, aid_offset: int) -> pl.DataFrame:
    return ranked.group_by("session", maintain_order=True).agg(
        (pl.col("aid") - aid_offset).cast(pl.Utf8).str.join(" ").alias("labels")
    )


def main() -> None:
    args = parse_args()
    paths = sorted(Path(args.candidates_dir).glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No candidate parts matched {args.candidates_dir}/{args.pattern}")
    if args.topk < 1 or args.aid_offset < 0:
        raise ValueError("topk must be positive and aid_offset must be non-negative")

    models = {
        "clicks": lgb.Booster(model_file=args.click_model),
        "carts": lgb.Booster(model_file=args.cart_model),
        "orders": lgb.Booster(model_file=args.order_model),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["session_type", "labels"])
        for path in paths:
            frame = pl.read_parquet(path)
            session_count = frame.select(pl.col("session").n_unique()).item()
            for event_type in EVENT_MODELS:
                ranked = score_part(frame, models[event_type], args.topk)
                grouped = labels_by_session(ranked, args.aid_offset)
                writer.writerows(
                    (f"{int(session)}_{event_type}", labels)
                    for session, labels in grouped.iter_rows()
                )
                del ranked, grouped
                gc.collect()
            print(f"ranked {path.name}: sessions={session_count:,}", flush=True)
            del frame
            gc.collect()
    print(f"wrote streaming submission to {output}")


if __name__ == "__main__":
    main()
