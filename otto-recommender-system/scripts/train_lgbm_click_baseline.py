from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from otto_recommender.schema import AID, SESSION

BASE_FEATURES = [
    "candidate_score",
    "candidate_rank",
    "local_interaction_count",
    "local_click_count",
    "local_cart_count",
    "local_order_count",
    "is_repeated_item",
    "graph_weight_sum",
    "graph_weight_max",
    "graph_source_count",
    "graph_w_click_to_click",
    "graph_w_cart_order_to_cart_order",
    "graph_w_click_to_cart_order",
    "graph_weight_total",
    "dominant_graph_source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM click baseline and compare delta_t missing policies.")
    parser.add_argument("--train-dir", default="artifacts/ranker/time_split/click_downsampled_1to20_parts")
    parser.add_argument("--eval-dir", default="artifacts/ranker/time_split/labeled_features_parts")
    parser.add_argument("--output-dir", default="artifacts/models/time_split_click_lgbm_baseline")
    parser.add_argument("--reports-dir", default="artifacts/reports")
    parser.add_argument("--target-col", default="target_click")
    parser.add_argument("--eval-session-mod", type=int, default=10)
    parser.add_argument("--eval-session-rem", type=int, default=0)
    parser.add_argument("--num-boost-round", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pattern", default="*.parquet")
    return parser.parse_args()


def part_paths(directory: str | Path, pattern: str) -> list[Path]:
    paths = sorted(Path(directory).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No parquet parts matched {Path(directory) / pattern}")
    return paths


def collect_train_frame(
    paths: list[Path],
    target_col: str,
    delta_col: str,
    eval_session_mod: int,
    eval_session_rem: int,
    max_train_rows: int,
    seed: int,
) -> pd.DataFrame:
    columns = [SESSION, target_col, *BASE_FEATURES, delta_col]
    lazy = (
        pl.scan_parquet([str(path) for path in paths])
        .filter((pl.col(SESSION) % eval_session_mod) != eval_session_rem)
        .select(columns)
        .rename({delta_col: "delta_t"})
    )
    if max_train_rows > 0:
        total = lazy.select(pl.len()).collect().item()
        fraction = min(1.0, max_train_rows / max(total, 1))
        threshold = int(fraction * 1_000_000)
        lazy = lazy.with_columns(
            pl.struct([pl.col(SESSION)]).hash(seed=seed).mod(1_000_000).alias("_train_hash")
        ).filter(pl.col("_train_hash") < threshold).drop("_train_hash")
    frame = lazy.drop(SESSION).collect().to_pandas()
    return frame


def train_one_variant(
    train_frame: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> lgb.Booster:
    y = train_frame[target_col].astype(np.int8)
    x = train_frame[feature_cols]
    dataset = lgb.Dataset(x, label=y, free_raw_data=True)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "min_data_in_leaf": 100,
        "seed": args.seed,
        "verbosity": -1,
    }
    return lgb.train(params=params, train_set=dataset, num_boost_round=args.num_boost_round)


def evaluate_recall_at_20(
    model: lgb.Booster,
    paths: list[Path],
    target_col: str,
    delta_col: str,
    eval_session_mod: int,
    eval_session_rem: int,
) -> dict[str, float | int]:
    feature_cols = [*BASE_FEATURES, "delta_t"]
    total_positive = 0
    hit_positive = 0
    eval_rows = 0
    eval_sessions: set[int] = set()
    top_rows = 0

    for path in paths:
        frame = (
            pl.scan_parquet(path)
            .filter((pl.col(SESSION) % eval_session_mod) == eval_session_rem)
            .select(SESSION, AID, target_col, *BASE_FEATURES, delta_col)
            .rename({delta_col: "delta_t"})
            .collect()
        )
        if frame.is_empty():
            continue
        pdf = frame.select(feature_cols).to_pandas()
        scores = model.predict(pdf)
        scored = frame.select(SESSION, AID, target_col).with_columns(
            pl.Series("pred_score", scores).cast(pl.Float32)
        )
        total_positive += int(scored.select(pl.sum(target_col)).item())
        eval_rows += scored.height
        eval_sessions.update(scored.select(SESSION).unique().to_series().to_list())
        topk = (
            scored.with_columns(
                pl.col("pred_score")
                .rank(method="ordinal", descending=True)
                .over(SESSION)
                .cast(pl.UInt32)
                .alias("_rank")
            )
            .filter(pl.col("_rank") <= 20)
        )
        top_rows += topk.height
        hit_positive += int(topk.select(pl.sum(target_col)).item())
        del frame, pdf, scored, topk
        gc.collect()

    recall = hit_positive / total_positive if total_positive else 0.0
    return {
        "eval_rows": eval_rows,
        "eval_sessions": len(eval_sessions),
        "eval_positive_rows": total_positive,
        "top20_rows": top_rows,
        "top20_hits": hit_positive,
        "recall_at_20": recall,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_paths = part_paths(args.train_dir, args.pattern)
    eval_paths = part_paths(args.eval_dir, args.pattern)
    feature_cols = [*BASE_FEATURES, "delta_t"]
    variants = {
        "delta_t_nan": "delta_t_nan",
        "delta_t_filled": "delta_t_filled",
    }

    metrics: list[dict[str, float | int | str]] = []
    for variant, delta_col in variants.items():
        print(f"collect_train variant={variant}")
        train_frame = collect_train_frame(
            paths=train_paths,
            target_col=args.target_col,
            delta_col=delta_col,
            eval_session_mod=args.eval_session_mod,
            eval_session_rem=args.eval_session_rem,
            max_train_rows=args.max_train_rows,
            seed=args.seed,
        )
        positive_rows = int(train_frame[args.target_col].sum())
        train_rows = int(len(train_frame))
        negative_rows = train_rows - positive_rows
        print(f"train_rows={train_rows:,} positive={positive_rows:,} negative={negative_rows:,}")

        model = train_one_variant(train_frame, args.target_col, feature_cols, args)
        model_path = output_dir / f"click_lgbm_{variant}.txt"
        model.save_model(model_path)
        importance = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance_gain": model.feature_importance(importance_type="gain"),
                "importance_split": model.feature_importance(importance_type="split"),
            }
        ).sort_values("importance_gain", ascending=False)
        importance_path = reports_dir / f"time_split_click_lgbm_{variant}_feature_importance.csv"
        importance.to_csv(importance_path, index=False)

        eval_metrics = evaluate_recall_at_20(
            model=model,
            paths=eval_paths,
            target_col=args.target_col,
            delta_col=delta_col,
            eval_session_mod=args.eval_session_mod,
            eval_session_rem=args.eval_session_rem,
        )
        row = {
            "variant": variant,
            "train_rows": train_rows,
            "train_positive_rows": positive_rows,
            "train_negative_rows": negative_rows,
            "model_path": str(model_path),
            "feature_importance_path": str(importance_path),
            **eval_metrics,
        }
        metrics.append(row)
        print(row)
        del train_frame, model, importance
        gc.collect()

    metrics_frame = pd.DataFrame(metrics)
    metrics_path = reports_dir / "time_split_click_lgbm_baseline_metrics.csv"
    metrics_json_path = reports_dir / "time_split_click_lgbm_baseline_metrics.json"
    metrics_frame.to_csv(metrics_path, index=False)
    metrics_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"metrics -> {metrics_path}")
    print(metrics_frame)


if __name__ == "__main__":
    main()
