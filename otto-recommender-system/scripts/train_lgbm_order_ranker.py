from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from otto_recommender.metrics import ndcg_at_k_from_ranked_hits
from otto_recommender.schema import AID, SESSION, TYPE

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

ITEM_FEATURES = [
    "total_interactions",
    "click_count",
    "cart_count",
    "order_count",
    "recent_24h_interactions",
    "conversion_rate",
    "item_cart_conversion_rate",
    "item_buy_conversion_rate",
    "item_cart_to_order_rate",
    "item_funnel_dropoff_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM Order ranker and evaluate strict NDCG@20.")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--eval-dir", default="artifacts/ranker/time_split/labeled_features_parts")
    parser.add_argument("--labels", default="data/processed/time_split/valid_labels.parquet")
    parser.add_argument("--item-features", default=None)
    parser.add_argument("--output-dir", default="artifacts/models/time_split_order_lgbm")
    parser.add_argument("--reports-dir", default="artifacts/reports")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--target-col", default="target_order")
    parser.add_argument("--target-type", type=int, default=2)
    parser.add_argument("--delta-col", default="delta_t_filled")
    parser.add_argument("--num-boost-round", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--pattern", default="*.parquet")
    return parser.parse_args()


def part_paths(directory: str | Path, pattern: str) -> list[Path]:
    paths = sorted(Path(directory).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No parquet parts matched {Path(directory) / pattern}")
    return paths


def item_feature_scan(path: str | Path) -> pl.LazyFrame:
    schema = pl.scan_parquet(path).collect_schema().names()
    cols = [col for col in ITEM_FEATURES if col in schema]
    return pl.scan_parquet(path).select(pl.col(AID).cast(pl.Int32), *[pl.col(col) for col in cols])


def feature_columns(paths: list[Path], delta_col: str, item_features_path: str | Path | None) -> list[str]:
    names = pl.scan_parquet(str(paths[0])).collect_schema().names()
    cols = [col for col in BASE_FEATURES if col in names]
    cols.append("delta_t")
    if item_features_path is not None:
        item_names = item_feature_scan(item_features_path).collect_schema().names()
        cols.extend([col for col in ITEM_FEATURES if col in item_names])
    else:
        cols.extend([col for col in ITEM_FEATURES if col in names])
    return list(dict.fromkeys(cols))


def scan_feature_parts(
    paths: list[Path],
    target_col: str,
    delta_col: str,
    item_features_path: str | Path | None,
) -> pl.LazyFrame:
    frame = pl.scan_parquet([str(path) for path in paths])
    if item_features_path is not None:
        existing = frame.collect_schema().names()
        missing_item_cols = [col for col in ITEM_FEATURES if col not in existing]
        if missing_item_cols:
            frame = frame.join(item_feature_scan(item_features_path), on=AID, how="left")
    return frame.rename({delta_col: "delta_t"}).select(
        pl.col(SESSION).cast(pl.Int32),
        pl.col(AID).cast(pl.Int32),
        pl.col(target_col).cast(pl.Int8),
        *[pl.col(col) for col in BASE_FEATURES if col in frame.collect_schema().names()],
        pl.col("delta_t"),
        *[pl.col(col) for col in ITEM_FEATURES if col in frame.collect_schema().names()],
    )


def collect_train_frame(
    paths: list[Path],
    target_col: str,
    delta_col: str,
    item_features_path: str | Path | None,
    max_train_rows: int,
    seed: int,
) -> pd.DataFrame:
    lazy = scan_feature_parts(paths, target_col, delta_col, item_features_path).drop(SESSION, AID)
    if max_train_rows > 0:
        total = lazy.select(pl.len()).collect().item()
        fraction = min(1.0, max_train_rows / max(total, 1))
        threshold = int(fraction * 1_000_000)
        lazy = lazy.with_row_index("_row_nr").with_columns(
            pl.col("_row_nr").hash(seed=seed).mod(1_000_000).alias("_train_hash")
        ).filter(pl.col("_train_hash") < threshold).drop("_row_nr", "_train_hash")
    return lazy.collect().to_pandas()


def train_model(train_frame: pd.DataFrame, target_col: str, features: list[str], args: argparse.Namespace) -> lgb.Booster:
    y = train_frame[target_col].astype(np.int8)
    x = train_frame[features]
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


def order_labels(labels_path: str | Path, target_type: int) -> pd.DataFrame:
    return (
        pl.scan_parquet(labels_path)
        .filter(pl.col(TYPE) == target_type)
        .select(SESSION, "ground_truth")
        .collect()
        .to_pandas()
    )


def evaluate_ndcg_at_k(
    model: lgb.Booster,
    paths: list[Path],
    labels_path: str | Path,
    target_col: str,
    target_type: int,
    delta_col: str,
    item_features_path: str | Path | None,
    features: list[str],
    k: int,
) -> dict[str, float | int]:
    label_df = order_labels(labels_path, target_type)
    hit_parts: list[pd.DataFrame] = []
    eval_rows = 0
    eval_positive_rows = 0
    topk_rows = 0

    for path in paths:
        frame = scan_feature_parts([path], target_col, delta_col, item_features_path).collect()
        if frame.is_empty():
            continue
        scores = model.predict(frame.select(features).to_pandas())
        scored = frame.select(SESSION, AID, target_col).with_columns(
            pl.Series("pred_score", scores).cast(pl.Float32)
        )
        eval_rows += scored.height
        eval_positive_rows += int(scored.select(pl.sum(target_col)).item())
        topk = (
            scored.with_columns(
                pl.col("pred_score")
                .rank(method="ordinal", descending=True)
                .over(SESSION)
                .cast(pl.UInt32)
                .alias("rank")
            )
            .filter(pl.col("rank") <= k)
            .select(SESSION, AID, "rank", target_col)
        )
        topk_rows += topk.height
        hits = topk.filter(pl.col(target_col) == 1).select(SESSION, "rank")
        if not hits.is_empty():
            hit_parts.append(hits.to_pandas())
        del frame, scored, topk, hits
        gc.collect()

    ranked_hits = pd.concat(hit_parts, ignore_index=True) if hit_parts else pd.DataFrame(columns=[SESSION, "rank"])
    metrics = ndcg_at_k_from_ranked_hits(label_df, ranked_hits, k=k)
    metrics.update(
        {
            "eval_rows": eval_rows,
            "eval_positive_rows_in_candidates": eval_positive_rows,
            "topk_rows": topk_rows,
            "topk_hits": int(len(ranked_hits)),
        }
    )
    return metrics


def main() -> None:
    args = parse_args()
    train_paths = part_paths(args.train_dir, args.pattern)
    eval_paths = part_paths(args.eval_dir, args.pattern)
    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    features = feature_columns(train_paths, args.delta_col, args.item_features)
    train_frame = collect_train_frame(
        paths=train_paths,
        target_col=args.target_col,
        delta_col=args.delta_col,
        item_features_path=args.item_features,
        max_train_rows=args.max_train_rows,
        seed=args.seed,
    )
    positive_rows = int(train_frame[args.target_col].sum())
    train_rows = int(len(train_frame))
    negative_rows = train_rows - positive_rows
    print(f"train_rows={train_rows:,} positive={positive_rows:,} negative={negative_rows:,}")

    model = train_model(train_frame, args.target_col, features, args)
    model_path = output_dir / f"{args.experiment_name}_lgbm.txt"
    model.save_model(model_path)

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance_gain": model.feature_importance(importance_type="gain"),
            "importance_split": model.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    importance_path = reports_dir / f"{args.experiment_name}_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    metrics = evaluate_ndcg_at_k(
        model=model,
        paths=eval_paths,
        labels_path=args.labels,
        target_col=args.target_col,
        target_type=args.target_type,
        delta_col=args.delta_col,
        item_features_path=args.item_features,
        features=features,
        k=args.k,
    )
    row = {
        "experiment": args.experiment_name,
        "train_rows": train_rows,
        "train_positive_rows": positive_rows,
        "train_negative_rows": negative_rows,
        "model_path": str(model_path),
        "feature_importance_path": str(importance_path),
        **metrics,
    }
    metrics_path = reports_dir / f"{args.experiment_name}_metrics.csv"
    metrics_json_path = reports_dir / f"{args.experiment_name}_metrics.json"
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    metrics_json_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(row)
    print(f"metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
