from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import polars as pl

from otto_recommender.hard_negative_sampling import HardNegativeConfig, write_hard_negative_parts
from otto_recommender.metrics_polars import mean_ndcg_at_k_polars

FEATURES = [
    "score", "rank", "local_interaction_count", "local_click_count", "local_cart_count",
    "local_order_count", "total_interactions", "click_count", "cart_count", "order_count",
    "recent_24h_interactions", "conversion_rate", "item_cart_conversion_rate",
    "item_buy_conversion_rate", "item_cart_to_order_rate", "item_funnel_dropoff_rate",
    "session_length", "session_unique_items", "session_duration", "session_cart_count",
    "session_order_count", "nn_global_score", "nn_explore_score",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates-dir", default="artifacts/features/session_tail_candidates_static_parts")
    p.add_argument("--prefix", default="data/processed/session_tail_split/valid_prefix.parquet")
    p.add_argument("--targets", default="data/processed/session_tail_split/valid_targets.parquet")
    p.add_argument("--work-dir", default="artifacts/ranker/session_tail_order_ablation")
    p.add_argument("--models-dir", default="artifacts/models/session_tail_order_lgbm")
    p.add_argument("--reports-dir", default="artifacts/reports")
    p.add_argument("--target-col", default="target_order", choices=("target_click", "target_cart", "target_order"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=250)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--smoke-rows", type=int, default=0)
    return p.parse_args()


def git_state() -> tuple[str, str]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return commit, str(dirty)
    except Exception:
        return "unknown", "unknown"


def build_labeled_parts(a: argparse.Namespace, paths: list[Path]) -> list[Path]:
    out = Path(a.work_dir) / "labeled_parts"
    out.mkdir(parents=True, exist_ok=True)
    local = pl.scan_parquet(a.prefix).select("session", "aid", "type").group_by(["session", "aid"]).agg(
        pl.len().alias("local_interaction_count"),
        (pl.col("type") == 0).sum().cast(pl.Int16).alias("local_click_count"),
        (pl.col("type") == 1).sum().cast(pl.Int16).alias("local_cart_count"),
        (pl.col("type") == 2).sum().cast(pl.Int16).alias("local_order_count"),
    )
    targets = pl.scan_parquet(a.targets).select("session", "aid", a.target_col)
    result = []
    for path in paths:
        dest = out / path.name
        frame = pl.scan_parquet(path).join(local, on=["session", "aid"], how="left").join(
            targets, on=["session", "aid"], how="left"
        ).with_columns(
            pl.col("local_interaction_count").fill_null(0).cast(pl.Int16),
            pl.col("local_click_count").fill_null(0).cast(pl.Int16),
            pl.col("local_cart_count").fill_null(0).cast(pl.Int16),
            pl.col("local_order_count").fill_null(0).cast(pl.Int16),
            pl.col(a.target_col).fill_null(0).cast(pl.Int8),
            pl.col("score").alias("graph_weight_total"),
        )
        if a.smoke_rows > 0:
            frame = frame.head(a.smoke_rows)
        frame.sink_parquet(dest)
        result.append(dest)
    return result


def sample_parts(a: argparse.Namespace, labeled: list[Path], name: str, ratios: tuple[float, float, float]) -> list[Path]:
    out = Path(a.work_dir) / f"sampled_{name}"
    stats = Path(a.reports_dir) / f"session_tail_order_{name}_sampling_stats.csv"
    write_hard_negative_parts(
        labeled, out, stats,
        HardNegativeConfig(target_col=a.target_col, graph_col="graph_weight_total",
                           hard_click_neg_per_pos=ratios[0], hard_graph_neg_per_pos=ratios[1],
                           random_neg_per_pos=ratios[2], seed=a.seed),
    )
    return sorted(out.glob("*.parquet"))


def train(sampled: list[Path], a: argparse.Namespace):
    schema = pl.scan_parquet(str(sampled[0])).collect_schema().names()
    features = [x for x in FEATURES if x in schema]
    frame = pl.scan_parquet([str(x) for x in sampled]).select(features + [a.target_col]).collect()
    y = frame.get_column(a.target_col).to_numpy().astype(np.int8)
    frame = frame.drop(a.target_col)
    model = lgb.train({"objective": "binary", "metric": "binary_logloss", "learning_rate": a.learning_rate,
                       "num_leaves": a.num_leaves, "feature_fraction": 0.9, "bagging_fraction": 0.9,
                       "bagging_freq": 1, "min_data_in_leaf": 100, "seed": a.seed, "verbosity": -1},
                      lgb.Dataset(frame.to_pandas(), label=y), num_boost_round=a.rounds)
    return model, features, len(y), int(y.sum())


def evaluate(model, features: list[str], paths: list[Path], targets: str, target_col: str, k: int = 20):
    ranked_parts = []
    rows = 0
    for path in paths:
        frame = pl.read_parquet(path)
        scores = model.predict(frame.select(features).to_pandas())
        ranked_parts.append(frame.select("session", "aid").with_columns(
            pl.Series("score_pred", scores).cast(pl.Float32)
        ).with_columns(
            pl.col("score_pred").rank(method="ordinal", descending=True).over("session").alias("pred_rank")
        ).filter(pl.col("pred_rank") <= k).select("session", "aid", "score_pred", "pred_rank"))
        rows += frame.height
        del frame
        gc.collect()
    predictions = pl.concat(ranked_parts)
    relevant = pl.scan_parquet(targets).filter(pl.col(target_col) == 1).select("session", "aid").collect()
    hit_rows = predictions.join(relevant, on=["session", "aid"], how="inner").with_columns(
        (1.0 / (pl.col("pred_rank").cast(pl.Float64) + 1).log(base=2)).alias("gain")
    )
    dcg = hit_rows.group_by("session").agg(pl.sum("gain").alias("dcg"))
    session_counts = relevant.group_by("session").agg(pl.len().alias("n_relevant"))
    discounts = pl.DataFrame({"rank": list(range(1, k + 1))}).with_columns(
        (1.0 / (pl.col("rank").cast(pl.Float64) + 1).log(base=2)).alias("discount")
    )
    idcg = session_counts.with_columns(pl.col("n_relevant").clip(upper_bound=k).alias("n")).join(
        discounts, how="cross"
    ).filter(pl.col("rank") <= pl.col("n")).group_by("session").agg(pl.sum("discount").alias("idcg"))
    per_session = session_counts.select("session").join(dcg, on="session", how="left").join(
        idcg, on="session", how="left"
    ).with_columns(pl.col("dcg").fill_null(0.0),
                   (pl.col("dcg").fill_null(0.0) / pl.col("idcg")).alias("ndcg"))
    row = per_session.select(pl.len().alias("order_sessions"), pl.sum("dcg").alias("dcg_sum"),
                             pl.sum("idcg").alias("idcg_sum"), pl.mean("ndcg").alias("ndcg_at_20"),
                             (pl.col("dcg") > 0).sum().alias("hit_sessions")).row(0, named=True)
    ndcg_check = mean_ndcg_at_k_polars(
        predictions.select("session", "aid", "score_pred"),
        pl.scan_parquet(targets), score_col="score_pred", k=k, target_col=target_col,
    )
    return {"eval_rows": rows, "order_sessions": int(row["order_sessions"]), "hit_sessions": int(row["hit_sessions"]),
            "dcg_sum": float(row["dcg_sum"]), "idcg_sum": float(row["idcg_sum"]), "ndcg_at_20": float(ndcg_check)}


def main() -> None:
    a = parse_args()
    paths = sorted(Path(a.candidates_dir).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(a.candidates_dir)
    labeled = build_labeled_parts(a, paths)
    commit, dirty = git_state()
    mlflow.set_experiment("otto-local-validation")
    results = []
    for name, ratios in (("A_1_8_10_2", (8, 10, 2)), ("B_1_10_8_2", (10, 8, 2)), ("C_1_6_12_2", (6, 12, 2))):
        started = time.perf_counter()
        sampled = sample_parts(a, labeled, name, ratios)
        model, features, train_rows, positives = train(sampled, a)
        model_dir = Path(a.models_dir); model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / f"{name}.txt"; model.save_model(model_file)
        metrics = evaluate(model, features, labeled, a.targets, a.target_col)
        row = {"model": name, "train_rows": train_rows, "train_positive_rows": positives,
               "model_path": str(model_file), "features": json.dumps(features),
               "elapsed_seconds": time.perf_counter() - started, **metrics}
        results.append(row)
        with mlflow.start_run(run_name=f"session-tail-order-{name}"):
            mlflow.log_params({"model": name, "ratios": str(ratios), "seed": a.seed, "rounds": a.rounds,
                               "candidate_rows": metrics["eval_rows"], "git_commit": commit, "git_dirty": dirty})
            mlflow.log_metric("ndcg_at_20", metrics["ndcg_at_20"])
            mlflow.log_artifact(str(model_file))
        print(row, flush=True)
    report = Path(a.reports_dir); report.mkdir(parents=True, exist_ok=True)
    report_name = a.target_col.replace("target_", "")
    pl.DataFrame(results).write_csv(report / f"session_tail_{report_name}_ablation_ndcg.csv")
    print(pl.DataFrame(results))


if __name__ == "__main__":
    main()
