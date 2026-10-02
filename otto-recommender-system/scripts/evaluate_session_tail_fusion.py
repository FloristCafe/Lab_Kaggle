from __future__ import annotations
import argparse
from pathlib import Path
import gc
import lightgbm as lgb
import polars as pl
from otto_recommender.metrics_polars import mean_ndcg_at_k_polars

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--candidates-dir", required=True)
    p.add_argument("--targets", required=True)
    p.add_argument("--click-model", required=True)
    p.add_argument("--cart-model", required=True)
    p.add_argument("--order-model", required=True)
    p.add_argument("--click-weight", type=float, default=0.2)
    p.add_argument("--cart-weight", type=float, default=0.3)
    p.add_argument("--order-weight", type=float, default=0.5)
    p.add_argument("--output", required=True)
    a=p.parse_args()
    models={"click":lgb.Booster(model_file=a.click_model),"cart":lgb.Booster(model_file=a.cart_model),"order":lgb.Booster(model_file=a.order_model)}
    paths=sorted(Path(a.candidates_dir).glob("*.parquet"))
    if not paths: raise FileNotFoundError(a.candidates_dir)
    parts=[]
    for path in paths:
        frame=pl.read_parquet(path)
        scores={name:model.predict(frame.select(model.feature_name()).to_pandas()) for name,model in models.items()}
        part=frame.select("session","aid").with_columns(
            pl.Series("click_score",scores["click"]).cast(pl.Float32),
            pl.Series("cart_score",scores["cart"]).cast(pl.Float32),
            pl.Series("order_score",scores["order"]).cast(pl.Float32),
        ).with_columns((pl.col("click_score")*a.click_weight+pl.col("cart_score")*a.cart_weight+pl.col("order_score")*a.order_weight).alias("fusion_score"))
        parts.append(part)
        del frame,part
        gc.collect()
        print("predicted",path.name,flush=True)
    pred=pl.concat(parts)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    pred.write_parquet(a.output)
    targets=pl.scan_parquet(a.targets)
    result={target:mean_ndcg_at_k_polars(pred.select("session","aid","fusion_score"),targets,score_col="fusion_score",target_col=target,k=20) for target in ("target_click","target_cart","target_order")}
    print({"weights":{"click":a.click_weight,"cart":a.cart_weight,"order":a.order_weight},"rows":pred.height,"ndcg_at_20":result})
if __name__=="__main__":
    main()

