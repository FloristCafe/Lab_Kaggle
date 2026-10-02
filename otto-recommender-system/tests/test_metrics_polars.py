import polars as pl

from otto_recommender.metrics_polars import mean_ndcg_at_k_polars


def test_mean_ndcg_polars_uses_binary_order_targets():
    candidates = pl.DataFrame({"session": [1, 1, 1, 2, 2], "aid": [10, 20, 30, 40, 50], "score": [0.9, 0.8, 0.1, 0.3, 0.2]})
    targets = pl.DataFrame({"session": [1, 2], "aid": [20, 50], "target_order": [1, 1]})
    value = mean_ndcg_at_k_polars(candidates, targets, k=2)
    assert 0.0 < value < 1.0
