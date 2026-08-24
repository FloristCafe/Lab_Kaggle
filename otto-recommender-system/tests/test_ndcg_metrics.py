from __future__ import annotations

import pandas as pd

from otto_recommender.metrics import binary_dcg_at_ranks, binary_idcg_at_k, ndcg_at_k_from_ranked_hits


def test_binary_dcg_discount_rewards_early_hits_more() -> None:
    assert binary_dcg_at_ranks([1], k=20) > binary_dcg_at_ranks([20], k=20)
    assert binary_idcg_at_k(2, k=20) > binary_idcg_at_k(1, k=20)


def test_ndcg_counts_labeled_sessions_without_hits_as_zero() -> None:
    labels = pd.DataFrame(
        {
            "session": [1, 2],
            "ground_truth": [[10], [20]],
        }
    )
    hits = pd.DataFrame(
        {
            "session": [1],
            "rank": [1],
        }
    )

    metrics = ndcg_at_k_from_ranked_hits(labels, hits, k=20)

    assert metrics["label_sessions"] == 2
    assert metrics["hit_sessions"] == 1
    assert metrics["ndcg_at_k"] == 0.5
