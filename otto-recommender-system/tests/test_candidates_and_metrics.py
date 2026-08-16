from __future__ import annotations

from otto_recommender.candidates import candidates_from_recent_and_covisitation
from otto_recommender.metrics import recall_at_k
from otto_recommender.submission import predictions_for_metric
from otto_recommender.toy_data import make_toy_events, make_toy_labels


def test_toy_candidates_include_recent_items() -> None:
    events = make_toy_events()
    candidates = candidates_from_recent_and_covisitation(events, final_topk=3)

    assert candidates[1][0] == 104
    assert len(candidates[1]) == 3


def test_recall_at_k_on_toy_data() -> None:
    events = make_toy_events()
    labels = make_toy_labels()
    candidates = candidates_from_recent_and_covisitation(events, final_topk=20)
    predictions = predictions_for_metric(candidates)

    assert recall_at_k(labels, predictions, k=20) > 0
