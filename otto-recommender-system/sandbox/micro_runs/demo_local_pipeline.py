from __future__ import annotations

from otto_recommender.candidates import candidates_from_recent_and_covisitation
from otto_recommender.metrics import recall_at_k
from otto_recommender.submission import candidates_to_submission, predictions_for_metric
from otto_recommender.toy_data import make_toy_events, make_toy_labels


def main() -> None:
    events = make_toy_events()
    labels = make_toy_labels()

    candidates = candidates_from_recent_and_covisitation(events, final_topk=20)
    metric_frame = predictions_for_metric(candidates)
    score = recall_at_k(labels, metric_frame, k=20)

    submission = candidates_to_submission(candidates)
    print("Toy Recall@20:", round(score, 6))
    print(submission.head(9).to_string(index=False))


if __name__ == "__main__":
    main()
