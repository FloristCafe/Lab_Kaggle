# Kaggle notebook template for the OTTO local baseline.
#
# In Kaggle, either paste the source modules into notebook cells or upload this
# repository as a dataset and adjust CODE_ROOT below.

from pathlib import Path
import sys

CODE_ROOT = Path("/kaggle/input/otto-recommender-code/src")
DATA_ROOT = Path("/kaggle/input/otto-recommender-system")
WORKING = Path("/kaggle/working")

if CODE_ROOT.exists():
    sys.path.append(str(CODE_ROOT))

from otto_recommender.candidates import candidates_from_recent_and_covisitation
from otto_recommender.io import read_otto_jsonl
from otto_recommender.submission import candidates_to_submission


test_events = read_otto_jsonl(DATA_ROOT / "test.jsonl")
candidates = candidates_from_recent_and_covisitation(
    test_events,
    final_topk=20,
    covisitation_topk=40,
)
submission = candidates_to_submission(candidates)
submission.to_csv(WORKING / "submission.csv", index=False)
submission.head()
