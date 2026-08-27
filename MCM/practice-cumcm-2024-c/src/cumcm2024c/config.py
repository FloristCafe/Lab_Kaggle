from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path(r"D:\MCM\CUMCM-2024\problems\CUMCM2024ProblemsE\ProblemC")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
YEARS = tuple(range(2024, 2031))
SEASONS = (1, 2)

