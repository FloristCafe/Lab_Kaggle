"""Utilities for making local micro datasets from OTTO JSONL files."""

from __future__ import annotations

import json
import random
from pathlib import Path


def sample_jsonl_sessions(
    input_path: str | Path,
    output_path: str | Path,
    n_sessions: int,
    seed: int = 42,
) -> int:
    """Reservoir-sample complete sessions from a large OTTO JSONL file."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0
    with input_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            json.loads(line)
            seen += 1
            if len(reservoir) < n_sessions:
                reservoir.append(line)
                continue
            index = rng.randint(0, seen - 1)
            if index < n_sessions:
                reservoir[index] = line

    with output_path.open("w", encoding="utf-8") as target:
        target.writelines(reservoir)
    return len(reservoir)

