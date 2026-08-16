"""Input/output helpers for OTTO session data."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import AID, SESSION, TS, TYPE


def flatten_sessions(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Convert OTTO JSONL records into an event-level DataFrame."""
    rows: list[dict[str, Any]] = []
    for record in records:
        session = record[SESSION]
        for event in record.get("events", []):
            rows.append(
                {
                    SESSION: session,
                    AID: event[AID],
                    TS: event[TS],
                    TYPE: event[TYPE],
                }
            )
    return pd.DataFrame(rows, columns=[SESSION, AID, TS, TYPE])


def iter_otto_jsonl(path: str | Path, chunk_size: int = 10_000) -> Iterator[pd.DataFrame]:
    """Yield flattened event chunks from Kaggle's OTTO JSONL files."""
    path = Path(path)
    chunk: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            chunk.append(json.loads(line))
            if len(chunk) >= chunk_size:
                yield flatten_sessions(chunk)
                chunk.clear()
    if chunk:
        yield flatten_sessions(chunk)


def read_otto_jsonl(path: str | Path, n_sessions: int | None = None) -> pd.DataFrame:
    """Read a small or medium OTTO JSONL file into one event-level DataFrame."""
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            if n_sessions is not None and i >= n_sessions:
                break
            if line.strip():
                records.append(json.loads(line))
    return flatten_sessions(records)


def read_events(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read events from parquet, csv, or OTTO JSONL."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix in {".jsonl", ".json"}:
        return read_otto_jsonl(path, **kwargs)
    raise ValueError(f"Unsupported event file type: {path}")


def write_events(df: pd.DataFrame, path: str | Path) -> None:
    """Write event data using the file extension to choose the format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported output file type: {path}")

