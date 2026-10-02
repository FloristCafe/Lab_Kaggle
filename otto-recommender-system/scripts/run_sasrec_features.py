"""Train a small SASRec and materialize NN candidate scores.

The script keeps the disk boundary explicit: prefix events and candidate parts
are read from Parquet, while the model checkpoint and score parts are written
back to Parquet.  It is intended for local smoke runs first; full runs should
use the same command with a larger epoch/row budget after the smoke passes.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from otto_recommender.sasrec import SASRecConfig, SASRecEncoder, masked_in_batch_nce


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("train", "score"), required=True)
    p.add_argument("--prefix", default="data/processed/session_tail_split/valid_prefix.parquet")
    p.add_argument("--candidates-dir", default="artifacts/features/session_tail_candidates_static_parts")
    p.add_argument("--output-dir", default="artifacts/features/session_tail_nn_parts")
    p.add_argument("--checkpoint", default="artifacts/models/session_tail_sasrec.pt")
    p.add_argument("--max-len", type=int, default=20)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--max-sessions", type=int, default=0)
    p.add_argument("--max-candidate-rows", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def device_name(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return value


def load_sequences(path: str, max_len: int, max_sessions: int, seed: int, include_singletons: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray], list[int]]:
    frame = pl.scan_parquet(path).select("session", "aid", "type", "ts").sort(["session", "ts"]).collect()
    groups = frame.group_by("session", maintain_order=True).agg(
        pl.col("aid").alias("aids"), pl.col("type").alias("types")
    )
    if max_sessions and groups.height > max_sessions:
        groups = groups.sample(n=max_sessions, seed=seed)
    items: list[list[int]] = []
    types: list[list[int]] = []
    positives: list[int] = []
    history: dict[int, np.ndarray] = {}
    session_ids: list[int] = []
    for row in groups.iter_rows(named=True):
        aids = [int(x) for x in row["aids"]]
        ty = [int(x) for x in row["types"]]
        if len(aids) < 1 or (len(aids) < 2 and not include_singletons):
            continue
        positives.append(aids[-1] if len(aids) > 1 else 0)
        session_id = int(row["session"])
        session_ids.append(session_id)
        history[session_id] = np.asarray(list(set(aids)), dtype=np.int64)
        if len(aids) > 1:
            aids, ty = aids[:-1][-max_len:], ty[:-1][-max_len:]
        else:
            aids, ty = aids[-max_len:], ty[-max_len:]
        pad = max_len - len(aids)
        # Right padding avoids fully masked query rows under causal attention;
        # left padding would create 0/0 softmax rows and NaN representations.
        items.append(aids + [0] * pad)
        types.append(ty + [0] * pad)
    if not items:
        raise ValueError("No sessions with at least two prefix events")
    return np.asarray(items, dtype=np.int64), np.asarray(types, dtype=np.int64), np.asarray(positives, dtype=np.int64), history, session_ids


def train(a: argparse.Namespace, device: str) -> None:
    item_ids, type_ids, positives, _, _ = load_sequences(a.prefix, a.max_len, a.max_sessions, a.seed)
    # The embedding vocabulary must cover the complete prefix catalog even when
    # training uses a session sample; otherwise full-catalog scoring can index
    # past the sampled vocabulary on CUDA.
    max_aid = int(pl.scan_parquet(a.prefix).select(pl.col("aid").max()).collect().item())
    candidate_paths = sorted(Path(a.candidates_dir).glob("*.parquet"))
    if candidate_paths:
        candidate_max = pl.scan_parquet([str(path) for path in candidate_paths]).select(pl.col("aid").max()).collect().item()
        max_aid = max(max_aid, int(candidate_max))
    n_items = max_aid
    config = SASRecConfig(n_items=n_items, d_model=a.d_model, n_heads=a.heads, n_layers=a.layers, max_len=a.max_len)
    model = SASRecEncoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate)
    loader = DataLoader(TensorDataset(torch.from_numpy(item_ids), torch.from_numpy(type_ids), torch.from_numpy(positives)), batch_size=a.batch_size, shuffle=True)
    model.train()
    started = time.perf_counter()
    for epoch in range(a.epochs):
        losses = []
        for batch_items, batch_types, batch_pos in loader:
            batch_items, batch_types, batch_pos = batch_items.to(device), batch_types.to(device), batch_pos.to(device)
            optimizer.zero_grad(set_to_none=True)
            session_vec = model(batch_items, batch_types)
            loss = masked_in_batch_nce(session_vec, batch_pos, model.item, config.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch + 1} loss={np.mean(losses):.6f}", flush=True)
    Path(a.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.__dict__, "state_dict": model.state_dict(), "seed": a.seed}, a.checkpoint)
    print(json.dumps({"checkpoint": a.checkpoint, "sessions": len(item_ids), "n_items": n_items, "elapsed_seconds": time.perf_counter() - started}))


def score(a: argparse.Namespace, device: str) -> None:
    checkpoint = torch.load(a.checkpoint, map_location=device, weights_only=False)
    config = SASRecConfig(**checkpoint["config"])
    model = SASRecEncoder(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    item_ids, type_ids, _, history, session_ids = load_sequences(a.prefix, a.max_len, a.max_sessions, a.seed, include_singletons=True)
    observed_max = int(pl.scan_parquet(a.prefix).select(pl.col("aid").max()).collect().item())
    if observed_max > config.n_items:
        raise ValueError(
            f"checkpoint vocabulary n_items={config.n_items} is smaller than prefix max aid={observed_max}; retrain checkpoint"
        )
    history_pairs = pl.scan_parquet(a.prefix).select("session", "aid").unique().with_columns(
        pl.lit(1, dtype=pl.Int8).alias("_seen")
    ).collect()
    session_vecs: dict[int, torch.Tensor] = {}
    with torch.inference_mode():
        for start in range(0, len(item_ids), a.batch_size):
            batch = model(torch.from_numpy(item_ids[start:start + a.batch_size]).to(device), torch.from_numpy(type_ids[start:start + a.batch_size]).to(device))
            for sid, vec in zip(session_ids[start:start + len(batch)], batch.detach().cpu()):
                session_vecs[int(sid)] = vec
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(a.candidates_dir).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(a.candidates_dir)
    candidate_max = pl.scan_parquet([str(path) for path in paths]).select(pl.col("aid").max()).collect().item()
    if int(candidate_max) > config.n_items:
        raise ValueError(
            f"checkpoint vocabulary n_items={config.n_items} is smaller than candidate max aid={candidate_max}; retrain checkpoint"
        )
    for path in paths:
        frame = pl.read_parquet(path)
        if a.max_candidate_rows:
            frame = frame.head(a.max_candidate_rows)
        keys = frame.select("session", "aid")
        session_array = keys.get_column("session").to_numpy()
        aid_array = keys.get_column("aid").to_numpy().astype(np.int64)
        vectors = torch.from_numpy(np.stack([
            session_vecs.get(int(s), torch.zeros(config.d_model)).numpy() for s in session_array
        ])).to(device)
        valid = (aid_array >= 0) & (aid_array <= config.n_items)
        # Sessions with a single prefix event cannot provide a supervised
        # training target and therefore have no learned vector.  Use a neutral
        # zero score for those rows so the feature contract never emits NaN.
        score_values = np.zeros(frame.height, dtype=np.float32)
        with torch.inference_mode():
            if valid.any():
                aid_tensor = torch.from_numpy(aid_array[valid]).to(device)
                score_values[valid] = (vectors[valid] * model.item(aid_tensor)).sum(dim=1).detach().cpu().numpy()
        seen = keys.join(history_pairs, on=["session", "aid"], how="left", coalesce=True).select("_seen").to_series().fill_null(0).to_numpy()
        explore_values = score_values.copy()
        explore_values[seen == 1] = -np.inf
        frame = frame.with_columns(
            pl.Series("nn_global_score", score_values, dtype=pl.Float32),
            pl.Series("nn_explore_score", explore_values, dtype=pl.Float32),
        )
        frame.write_parquet(out / path.name)
        del frame, vectors, score_values, explore_values
        gc.collect()
        print(f"scored {path.name}", flush=True)


def main() -> None:
    a = args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = device_name(a.device)
    if a.mode == "train":
        train(a, device)
    else:
        score(a, device)


if __name__ == "__main__":
    main()
