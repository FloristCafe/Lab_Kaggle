"""Memory-safe SASRec components for candidate scoring.

The module deliberately exposes training and inference primitives instead of an
end-to-end data loader. Parquet chunking remains the pipeline boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SASRecConfig:
    n_items: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_len: int = 20
    dropout: float = 0.1
    temperature: float = 0.07


class SASRecEncoder(nn.Module):
    """Causal sequence encoder with explicit padding masking."""

    def __init__(self, config: SASRecConfig):
        super().__init__()
        self.config = config
        self.item = nn.Embedding(config.n_items + 1, config.d_model, padding_idx=0)
        self.position = nn.Embedding(config.max_len, config.d_model)
        self.type_embedding = nn.Embedding(3, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads, dropout=config.dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, item_ids: Tensor, type_ids: Tensor) -> Tensor:
        if item_ids.ndim != 2 or item_ids.shape[1] != self.config.max_len:
            raise ValueError("item_ids must have shape [batch, max_len]")
        padding = item_ids.eq(0)
        positions = torch.arange(self.config.max_len, device=item_ids.device).unsqueeze(0)
        x = self.item(item_ids) + self.position(positions) + self.type_embedding(type_ids.clamp(0, 2))
        causal = torch.triu(torch.ones(self.config.max_len, self.config.max_len, device=x.device, dtype=torch.bool), diagonal=1)
        x = self.encoder(x, mask=causal, src_key_padding_mask=padding)
        lengths = (~padding).sum(dim=1).clamp_min(1) - 1
        return self.norm(x[torch.arange(x.shape[0], device=x.device), lengths])


def masked_in_batch_nce(session_vec: Tensor, positive_ids: Tensor, item_embedding: nn.Embedding, temperature: float = 0.07) -> Tensor:
    """InfoNCE with diagonal and same-aid false-negative masking."""
    positives = item_embedding(positive_ids)
    logits = session_vec @ positives.transpose(0, 1) / temperature
    same_aid = positive_ids[:, None].eq(positive_ids[None, :])
    same_aid.fill_diagonal_(True)
    logits = logits.masked_fill(same_aid, torch.finfo(logits.dtype).min)
    # Keep the positive class available on the diagonal while masking accidental duplicates.
    logits[torch.arange(logits.shape[0], device=logits.device), torch.arange(logits.shape[0], device=logits.device)] = (
        (session_vec * positives).sum(dim=1) / temperature
    )
    return nn.functional.cross_entropy(logits, torch.arange(logits.shape[0], device=logits.device))


def score_candidates(model: SASRecEncoder, item_ids: Tensor, type_ids: Tensor, candidate_ids: Tensor) -> Tensor:
    """Detach model scores before returning CPU features."""
    with torch.inference_mode():
        session_vec = model(item_ids, type_ids)
        scores = session_vec @ model.item(candidate_ids).transpose(-1, -2)
    return scores.detach().cpu()
