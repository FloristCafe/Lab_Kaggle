"""C. Permutation-equivariant unit encoding and conditional fleet assignment."""
import math

import torch
from torch import nn

from .shared.model import SpatialActor, WIDTH, CELLS, MODES


class Actor(SpatialActor):
    def __init__(self):
        super().__init__()
        self.units_attention = nn.MultiheadAttention(WIDTH, 2, batch_first=True)
        self.team_norm = nn.LayerNorm(WIDTH)
        self.prefix_query = nn.Linear(WIDTH * 2, WIDTH)
        self.prefix_gate = nn.Linear(WIDTH, MODES + 2)
        self.coverage = nn.Parameter(torch.zeros(MODES))
        cells = torch.arange(CELLS)
        xy = torch.stack((cells // 24, cells % 24), dim=-1)
        self.register_buffer("geometry", xy, persistent=False)

    def contextualize(self, tiles, tokens, state):
        mask = ~state["alive"].clone()
        mask[:, 0] &= ~mask.all(1)
        interactions, _ = self.units_attention(tokens, tokens, tokens, key_padding_mask=mask,
                                               need_weights=False)
        return tiles, self.team_norm(tokens + interactions)

    def prefix_adjustment(self, encoded, uid, claims):
        r, modes, goals, tiles, tokens = encoded
        active = claims >= 0
        indices = claims.clamp_min(0)
        claimed = tiles.gather(1, indices[..., None].expand(-1, -1, WIDTH))
        pooled = (claimed * active.unsqueeze(-1)).sum(1) / active.sum(1, keepdim=True).clamp_min(1)
        q = torch.tanh(self.prefix_query(torch.cat((tokens[:, uid], pooled), dim=-1)))
        signal = self.prefix_gate(q)
        d = (self.geometry[indices, None] - self.geometry[None, None]).abs().sum(-1)
        overlap = ((d <= 4) & active[..., None]).sum(1).clamp(max=4).float() / 4
        context = torch.einsum("bd,bgd->bg", q, tiles) / math.sqrt(WIDTH)
        logits = goals[:, uid] + signal[:, 2:, None] * context[:, None]
        return r[:, uid] + signal[:, :2], modes[:, uid] + signal[:, 2:], (
            logits + self.coverage[None, :, None] * overlap[:, None]
        )
