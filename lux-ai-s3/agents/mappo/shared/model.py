"""Small spatial actor and action-prefix value baseline shared by all variants."""
import math

import torch
from torch import nn
from torch.nn import functional as F

CELLS = 576
WIDTH = 32
MODES = 5


class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Conv2d(19, WIDTH, 3, padding=1)
        self.context = nn.Sequential(
            nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.SiLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, dilation=3, padding=3), nn.SiLU(),
            nn.Conv2d(WIDTH, WIDTH, 3, dilation=4, padding=4), nn.SiLU(),
        )

    def forward(self, grid):
        local = F.silu(self.local(grid.float()))
        global_context = F.interpolate(self.context(local), size=(24, 24), mode="nearest")
        return (local + global_context).flatten(2).transpose(1, 2)


class SpatialActor(nn.Module):
    uses_memory = False

    def __init__(self):
        super().__init__()
        self.encoder = SpatialEncoder()
        self.unit = nn.Sequential(nn.Linear(22 + WIDTH * 2 + 6, 64), nn.SiLU(),
                                  nn.Linear(64, WIDTH))
        self.query = nn.Linear(WIDTH, MODES * WIDTH)
        self.key = nn.Linear(WIDTH, WIDTH)
        self.replan = nn.Linear(WIDTH, 2)
        self.mode = nn.Linear(WIDTH, MODES)
        self.cell_prior = nn.Linear(19, MODES, bias=False)
        self.route_prior = nn.Parameter(torch.zeros(MODES, 5))
        self.mode_gain = nn.Parameter(torch.zeros(MODES))
        with torch.no_grad():
            self.cell_prior.weight.zero_()
            self.route_prior[:, 0].fill_(-0.5)

    def contextualize(self, tiles, tokens, state):
        return tiles, tokens

    def forward(self, state):
        tiles = self.encoder(state["grid"])
        units = state["units"].float()
        xy = (units[..., :2] * 23).round().long()
        indices = xy[..., 0] * 24 + xy[..., 1]
        local = tiles.gather(1, indices[..., None].expand(-1, -1, WIDTH))
        mean = tiles.mean(dim=1, keepdim=True).expand_as(local)
        team = state["team"].float()[:, None].expand(-1, 16, -1)
        tokens = self.unit(torch.cat((units, local, mean, team), dim=-1))
        tiles, tokens = self.contextualize(tiles, tokens, state)
        queries = self.query(tokens).unflatten(-1, (MODES, WIDTH))
        goal = torch.einsum("bumd,bgd->bumg", queries, self.key(tiles)) / math.sqrt(WIDTH)
        physical = self.cell_prior(state["grid"].float().flatten(2).transpose(1, 2))
        route = torch.einsum("bucg,mc->bumg", state["reach"].float(), self.route_prior)
        return (self.replan(tokens), self.mode(tokens),
                goal + physical.transpose(1, 2)[:, None] + route, tiles, tokens)

    def prefix_adjustment(self, encoded, uid, claims):
        r, modes, goals, _, _ = encoded
        return r[:, uid], modes[:, uid], goals[:, uid]


class PrefixCritic(nn.Module):
    """Values condition on prior decisions, never the variable they baseline."""
    def __init__(self):
        super().__init__()
        self.encoder = SpatialEncoder()
        self.unit = nn.Linear(22, WIDTH)
        self.state = nn.Sequential(nn.Linear(WIDTH * 2 + 6, 64), nn.SiLU())
        self.primitive = nn.Linear(64, 1)
        self.prefix = nn.Sequential(nn.Linear(64 + WIDTH + 5, 48), nn.SiLU())
        self.replan = nn.Linear(48, 1)
        self.mode = nn.Linear(48, 1)
        self.goal = nn.Linear(48, MODES)

    def encode(self, state):
        units = F.silu(self.unit(state["units"].float()))
        alive = state["alive"].unsqueeze(-1)
        pool = (units * alive).sum(1) / alive.sum(1).clamp_min(1)
        grid = self.encoder(state["grid"]).mean(1)
        joint = self.state(torch.cat((grid, pool, state["team"].float()), dim=-1))
        return joint, units

    def forward(self, state):
        joint, _ = self.encode(state)
        return self.primitive(joint).squeeze(-1)

    def conditional(self, encoded, uid, claims):
        joint, units = encoded
        available = claims >= 0
        xy = torch.stack((claims.clamp_min(0) // 24, claims.clamp_min(0) % 24), -1) / 23
        average = (xy * available.unsqueeze(-1)).sum(1) / available.sum(1, keepdim=True).clamp_min(1)
        count = available.sum(1, keepdim=True) / 16
        area = (xy.square() * available.unsqueeze(-1)).sum(1) / available.sum(1, keepdim=True).clamp_min(1)
        hidden = self.prefix(torch.cat((joint, units[:, uid], count, average, area), -1))
        return self.replan(hidden).squeeze(-1), self.mode(hidden).squeeze(-1), self.goal(hidden)
