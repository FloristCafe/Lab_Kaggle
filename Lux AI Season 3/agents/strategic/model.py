"""Grouped per-frame encoders, explicit ConvGRU history and factorized unit actor."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .features import UNIT_DIM


def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.GroupNorm(4, out_channels),
        nn.SiLU(),
    )


class HistoryEncoder(nn.Module):
    groups = ((0, 1, 2, 3, 4, 5, 6), (7, 10, 11, 12, 13), (8, 9, 14, 15, 16, 17))

    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([conv(len(group), 8) for group in self.groups])
        self.gates = nn.Conv2d(24 + 32, 64, 3, padding=1)
        self.candidate = nn.Conv2d(24 + 32, 32, 3, padding=1)
        self.stem = conv(34, 32)
        self.down12 = conv(32, 48)
        self.down6 = conv(48, 64)
        self.up12 = conv(64 + 48, 48)
        self.up24 = conv(48 + 32, 32)
        axis = torch.linspace(-1, 1, 24)
        self.register_buffer(
            "coordinates", torch.stack(torch.meshgrid(axis, axis, indexing="ij"))[None]
        )

    def forward(self, history):
        B, T, C, X, Y = history.shape
        if (T, C, X, Y) != (3, 18, 24, 24):
            raise ValueError("expected [B,3,18,24,24] causal maps")
        hidden = history.new_zeros((B, 32, 24, 24))
        for t in (2, 1, 0):  # oldest -> current; absent frames are zero padded
            encoded = torch.cat(
                [
                    encoder(history[:, t, list(group)])
                    for encoder, group in zip(self.encoders, self.groups, strict=True)
                ],
                1,
            )
            reset, update = (
                self.gates(torch.cat((encoded, hidden), 1)).sigmoid().chunk(2, 1)
            )
            candidate = self.candidate(torch.cat((encoded, reset * hidden), 1)).tanh()
            hidden = (1 - update) * hidden + update * candidate
        x24 = self.stem(torch.cat((hidden, self.coordinates.expand(B, -1, -1, -1)), 1))
        x12 = self.down12(F.avg_pool2d(x24, 2))
        x6 = self.down6(F.avg_pool2d(x12, 2))
        x6 = x6 + x6.mean((2, 3), keepdim=True)
        x12 = self.up12(
            torch.cat((F.interpolate(x6, size=(12, 12), mode="nearest"), x12), 1)
        )
        return self.up24(
            torch.cat((F.interpolate(x12, size=(24, 24), mode="nearest"), x24), 1)
        )


class StrategicActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = HistoryEncoder()
        self.unit_embedding = nn.Sequential(
            nn.Linear(32 + UNIT_DIM, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU()
        )
        self.mode_head = nn.Linear(32, 5)
        self.mode_queries = nn.Linear(32, 5 * 32)
        self.goal_keys = nn.Conv2d(32, 32, 1)
        self.goal_bias = nn.Conv2d(32, 5, 1)
        self.cost_weight = nn.Linear(32, 5 * 2)
        self.previous_weight = nn.Parameter(torch.zeros(5))
        axis = torch.arange(24)
        gx, gy = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("gx", gx.float())
        self.register_buffer("gy", gy.float())

    def forward(self, history, positions, units, alive):
        spatial = self.encoder(history)
        B, K, _ = positions.shape
        pos = positions.clamp(0, 23)
        ids = pos[:, :, 0] * 24 + pos[:, :, 1]
        local = (
            spatial.flatten(2)
            .gather(2, ids[:, None].expand(-1, 32, -1))
            .transpose(1, 2)
        )
        query = self.unit_embedding(torch.cat((local, units), -1))
        mode_logits = self.mode_head(query)
        mode_query = self.mode_queries(query).reshape(B, K, 5, 32)
        goal_logits = torch.einsum(
            "bkmd,bdxy->bkmxy", mode_query, self.goal_keys(spatial)
        ) / math.sqrt(32)
        goal_logits = goal_logits + self.goal_bias(spatial)[:, None]
        distance = (self.gx - pos[:, :, 0, None, None]).abs() + (
            self.gy - pos[:, :, 1, None, None]
        ).abs()
        time_lower_bound = distance / 46
        energy_lower_bound = distance * (units[:, :, 19, None, None] * 5) / 400
        costs = torch.stack((time_lower_bound, energy_lower_bound), 2)
        cost_weights = self.cost_weight(query).reshape(B, K, 5, 2)
        goal_logits = goal_logits + torch.einsum(
            "bkmc,bkcxy->bkmxy", cost_weights, costs
        )
        previous = (units[:, :, 8:10] * 23).round().long().clamp(0, 23)
        previous_mask = F.one_hot(
            previous[:, :, 0] * 24 + previous[:, :, 1], 576
        ).reshape(B, K, 24, 24)
        previous_mask = previous_mask * units[:, :, 11, None, None]
        goal_logits = (
            goal_logits
            + previous_mask[:, :, None]
            * self.previous_weight[None, None, :, None, None]
        )
        return mode_logits, goal_logits


class CentralizedCritic(nn.Module):
    """One team value from all friendly queries, legal map/belief and goals.

    Deliberately no privileged replay/engine state. CTDE can be strengthened
    later without changing the Actor information boundary.
    """

    def __init__(self):
        super().__init__()
        self.encoder = HistoryEncoder()
        self.value = nn.Sequential(
            nn.Linear(32 + UNIT_DIM + 2 + 4, 64), nn.SiLU(), nn.Linear(64, 1)
        )

    def forward(self, history, positions, units, alive, team_context=None):
        spatial = self.encoder(history).mean((2, 3))
        mask = alive.float()
        pooled = (units * mask[:, :, None]).sum(1) / mask.sum(
            1, keepdim=True
        ).clamp_min(1)
        counts = torch.stack((mask.sum(1) / 16, units[:, :, 2].sum(1) / 16), 1)
        if team_context is None:
            team_context = units.new_zeros((len(units), 4))
        return self.value(
            torch.cat((spatial, pooled, counts, team_context), 1)
        ).squeeze(-1)


def actor_input(batch):
    return (batch["history"], batch["positions"], batch["units"], batch["alive"])


def supervised_loss(mode_logits, goal_logits, batch):
    valid = batch["valid"] & batch["alive"]
    if not valid.any():
        return mode_logits.sum() * 0, dict(mode_loss=0.0, goal_loss=0.0)
    mode = batch["mode"][valid]
    ml = F.cross_entropy(mode_logits[valid].float(), mode, reduction="none")
    all_goals = goal_logits[valid].float()
    goals = all_goals[torch.arange(len(mode), device=mode.device), mode].flatten(1)
    region = batch["regions"][valid].flatten(1)
    if not region.any(1).all():
        raise ValueError("valid sample without strategic anchor region")
    logp = F.log_softmax(goals, 1)
    # Multi-positive NLL equals hard CE for a single anchor, not Gaussian KL.
    gl = -torch.logsumexp(logp.masked_fill(~region, -torch.inf), 1)
    weight = batch["confidence"][valid].float()
    mode_loss = (ml * weight).sum() / weight.sum().clamp_min(1e-6)
    goal_loss = (gl * weight).sum() / weight.sum().clamp_min(1e-6)
    return mode_loss + goal_loss, dict(
        mode_loss=mode_loss.detach().item(), goal_loss=goal_loss.detach().item()
    )


def biased_logits(mode_logits, goal_logits, units, strength=0.0):
    if not strength:
        return mode_logits, goal_logits
    previous = (units[:, :, 8:10] * 23).round().long().clamp(0, 23)
    mask = F.one_hot(previous[:, :, 0] * 24 + previous[:, :, 1], 576).reshape(
        *previous.shape[:2], 24, 24
    )
    return mode_logits, goal_logits + strength * mask[:, :, None] * units[
        :, :, 11, None, None, None
    ]


def choose_actions(mode_logits, goal_logits, greedy=False):
    B, K, M = mode_logits.shape
    flat_goal = goal_logits.float().flatten(-2)
    mode_logp = F.log_softmax(mode_logits.float(), -1)
    goal_logp = F.log_softmax(flat_goal, -1)
    if greedy:
        joint = (mode_logp[:, :, :, None] + goal_logp).flatten(-2).argmax(-1)
        modes, goals = joint // 576, joint % 576
    else:
        modes = torch.distributions.Categorical(logits=mode_logits.float()).sample()
        selected = flat_goal.gather(
            2, modes[:, :, None, None].expand(-1, -1, 1, 576)
        ).squeeze(2)
        goals = torch.distributions.Categorical(logits=selected).sample()
    logprob = mode_logp.gather(-1, modes[:, :, None]).squeeze(-1)
    conditional = goal_logp.gather(
        2, modes[:, :, None, None].expand(-1, -1, 1, 576)
    ).squeeze(2)
    logprob = logprob + conditional.gather(-1, goals[:, :, None]).squeeze(-1)
    return modes, goals, logprob


def action_statistics(mode_logits, goal_logits, modes, goals):
    mp = F.log_softmax(mode_logits.float(), -1)
    gp = F.log_softmax(goal_logits.float().flatten(-2), -1)
    conditional = gp.gather(2, modes[:, :, None, None].expand(-1, -1, 1, 576)).squeeze(
        2
    )
    logprob = mp.gather(-1, modes[:, :, None]).squeeze(-1) + conditional.gather(
        -1, goals[:, :, None]
    ).squeeze(-1)
    entropy = -(mp.exp() * mp).sum(-1) + (mp.exp() * (-(gp.exp() * gp).sum(-1))).sum(-1)
    return logprob, entropy
