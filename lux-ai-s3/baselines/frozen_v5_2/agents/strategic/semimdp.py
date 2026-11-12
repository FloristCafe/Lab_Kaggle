"""Duration-correct team SMDP returns and per-agent clipped MAPPO surrogate."""

import numpy as np
import torch


def smdp_gae(
    rewards,
    durations,
    values,
    next_values,
    terminals,
    continues,
    gamma=0.99,
    lam=0.95,
    primitive_lambda=True,
):
    arrays = list(
        map(np.asarray, (rewards, durations, values, next_values, terminals, continues))
    )
    rewards, durations, values, next_values, terminals, continues = arrays
    if any(a.shape != rewards.shape for a in arrays) or (durations < 1).any():
        raise ValueError(
            "one positive duration and value/flags per team option required"
        )
    advantages = np.zeros_like(rewards, dtype=np.float64)
    carry = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        discount = gamma ** int(durations[t])
        delta = rewards[t] + discount * (not terminals[t]) * next_values[t] - values[t]
        trace = (
            (gamma * lam) ** int(durations[t]) if primitive_lambda else discount * lam
        )
        carry = delta + trace * (not terminals[t]) * bool(continues[t]) * carry
        advantages[t] = carry
    return advantages, advantages + values


def clipped_actor_loss(new_logprob, old_logprob, advantages, active, clip=0.2):
    if new_logprob.shape != old_logprob.shape or active.shape != new_logprob.shape:
        raise ValueError("per-unit behavior likelihoods and start-active mask required")
    ratio = (new_logprob - old_logprob).exp()
    raw = ratio * advantages[:, None]
    clipped = ratio.clamp(1 - clip, 1 + clip) * advantages[:, None]
    count = active.sum(1).clamp_min(1)
    per_epoch = (torch.minimum(raw, clipped) * active).sum(1) / count
    valid_epochs = active.any(1)
    loss = (
        -per_epoch[valid_epochs].mean() if valid_epochs.any() else new_logprob.sum() * 0
    )
    fraction = (
        ((ratio - 1).abs() > clip)[active].float().mean()
        if active.any()
        else ratio.sum() * 0
    )
    return loss, fraction
