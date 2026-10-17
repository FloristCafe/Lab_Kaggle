"""Multiclass Focal Loss and provably optimistic destination constraints.

The existing NPZ has tile-summed own energy, not individual energy or inferred
match-specific costs. Bounds use official minimum move cost 1, maximum energy
400 and maximum recovery 20; unknown nebula drain has lower bound 0. Never
pretend a deep nebula is impossible based on current energy alone.
"""

import math
from functools import lru_cache

import torch
from torch.nn import functional as F


@lru_cache(maxsize=8)
def _grid(device):
    axis = torch.arange(24, device=device)
    return torch.meshgrid(axis, axis, indexing="ij")


def optimistic_destination_mask(features, horizon=12):
    """Allow every destination possible under optimistic legal energy bounds.

    Movement precedes energy recovery. Enumerate staying and four first-step
    moves using current visible asteroids and current visible energy bounds.
    After step one, optimistically allow +20 energy per step, zero nebula drain,
    no obstacles or opponent damage. A zero-energy branch needs one extra wait.
    Thus masks never use future observations, target labels or projected maps.
    For 12-step targets this safely masks out-of-range or energy-delay cells.
    """
    if features.ndim != 4 or features.shape[1:] != (28, 24, 24) or horizon < 2:
        raise ValueError("expected [K,28,24,24] and horizon >=2")
    count = len(features)
    origin = features[:, 25].flatten(1).argmax(1)
    x0, y0 = origin // 24, origin % 24
    ids = torch.arange(count, device=features.device)
    # Energies are nonnegative and capped at 400 per unit. For tile sums <=400,
    # FP16 quantization error after scaling is <1 energy, so ceil is an upper
    # bound on each integer unit energy; larger sums conservatively cap at 400.
    energy = (features[ids, 14, x0, y0].float() * 6400).ceil().clamp(0, 400)
    gx, gy = _grid(str(features.device))

    def gain_upper(x, y):
        observed = features[ids, 0, x, y] > 0.5
        current = (features[ids, 9, x, y].float() * 20).ceil().clamp(-20, 20)
        return torch.where(observed, current, torch.full_like(current, 20))

    distance = (gx[None] - x0[:, None, None]).abs() + (gy[None] - y0[:, None, None]).abs()
    after_wait = (energy + gain_upper(x0, y0)).clamp(0, 400)
    wait_radius = horizon - 1 - (after_wait < 1).long()
    allowed = distance <= wait_radius[:, None, None]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x0 + dx, y0 + dy
        inside = (nx >= 0) & (nx < 24) & (ny >= 0) & (ny < 24)
        sx, sy = nx.clamp(0, 23), ny.clamp(0, 23)
        first_move = inside & (energy >= 1) & (features[ids, 5, sx, sy] < 0.5)
        after_move = (energy - 1 + gain_upper(sx, sy)).clamp(0, 400)
        radius = horizon - 1 - (after_move < 1).long()
        remaining = (gx[None] - nx[:, None, None]).abs() + (gy[None] - ny[:, None, None]).abs()
        allowed |= first_move[:, None, None] & (remaining <= radius[:, None, None])
    return allowed[:, None]


def apply_destination_mask(logits, features, target=None, horizon=12):
    if logits.shape != (len(features), 1, 24, 24):
        raise ValueError("expected [K,1,24,24] logits")
    allowed = optimistic_destination_mask(features, horizon)
    if target is not None and not allowed.flatten(1).gather(1, target[:, None]).all():
        raise ValueError("physical mask excludes a supervised target; inspect energy/timing bounds")
    # FP32 prevents -1e9 overflow in mixed precision and keeps CE/focal stable.
    return logits.float().masked_fill(~allowed, -1e9), allowed


def focal_spatial_loss(logits, target, gamma=2.0, reduction="mean"):
    """-(1-p_t)^gamma log(p_t), over 576 destination classes.

    Focal scales by confidence, not by whether a label equals the anchor. Gamma
    zero equals CE. No undisclosed class weighting or label removal is applied.
    """
    if gamma < 0 or not math.isfinite(gamma):
        raise ValueError("focal gamma must be finite and >=0")
    if logits.ndim != 4 or logits.shape[1:] != (1, 24, 24):
        raise ValueError("expected [K,1,24,24] logits")
    if target.shape != (len(logits),) or target.dtype != torch.long:
        raise ValueError("expected K spatial class IDs")
    ce = F.cross_entropy(logits.float().flatten(1), target, reduction="none")
    loss = (-torch.expm1(-ce)).pow(gamma) * ce
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction != "mean":
        raise ValueError("invalid reduction")
    return loss.mean()



def masked_heatmap_mse(prediction, target, valid):
    """Mean pixel MSE averaged over valid units; invalid NaNs are neutral.

    Predictions are sigmoid heatmaps [K,1,24,24], targets are peak-1 Gaussian
    [K,24,24]. No CE, spatial softmax, class IDs or class weights are used.
    """
    if prediction.ndim != 4 or prediction.shape[1:] != (1, 24, 24):
        raise ValueError("expected prediction [K,1,24,24]")
    if target.shape != (len(prediction), 24, 24):
        raise ValueError("expected Gaussian target [K,24,24]")
    if valid.shape != (len(prediction),) or valid.dtype != torch.bool:
        raise ValueError("expected valid bool [K]")
    if not valid.any():
        return prediction[valid].sum()
    # Select before subtracting: 0 * NaN is still NaN for invalid units.
    errors = torch.nn.MSELoss(reduction="none")(prediction[valid, 0].float(), target[valid].float())
    return errors.mean((1, 2)).mean()


def causal_destination_mask(features, horizon=12):
    """Optional decoding constraint only, independent of regression training."""
    if features.ndim != 4 or features.shape[1:] != (35, 24, 24):
        raise ValueError("expected causal [K,35,24,24]")
    legacy = features.new_zeros((len(features), 28, 24, 24))
    legacy[:, 0] = features[:, 0]
    legacy[:, 5] = features[:, 5]
    legacy[:, 9] = features[:, 6]
    legacy[:, 14] = features[:, 11]
    legacy[:, 25] = features[:, 32]
    return optimistic_destination_mask(legacy, horizon)



def spatial_kl_per_unit(logits, target):
    """KL(target || softmax(logits)), sum over 576 cells per unit, in FP32."""
    if logits.ndim != 4 or logits.shape[1:] != (1, 24, 24):
        raise ValueError("expected raw logits [K,1,24,24]")
    if target.shape != (len(logits), 24, 24):
        raise ValueError("expected Gaussian probabilities [K,24,24]")
    flat_target = target.float().flatten(1)
    if (not torch.isfinite(flat_target).all() or (flat_target < 0).any()
            or not torch.allclose(flat_target.sum(1), flat_target.new_ones(len(target)),
                                  atol=1e-5, rtol=1e-5)):
        raise ValueError("each valid target must be a finite nonnegative distribution summing to one")
    log_prediction = F.log_softmax(logits.float().flatten(1), dim=1)
    terms = torch.nn.KLDivLoss(reduction="none")(log_prediction, flat_target)
    return terms.sum(1)


def masked_spatial_kl(logits, target, valid):
    """Average per-unit KL over valid units, neutralizing invalid NaNs first."""
    if valid.shape != (len(logits),) or valid.dtype != torch.bool:
        raise ValueError("expected valid bool [K]")
    if not valid.any():
        return logits[valid].float().sum()
    return spatial_kl_per_unit(logits[valid], target[valid]).mean()
