"""Original decision-indexed random variates; no environment/global RNG mutation."""
import hashlib
import json

import torch

def uniform(domain, *key):
    encoded = json.dumps(["PGC01", domain, *key], separators=(",", ":")).encode()
    # 52 bits with midpoint offsets, strictly inside (0, 1) even in binary64.
    return ((int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") >> 12) + 0.5) * 2**-52

def inverse(probs, draw):
    p = probs.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise RuntimeError("invalid sampling probabilities")
    cdf = p.cumsum(0)
    index = int(torch.searchsorted(cdf, draw * cdf[-1], right=True))
    if index >= len(p) or p[index] <= 0:
        raise RuntimeError("inverse CDF returned impossible outcome")
    return index


def fixed_pair(probabilities, legal, context, step, uid):
    """One immutable pair from the original conditional distribution; duplicates stay."""
    probabilities=probabilities.detach().to("cpu",torch.float64)
    legal=torch.as_tensor(legal,dtype=torch.bool,device="cpu")
    if probabilities.ndim!=1 or legal.shape!=probabilities.shape or int(legal.sum())<2:
        raise RuntimeError("focal node requires at least two legal Goals")
    if not torch.isfinite(probabilities).all() or (probabilities<0).any() or probabilities.sum()<=0:
        raise RuntimeError("invalid conditional probabilities")
    if (probabilities[~legal]!=0).any():
        raise RuntimeError("conditional distribution has mass outside legal support")
    return [inverse(probabilities,uniform("pair",context,int(step),int(uid),arm)) for arm in "AB"]

def continuation_draw(context, repeat, step, uid, head):
    if repeat not in (0,1) or head not in ("R","Mode","Goal"):
        raise RuntimeError("invalid registered continuation identity")
    # Arm is deliberately absent: A/B share a variate within a repeat, not actions.
    return uniform("continuation",context,repeat,int(step),int(uid),head)

def verify_fixed_node(expected, actual):
    from .state import digest
    required={"step","uid","R","mode","legal","owners","earlier","packet","pair"}
    if not required<=expected.keys() or not required<=actual.keys():
        raise RuntimeError("incomplete frozen decision certificate")
    if digest(expected)!=digest(actual):
        raise RuntimeError("focal state/R/Mode/support/pair differs from committed prefix")
