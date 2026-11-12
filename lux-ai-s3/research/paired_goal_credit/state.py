"""Deterministic state hashing; tensors and arrays are consumed synchronously."""
import hashlib
import sys
from collections import deque
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from fractions import Fraction

import numpy as np


def stable(value, ancestors=()):
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().numpy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Fraction):
        return ("Fraction", value.numerator, value.denominator)
    if value is None or isinstance(value, (str, bytes, int, bool, float, np.ndarray)):
        return value
    if id(value) in ancestors:
        raise RuntimeError("cyclic state in audit")
    chain = (*ancestors, id(value))
    if isinstance(value, Mapping):
        return {k: stable(v, chain) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return (type(value).__name__, [stable(v, chain) for v in value])
    if isinstance(value, (set, frozenset)):
        return (type(value).__name__, sorted((stable(v, chain) for v in value), key=repr))
    if is_dataclass(value):
        return {f.name: stable(getattr(value, f.name), chain) for f in fields(value)}
    if hasattr(value, "__dict__"):
        return (type(value).__qualname__, stable(vars(value), chain))
    raise TypeError(f"unsupported audit state: {type(value)!r}")


def digest(value):
    h = hashlib.sha256()

    def add(data):
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)

    def visit(item):
        add(type(item).__name__.encode())
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise TypeError("object-dtype array cannot be audited")
            add(str((item.shape, item.dtype.str)).encode())
            add(item.tobytes())
        elif isinstance(item, dict):
            add(str(len(item)).encode())
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            add(str(len(item)).encode())
            for child in item:
                visit(child)
        else:
            add(repr(item).encode())

    visit(stable(value))
    return h.hexdigest()
