"""Isolated unmodified native V2 on consecutive team-public observations only.

No environment imports. The native Router runs in the shadow but its actions are
never applied. Its planner/solver/decay side effects belong to this V2 instance.
The RNG sandbox requires one serial controller per process (no concurrent draws).
"""

import builtins
import copy
import importlib
import importlib.util
import numbers
import random
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import torch

from research.mappo_v5_2_learning_curve.store import tree_hash

from .provenance import ROOT

_RNG_LOCK = threading.RLock()
_AGENT_CLASS = None
FIELDS = frozenset(
    (
        "units",
        "units_mask",
        "sensor_mask",
        "map_features",
        "relic_nodes",
        "relic_nodes_mask",
        "team_points",
        "team_wins",
        "steps",
        "match_steps",
    )
)
PUBLIC_CONFIG = frozenset(
    (
        "map_width",
        "map_height",
        "max_units",
        "max_steps_in_match",
        "match_count_per_episode",
        "unit_move_cost",
        "unit_sap_cost",
        "unit_sap_range",
        "unit_sensor_range",
    )
)


def integer(x, low, high):
    if (
        isinstance(x, (bool, np.bool_))
        or not isinstance(x, numbers.Integral)
        or not low <= int(x) <= high
    ):
        raise ValueError("strict integer/range required")
    return int(x)


def rng_snapshot():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": [s.clone() for s in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_initialized()
        else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


class PrivateRNG:
    def __init__(self):
        self.state = {
            "python": random.Random(0).getstate(),
            "numpy": np.random.RandomState(0).get_state(),
            "torch": torch.Generator(device="cpu").manual_seed(0).get_state(),
            "cuda": None,
        }

    @contextmanager
    def scope(self):
        with _RNG_LOCK:
            external = rng_snapshot()
            if external["cuda"] is not None and self.state["cuda"] is None:
                self.state["cuda"] = [s.clone() for s in external["cuda"]]
            restore_rng(self.state)
            try:
                yield
            finally:
                self.state = rng_snapshot()
                restore_rng(external)
                if tree_hash(rng_snapshot()) != tree_hash(external):
                    raise RuntimeError(
                        "shadow RNG sandbox failed to restore external streams"
                    )


def native_agent_class():
    global _AGENT_CLASS
    if _AGENT_CLASS is None:
        # Original file uses bare imports. Resolve only those imports locally;
        # do not change sys.path/sys.modules or the worker's Agent bindings.
        path = ROOT / "agents/v2/agent.py"
        spec = importlib.util.spec_from_file_location(
            "research.mappo_v5_2_goal_adapter._native_v2", path
        )
        native = importlib.util.module_from_spec(spec)
        original = builtins.__import__

        def local_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and name in {
                "dispatcher",
                "equation_solver",
                "obs_parser",
                "router",
            }:
                return importlib.import_module("agents.v2." + name)
            return original(name, globals, locals, fromlist, level)

        scope = dict(vars(builtins))
        scope["__import__"] = local_import
        native.__dict__["__builtins__"] = scope
        spec.loader.exec_module(native)
        _AGENT_CLASS = native.Agent
    return _AGENT_CLASS


def public_observation(obs, side):
    if not isinstance(obs, dict) or set(obs) != FIELDS:
        raise ValueError("only the exact pinned team-public EnvObs schema is accepted")
    if set(obs["units"]) != {"position", "energy"} or set(obs["map_features"]) != {
        "tile_type",
        "energy",
    }:
        raise ValueError("hidden/unknown nested fields")
    out = copy.deepcopy(obs)
    for key, shape in [("units_mask", (2, 16)), ("sensor_mask", (24, 24))]:
        a = np.asarray(out[key])
        if a.shape != shape or not np.isin(a, (0, 1)).all():
            raise ValueError("invalid public mask")
        out[key] = a.astype(bool, copy=True)
    pos = np.asarray(out["units"]["position"])
    energy = np.asarray(out["units"]["energy"])
    if (
        pos.shape != (2, 16, 2)
        or energy.shape != (2, 16)
        or pos.dtype.kind not in "iu"
        or energy.dtype.kind not in "iu"
    ):
        raise ValueError("invalid public unit tensors")
    mask = out["units_mask"]
    if (
        ((pos[mask] < 0) | (pos[mask] >= 24)).any()
        or (pos[~mask] != -1).any()
        or (energy[~mask] != -1).any()
    ):
        raise ValueError("unit visibility redaction missing")
    for uid in np.flatnonzero(mask[1 - side]):
        if not out["sensor_mask"][tuple(pos[1 - side, uid])]:
            raise ValueError("opponent unit outside own public visibility")
    out["units"] = {"position": pos.copy(), "energy": energy.copy()}
    for key in ["tile_type", "energy"]:
        a = np.asarray(out["map_features"][key])
        if (
            a.shape != (24, 24)
            or a.dtype.kind not in "iu"
            or (a[~out["sensor_mask"]] != -1).any()
        ):
            raise ValueError("map visibility redaction missing")
        if key == "tile_type" and not np.isin(a[out["sensor_mask"]], (0, 1, 2)).all():
            raise ValueError("invalid public terrain")
        out["map_features"][key] = a.copy()
    for key in ["team_points", "team_wins"]:
        a = np.asarray(out[key])
        if a.shape != (2,) or a.dtype.kind not in "iu" or (a < 0).any():
            raise ValueError("invalid public score")
        out[key] = a.copy()
    rmask = np.asarray(out["relic_nodes_mask"])
    rpos = np.asarray(out["relic_nodes"])
    if (
        rmask.shape != (6,)
        or rpos.shape != (6, 2)
        or not np.isin(rmask, (0, 1)).all()
        or rpos.dtype.kind not in "iu"
    ):
        raise ValueError("invalid relic observation")
    rmask = rmask.astype(bool)
    if (rpos[~rmask] != -1).any() or ((rpos[rmask] < 0) | (rpos[rmask] >= 24)).any():
        raise ValueError("relic redaction missing")
    for xy in rpos[rmask]:
        if not out["sensor_mask"][tuple(xy)]:
            raise ValueError("relic outside public visibility")
    out["relic_nodes_mask"] = rmask.copy()
    out["relic_nodes"] = rpos.copy()
    out["steps"] = integer(obs["steps"], 0, 505)
    out["match_steps"] = integer(obs["match_steps"], 0, 100)
    if out["match_steps"] != out["steps"] % 101:
        raise ValueError("primitive/match clock mismatch")
    if int(out["team_wins"].sum()) != out["steps"] // 101:
        raise ValueError("public win settlement clock mismatch")
    return out


@dataclass(frozen=True)
class Proposal:
    step: int
    side: int
    targets: dict
    positions: dict
    input_sha256: str


class ShadowV2:
    def __init__(self, side, config):
        self.side = integer(side, 0, 1)
        if set(config) != PUBLIC_CONFIG:
            raise ValueError("only pinned public environment configuration allowed")
        self.config = {k: integer(v, 0, 10000) for k, v in config.items()}
        if any(
            self.config[k] != v
            for k, v in [
                ("map_width", 24),
                ("map_height", 24),
                ("max_units", 16),
                ("max_steps_in_match", 100),
                ("match_count_per_episode", 5),
            ]
        ):
            raise ValueError("only frozen 24x24/505-step protocol supported")
        self.rng = PrivateRNG()
        with self.rng.scope():
            self._agent = native_agent_class()(
                f"player_{side}", copy.deepcopy(self.config)
            )
        self.last_step = None
        self.failed = False
        self.updates = 0
        self.events = []
        self._proposal = None

    def observe(self, obs):
        if self.failed:
            raise RuntimeError("failed shadow cannot continue")
        clean = public_observation(obs, self.side)
        step = clean["steps"]
        if step != (0 if self.last_step is None else self.last_step + 1):
            raise ValueError(
                "observation must be consecutive and consumed exactly once from step0"
            )
        digest = tree_hash(clean)
        try:
            with self.rng.scope():
                self._agent.act(step, clean)
                trace = copy.deepcopy(self._agent.last_trace)
            if tree_hash(clean) != digest:
                raise RuntimeError("native V2 mutated its owned public observation")
            targets = {int(uid): tuple(xy) for uid, xy in trace["targets"].items()}
            positions = {int(uid): tuple(xy) for uid, xy in trace["positions"].items()}
            if set(targets) != set(positions) or any(not 0 <= u < 16 for u in targets):
                raise RuntimeError("incomplete native UID-target allocation")
            for target in targets.values():
                if len(target) != 2 or any(
                    not isinstance(a, numbers.Integral) or not 0 <= int(a) < 24
                    for a in target
                ):
                    raise RuntimeError("native V2 emitted invalid target")
            self._proposal = Proposal(
                step,
                self.side,
                MappingProxyType(targets),
                MappingProxyType(positions),
                digest,
            )
            self.events.append(
                {
                    "step": step,
                    "match_step": clean["match_steps"],
                    "mini_reset": step > 0 and clean["match_steps"] == 0,
                    "uids": sorted(targets),
                    "goal_count": len(targets),
                    "yielded_uid": trace["yielded_unit"],
                    "parser_reason": trace["parser_reason"],
                    "rank": trace["rank"],
                    "external_rng_preserved": True,
                    "future_observations_used": False,
                }
            )
            self.last_step = step
            self.updates += 1
        except BaseException:
            self.failed = True
            raise
        return self._proposal

    def proposals(self, step):
        if self.failed or self.last_step != step or self._proposal is None:
            raise ValueError("stale/future or missing shadow proposal")
        return self._proposal

    def memory_fingerprint(self):
        a = self._agent
        return tree_hash(
            {
                "tile_type": a.tile_type,
                "last_seen": a.last_seen,
                "known_relics": a.known_relics,
                "decay": a.dispatcher.decay,
                "equations": repr(a.solver.equations),
                "basis": repr(a.solver._basis),
                "dynamic_pool": repr(a.solver.dynamic_pool),
                "values": a.solver.values,
                "rank": a.solver.rank,
                "parser": [
                    a.parser.last_step,
                    a.parser.last_match_step,
                    a.parser.last_points,
                    a.parser.static_epoch,
                ],
            }
        )
