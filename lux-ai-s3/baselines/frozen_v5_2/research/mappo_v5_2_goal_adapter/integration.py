"""Composition hooks only. Caller supplies an existing environment/bridge.

This module never creates a worker or runs a case, and supplies no evaluation CLI.
"""

import copy

from agents.mappo_v5.learning import device
from research.mappo_v5_2_learning_curve.store import tree_hash

from .adapter import decode
from .diagnostics import Journal, TacticalTap
from .shadow import ShadowV2, public_observation


class PublicObservationTap:
    def __init__(self, delegate, owner):
        self.delegate = delegate
        self.owner = owner

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def request(self, **kw):
        response = self.delegate.request(**kw)
        if kw["op"] == "reset":
            self.owner.journal = Journal(
                int(kw["side"]),
                {
                    "root": int(kw["seed"]),
                    "side": int(kw["side"]),
                    "game_index": len(self.owner.game_attempts),
                    "opponent": kw["opponent"],
                },
            )
            self.owner.shadow = ShadowV2(int(kw["side"]), response["env_cfg"])
            self.owner.shadow.observe(response["obs"])
        elif kw["op"] == "step":
            self.owner.shadow.observe(response["obs"])
            self.owner.journal.finish(
                public_observation(response["obs"], self.owner.shadow.side)
            )
        return response


class GoalAdapterEnv:
    def __init__(self, env, *, variant):
        if variant not in ("C", "Bprime"):
            raise ValueError("unknown adapter arm")
        if env.config["scheduler"] != "B0":
            raise ValueError("B0 execution only")
        self.env = env
        self.variant = variant
        self.shadow = None
        self.journal = None
        self.pending_action_hash = None
        self.game_attempts = []
        env.bridge = PublicObservationTap(env.bridge, self)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        if self.journal is not None:
            self.game_attempts.append(
                {
                    "case": copy.deepcopy(self.journal.context),
                    "shadow_events": copy.deepcopy(self.shadow.events),
                    "decisions": copy.deepcopy(self.journal.decisions),
                    "primitives": copy.deepcopy(self.journal.primitives),
                }
            )
        packet = self.env.reset(*args, **kwargs)
        self.env.tactical = TacticalTap(self.env.tactical, self.journal)
        self.pending_action_hash = None
        return packet

    def decide(self, actor, packet):
        if self.pending_action_hash is not None:
            raise RuntimeError("previous reviewed action not yet executed")
        action, entropy, record = decode(
            actor,
            device(packet, next(actor.parameters()).device),
            self.env.config,
            self.shadow,
            variant=self.variant,
        )
        self.journal.decision(record)
        self.pending_action_hash = tree_hash(action)
        return action, entropy, record

    def step(self, action, max_steps=None):
        if self.pending_action_hash != tree_hash(action):
            raise ValueError("execute exactly the decoded/adapted action once")
        result = self.env.step(action, max_steps=max_steps)
        self.pending_action_hash = None
        self.journal.close_option(self.env.manager, self.env.frame["step"])
        return result

    def diagnostics(self):
        if self.journal is None:
            raise RuntimeError("reset before diagnostics")
        return copy.deepcopy(
            {
                "case": self.journal.context,
                "variant": self.variant,
                "shadow_events": self.shadow.events,
                "decisions": self.journal.decisions,
                "primitives": self.journal.primitives,
                "metrics": self.journal.summary(),
            }
        )
