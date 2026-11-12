"""Passive attribution: proposal != acceptance != routed target != movement."""

import copy
from collections import Counter

import numpy as np


class Journal:
    def __init__(self, side, context=None):
        self.side = side
        self.context = copy.deepcopy(
            context or {"source": "synthetic_or_unbound_engineering"}
        )
        self.decisions = []
        self.primitives = []
        self.current = {}
        self.pending = None

    def decision(self, record):
        record = copy.deepcopy(record)
        record["case"] = copy.deepcopy(self.context)
        self.decisions.append(record)
        for row in record["rows"]:
            if row["alive"] and row["reviewed"]:
                self.current[row["uid"]] = row

    def begin(self, frame, manager, route, actions):
        if self.pending is not None:
            raise RuntimeError("unclosed primitive attribution")
        start = int(frame["step"])
        rows = []
        for uid in np.flatnonzero(frame["alive"]):
            uid = int(uid)
            valid = bool(manager.valid[uid])
            goal = tuple(map(int, manager.goals[uid]))
            mode = int(manager.modes[uid])
            pos = tuple(map(int, frame["positions"][uid]))
            node = self.current.get(uid)
            if node is not None and node["step"] == start:
                if valid and (
                    goal[0] * 24 + goal[1] != node["selected_goal"]
                    or mode != node["B0_mode"]
                ):
                    raise RuntimeError(
                        "GoalManager did not accept the declared unchanged Mode/selected Goal"
                    )
                node["manager_received_goal"] = list(goal) if valid else None
                node["manager_received_mode"] = mode if valid else None
            target = route["targets"].get(uid)
            redirected = valid and target != goal
            sap = int(actions[uid, 0]) == 5
            rows.append(
                {
                    "case": copy.deepcopy(self.context),
                    "side": self.side,
                    "step": start,
                    "uid": uid,
                    "decision_node": None if node is None else node["node_id"],
                    "position_before": list(pos),
                    "manager_valid": valid,
                    "manager_goal": list(goal) if valid else None,
                    "mode": mode,
                    "Router_target": list(target) if target is not None else None,
                    "Router_action": route["actions"][uid].tolist(),
                    "executed_action": actions[uid].tolist(),
                    "tactical_target_redirected": bool(redirected),
                    "SAP_selected": sap,
                    "tactical_override": bool(redirected or sap),
                    "tactical_state": route.get("tactical_states", {}).get(uid),
                    "distance_before": abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
                    if valid
                    else None,
                }
            )
        self.pending = rows

    def finish(self, obs):
        if self.pending is None:
            raise RuntimeError(
                "primitive observation without matching controller action"
            )
        step = int(obs["steps"])
        reset = step > 0 and int(obs["match_steps"]) == 0
        mask = np.asarray(obs["units_mask"], bool)[self.side]
        positions = np.asarray(obs["units"]["position"])[self.side]
        energy = np.asarray(obs["units"]["energy"])[self.side]
        for row in self.pending:
            uid = row["uid"]
            pos = tuple(map(int, positions[uid]))
            alive = bool(mask[uid] and energy[uid] >= 0 and not reset)
            if step != row["step"] + 1:
                raise RuntimeError("attribution time mismatch")
            # Public one-step displacement >1 identifies identity reuse/respawn;
            # don't label it as navigation progress toward an old target.
            identity_jump = (
                alive
                and sum(abs(pos[i] - row["position_before"][i]) for i in (0, 1)) > 1
            )
            valid = bool(row["manager_valid"] and alive and not identity_jump)
            goal = row["manager_goal"]
            distance = sum(abs(pos[i] - goal[i]) for i in (0, 1)) if valid else None
            row.update(
                end_step=step,
                position_after=list(pos) if alive else None,
                alive_after=alive,
                mini_reset=reset,
                identity_jump=bool(identity_jump),
                progress_eligible=bool(valid and row["distance_before"] > 0),
                distance_after=distance,
                advanced=bool(
                    valid
                    and row["distance_before"] > 0
                    and distance < row["distance_before"]
                ),
                actual_moved=bool(alive and pos != tuple(row["position_before"])),
                reached=bool(valid and distance == 0),
            )
            self.primitives.append(row)
        self.pending = None
        if reset:
            self.current.clear()
        else:
            for uid in list(self.current):
                if not mask[uid] or energy[uid] < 0:
                    self.current.pop(uid, None)

    def close_option(self, manager, step):
        for uid, row in self.current.items():
            goal = manager.goals[uid]
            row["checked_through_step"] = int(step)
            row["goal_persisted_at_option_end"] = bool(
                manager.valid[uid]
                and int(manager.modes[uid]) == row["B0_mode"]
                and int(goal[0]) * 24 + int(goal[1]) == row["selected_goal"]
            )

    def summary(self):
        rows = [r for d in self.decisions for r in d["rows"] if r["eligible_replan"]]
        valid = [r for r in rows if r["V2_proposal_present"] and r["coordinate_valid"]]
        attempted = [r for r in rows if r["attempted"]]
        accepted = [r for r in rows if r["accepted"]]
        intents = [r for r in self.primitives if r["manager_valid"]]
        progress = [r for r in self.primitives if r["progress_eligible"]]

        def rate(n, d):
            return {"numerator": n, "denominator": d, "rate": n / d if d else None}

        return {
            "eligible_replan_nodes": len(rows),
            "V2_proposal_coverage": rate(
                sum(r["V2_proposal_present"] for r in rows), len(rows)
            ),
            "legal_acceptance": rate(len(accepted), len(attempted)),
            "effective_goal_changes": sum(r["effective_goal_change"] for r in accepted),
            "Mode_condition_incompatibility": rate(
                sum(r["mode_compatible"] is False for r in valid), len(valid)
            ),
            "reservation_conflict": rate(
                sum(r["reservation_conflict"] for r in valid), len(valid)
            ),
            "tactical_override": rate(
                sum(r["tactical_override"] for r in intents), len(intents)
            ),
            "actual_Goal_progress": rate(
                sum(r["advanced"] for r in progress), len(progress)
            ),
            "fallback_reason_counts": dict(
                Counter(r["reason"] for r in attempted if not r["accepted"])
            ),
            "scope": "review-node proposal/acceptance vs primitive-unit execution; overlap reasons retained; no rates imply task success or causality",
        }


class RouterTap:
    def __init__(self, delegate):
        self.delegate = delegate
        self.last = None

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def route(self, units, targets, blocked, priorities=None):
        actions = self.delegate.route(units, targets, blocked, priorities)
        self.last = {"targets": copy.deepcopy(targets), "actions": actions.copy()}
        return actions


class TacticalTap:
    def __init__(self, delegate, journal):
        self.delegate = delegate
        self.journal = journal
        self.tap = RouterTap(delegate.router)
        delegate.router = self.tap

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def act(self, frame, manager):
        actions, info = self.delegate.act(frame, manager)
        if self.tap.last is None:
            raise RuntimeError("controller bypassed the unchanged Router")
        route = dict(self.tap.last)
        route["tactical_states"] = {
            int(u): int(self.delegate.states[u]) for u in np.flatnonzero(frame["alive"])
        }
        self.journal.begin(frame, manager, route, actions)
        return actions, info
