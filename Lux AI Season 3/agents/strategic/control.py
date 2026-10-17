"""Deterministic configurable tactical overrides around unchanged V2 reservations."""

from collections import Counter

import numpy as np

from agents.v2.router import Router

STATES = ("NAVIGATE", "RALLY", "ENGAGE", "RETREAT", "RECHARGE")


class GoalManager:
    def __init__(self, config):
        self.config = config
        self.goals = np.zeros((16, 2), np.int16)
        self.modes = np.zeros(16, np.int8)
        self.ages = np.zeros(16, np.int16)
        self.valid = np.zeros(16, bool)
        self.stats = Counter(assignments=0, changes=0, unchanged_assignments=0)

    def assign(self, modes, goals, alive):
        for uid in range(16):
            if not alive[uid]:
                self.valid[uid] = False
                continue
            xy = np.array([int(goals[uid]) // 24, int(goals[uid]) % 24])
            same = (
                self.valid[uid]
                and self.modes[uid] == modes[uid]
                and np.array_equal(xy, self.goals[uid])
            )
            self.stats["assignments"] += 1
            self.stats["unchanged_assignments"] += bool(same)
            self.stats["changes"] += not same
            self.ages[uid] = self.ages[uid] if same else 0
            self.modes[uid], self.goals[uid], self.valid[uid] = modes[uid], xy, True

    def advance(self, frame):
        self.ages[self.valid] += 1
        self.valid &= frame["alive"]
        if frame["match_step"] == 0:
            self.valid[:] = False

    def invalidated(self, frame):
        return any(
            self.valid[u]
            and self.modes[u] == 0
            and frame["base"][19, *self.goals[u]] > 0.5
            for u in np.flatnonzero(frame["alive"])
        )


class TacticalController:
    def __init__(self, env_cfg, config):
        self.env_cfg, self.cfg = env_cfg, config
        self.router = Router(
            24, 24, 16, env_cfg["unit_move_cost"], horizon=config["router_horizon"]
        )
        self.states = np.zeros(16, np.int8)
        self.state_age = np.zeros(16, np.int16)
        self.stats = Counter({name: 0 for name in STATES})
        self.stats.update(
            dict(
                overrides=0,
                sap_selected=0,
                rally_success=0,
                retreat_success=0,
                retreat_failure=0,
                router_stall=0,
                expected_sap_damage=0.0,
            )
        )
        self.grid = np.stack(np.indices((24, 24)), axis=-1)

    def local(self, frame, uid):
        xy = frame["positions"][uid]
        friends = [
            (u, tuple(frame["positions"][u]), int(frame["energy"][u]))
            for u in np.flatnonzero(frame["alive"])
            if np.abs(frame["positions"][u] - xy).sum() <= self.cfg["local_radius"]
        ]
        enemies = [
            e
            for e in frame["enemy"]
            if np.abs(np.asarray(e[1]) - xy).sum() <= self.cfg["local_radius"]
        ]
        fp = sum(e for _, _, e in friends)
        ep = sum(e for _, _, e in enemies)
        support = sum(
            1
            for u, p, _ in friends
            if u != uid
            and np.abs(np.asarray(p) - xy).sum() <= self.cfg["support_radius"]
        )
        e = int(frame["energy"][uid])
        sap_risk = sum(
            self.env_cfg["unit_sap_cost"]
            for _, p, ee in enemies
            if ee >= self.env_cfg["unit_sap_cost"]
            and max(abs(np.asarray(p) - xy)) <= self.env_cfg["unit_sap_range"]
        ) / max(1, e)
        collision = sum(
            ee for _, p, ee in enemies if np.abs(np.asarray(p) - xy).sum() <= 1
        ) / max(1, fp)
        energy_disadvantage = max(0, ep - fp) / max(1, fp)
        threat = (
            self.cfg["threat_enemy_weight"] * len(enemies)
            + self.cfg["threat_sap_weight"] * sap_risk
            + self.cfg["threat_energy_weight"] * energy_disadvantage
            + self.cfg["threat_collision_weight"] * collision
            - self.cfg["threat_support_weight"] * support
        )
        return friends, enemies, fp / max(1, ep), threat

    def _safe_anchor(self, frame, uid, rally=False):
        xy = frame["positions"][uid]
        distance = np.abs(self.grid - xy).sum(-1)
        enemy_distance = np.full((24, 24), 48.0, np.float32)
        friend_distance = np.full((24, 24), 48.0, np.float32)
        for _, pos, _ in frame["enemy"]:
            enemy_distance = np.minimum(enemy_distance, np.abs(self.grid - pos).sum(-1))
        for other in np.flatnonzero(frame["alive"]):
            if other != uid:
                friend_distance = np.minimum(
                    friend_distance,
                    np.abs(self.grid - frame["positions"][other]).sum(-1),
                )
        safety = (
            enemy_distance
            - self.cfg["escape_cost_weight"] * distance
            + frame["base"][9]
        )
        if rally:
            safety -= friend_distance
        else:
            safety -= (
                self.cfg["threat_support_weight"]
                * friend_distance
                / self.cfg["support_radius"]
            )
        safety[(distance > self.cfg["local_radius"]) | (frame["base"][5] > 0.5)] = -1e6
        for other in np.flatnonzero(frame["alive"]):
            if other != uid:
                safety[tuple(frame["positions"][other])] = -1e6
        return np.asarray(np.unravel_index(safety.argmax(), (24, 24)))

    def _recharge_anchor(self, frame, uid):
        xy = frame["positions"][uid]
        distance = np.abs(self.grid - xy).sum(-1)
        value = frame["base"][9] - self.cfg["escape_cost_weight"] * distance
        value[(frame["base"][5] > 0.5) | (frame["base"][9] <= 0)] = -1e6
        return (
            np.asarray(np.unravel_index(value.argmax(), (24, 24)))
            if value.max() > -1e5
            else xy
        )

    def sap(self, frame, uid, remaining):
        cfg = self.cfg
        energy = int(frame["energy"][uid])
        cost = self.env_cfg["unit_sap_cost"]
        if energy < cost + cfg["safe_energy"] or not frame["enemy"]:
            return None
        xy = frame["positions"][uid]
        candidates = self.grid[
            (np.abs(self.grid - xy).max(-1) <= self.env_cfg["unit_sap_range"])
        ]
        enemy_xy = np.array([p for _, p, _ in frame["enemy"]])
        enemy_energy = np.array([remaining.get(u, e) for u, _, e in frame["enemy"]])
        distance = np.abs(candidates[:, None] - enemy_xy[None]).max(-1)
        damage = np.where(
            distance == 0,
            cost,
            np.where(distance == 1, int(cost * cfg["sap_dropoff_lower_bound"]), 0),
        )
        effective = np.minimum(damage, np.maximum(0, enemy_energy + 1)[None])
        hits = (effective > 0).sum(1)
        kills = ((damage > enemy_energy[None]) & (enemy_energy[None] >= 0)).sum(1)
        # No friendly fire exists in the pinned official environment.
        important = np.array([frame["base"][20, *p] for _, p, _ in frame["enemy"]])
        score = (
            cfg["sap_damage_weight"] * effective.sum(1) / max(1, cost)
            + cfg["sap_kill_weight"] * kills
            + cfg["sap_cluster_weight"] * hits
            + cfg["sap_interrupt_weight"] * ((effective > 0) * important).sum(1)
            - cfg["sap_energy_weight"] * cost / max(1, energy)
            - cfg["sap_survival_weight"]
            * max(0, cfg["safe_energy"] - (energy - cost))
            / max(1, cfg["safe_energy"])
        )
        best = int(score.argmax())
        if score[best] <= cfg["sap_threshold"]:
            return None
        for i, (eid, _, e) in enumerate(frame["enemy"]):
            remaining[eid] = remaining.get(eid, e) - int(damage[best, i])
        return candidates[best] - xy, float(effective[best].sum())

    def act(self, frame, manager):
        targets, priorities, shots = {}, {}, {}
        severe = False
        remaining = {u: e for u, _, e in frame["enemy"]}
        for uid in np.flatnonzero(frame["alive"]):
            friends, enemies, ratio, threat = self.local(frame, uid)
            xy = frame["positions"][uid]
            energy = int(frame["energy"][uid])
            target = manager.goals[uid].copy() if manager.valid[uid] else xy.copy()
            previous_state = self.states[uid]
            goal_value = (
                float(frame["base"][11, *target])
                if manager.modes[uid] in (0, 2)
                else 0.0
            )
            escape_cost = self.env_cfg["unit_move_cost"] + max(
                0, -float(frame["base"][9, *xy]) * 20
            )
            risk = (
                threat
                + self.cfg["escape_cost_weight"] * escape_cost / max(1, energy)
                - self.cfg["objective_value_weight"] * goal_value
            )
            if enemies and risk > self.cfg["retreat_threat"]:
                state, target = 3, self._safe_anchor(frame, uid)
                severe = True
            elif energy < self.cfg["recharge_below"] or (
                previous_state == 4 and energy < self.cfg["recharge_target"]
            ):
                state, target = 4, self._recharge_anchor(frame, uid)
            elif enemies and ratio < self.cfg["rally_power_ratio"]:
                state, target = 1, self._safe_anchor(frame, uid, rally=True)
                if (
                    previous_state == 1
                    and self.state_age[uid] >= self.cfg["rally_timeout"]
                ):
                    state, target = 3, self._safe_anchor(frame, uid)
            elif enemies and ratio >= self.cfg["engage_power_ratio"]:
                state = 2
                shot = self.sap(frame, uid, remaining)
                if shot is not None:
                    shots[uid] = shot[0]
                    self.stats["expected_sap_damage"] += shot[1]
                    self.stats["sap_selected"] += 1
                    target = xy.copy()
                elif manager.modes[uid] == 3:
                    # Energy comparison is stack/local-power aware, not Eself>Eenemy alone.
                    focus = sorted(
                        enemies,
                        key=lambda e: (e[2], np.abs(np.asarray(e[1]) - xy).sum(), e[0]),
                    )[0]
                    if (
                        energy - self.env_cfg["unit_move_cost"]
                        > focus[2] + self.cfg["safe_energy"]
                    ):
                        target = np.asarray(focus[1])
            else:
                state = 0
            self.stats["rally_success"] += previous_state == 1 and state == 2
            self.stats["retreat_success"] += previous_state == 3 and state in (0, 4)
            self.state_age[uid] = (
                self.state_age[uid] + 1 if previous_state == state else 0
            )
            self.states[uid] = state
            self.stats[STATES[state]] += 1
            self.stats["overrides"] += manager.valid[uid] and not np.array_equal(
                target, manager.goals[uid]
            )
            targets[uid] = tuple(map(int, target))
            priorities[uid] = float(frame["base"][11, *target])
        own = [
            (int(u), tuple(frame["positions"][u]), int(frame["energy"][u]))
            for u in np.flatnonzero(frame["alive"])
        ]
        actions = self.router.route(own, targets, frame["base"][5] > 0.5, priorities)
        for uid, delta in shots.items():
            actions[uid] = (5, int(delta[0]), int(delta[1]))
        stalled = any(
            len(self.router.routes.get(u, []))
            and len(set(self.router.routes[u])) == 1
            and targets[u] != tuple(frame["positions"][u])
            and frame["energy"][u] >= self.env_cfg["unit_move_cost"]
            for u in targets
        )
        self.stats["router_stall"] += stalled
        return actions, dict(severe_threat=bool(severe), path_stall=bool(stalled))
