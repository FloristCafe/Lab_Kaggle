"""Primitive value targets and pre-action conditional Monte Carlo credit."""
import numpy as np


def primitive_targets(rewards, values, final_value, terminal, gamma, lam):
    if len(rewards) != len(values):
        raise ValueError("each primitive step needs a before-action value")
    out = np.empty(len(rewards), dtype=np.float32)
    previous = 0.0 if terminal else final_value
    gae = 0.0
    for t in range(len(out)-1, -1, -1):
        delta = rewards[t] + gamma * previous - values[t]
        gae = delta + gamma * lam * gae
        out[t] = values[t] + gae
        previous = values[t]
    return out


class OptionLedger:
    """Audit UID lifetimes independently of team transition boundaries."""
    def __init__(self):
        self.active = {}
        self.events = []

    def start(self, t, commands, review, side):
        for uid, (r, mode, goal) in enumerate(commands):
            if review[uid] and r:
                self.end(uid, t, "replan")
                self.active[uid] = {"uid": uid, "start": t, "mode": int(mode),
                                    "goal": int(goal if side == 0 else 575-goal),
                                    "first_arrival": None, "overridden_steps": 0}

    def after_step(self, env, reset, tactical_targets):
        t = env.steps
        for uid, option in list(self.active.items()):
            goal = option["goal"]
            if tactical_targets is not None and uid in tactical_targets:
                option["overridden_steps"] += int(tuple(tactical_targets[uid]) != divmod(goal, 24))
            if env.terminal:
                reason = "game_terminal"
            elif env.truncated:
                reason = "external_truncation"
            elif reset:
                reason = "match_reset"
            elif not env.frame["alive"][uid]:
                reason = "death_or_absent"
            elif not env.manager.valid[uid]:
                reason = env.manager.invalidation[uid] or "goal_invalid"
            else:
                if option["first_arrival"] is None and tuple(env.frame["positions"][uid]) == divmod(goal, 24):
                    option["first_arrival"] = t
                continue
            self.end(uid, t, reason)

    def end(self, uid, t, reason):
        if uid in self.active:
            event = self.active.pop(uid)
            event.update(end=t, duration=t-event["start"], reason=reason)
            self.events.append(event)

    def finish(self, t, rewards, gamma):
        if self.active:
            raise RuntimeError("active options at end of collection")
        result = []
        for event in self.events:
            start, end = event["start"], event["end"]
            result.append(dict(event,
                               lifetime_reward=sum(gamma**j * r for j, r in enumerate(rewards[start:end])),
                               continuation_reward=sum(gamma**j * r for j, r in enumerate(rewards[end:], end-start))))
        return result
