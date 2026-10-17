"""Team-synchronized event-driven macro wrapper over an isolated official engine."""

import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .control import GoalManager, TacticalController
from .experiment import ROOT
from .features import CausalObservationBuilder, entropy_map, unit_vectors


def lifecycle_events(before, after):
    """Public evidence only: round clearing is distinct from unit destruction."""
    reset = after["match_step"] < before["match_step"]
    both = before["alive"] & after["alive"]
    reused = both & (
        (np.abs(after["positions"] - before["positions"]).sum(1) > 1)
        | (after["energy"] - before["energy"] > 20)
    )
    lost = before["alive"] & ~after["alive"]
    deaths = (lost | reused) if not reset else np.zeros(16, bool)
    return reset, reused, lost, deaths


class OpponentAdapter:
    """IPC adapter; worker currently hosts existing v0, not a claimed expert."""

    def __init__(self, log_path):
        self.log = Path(log_path).open("a")
        env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
        self.process = subprocess.Popen(
            [
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / "scripts/strategic_env_worker.py"),
            ],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            env=env,
        )

    def request(self, **data):
        self.process.stdin.write(json.dumps(data) + "\n")
        self.process.stdin.flush()
        result = self.process.stdout.readline()
        if not result:
            raise RuntimeError("Lux worker ended; inspect env_worker.log")
        return json.loads(result)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"op":"close"}\n')
            self.process.stdin.flush()
            self.process.stdin.close()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait()
        self.log.close()


class GoalLevelEnv:
    def __init__(self, config, log_path, event_driven=True):
        self.config = config
        self.bridge = OpponentAdapter(log_path)
        self.event_driven = event_driven

    def reset(self, seed, side=0):
        self.side = side
        response = self.bridge.request(op="reset", seed=seed, side=side)
        self.env_cfg = response["env_cfg"]
        self.builder = CausalObservationBuilder(side, self.env_cfg)
        self.manager = GoalManager(self.config["tactical"])
        self.tactical = TacticalController(self.env_cfg, self.config["tactical"])
        self.obs = response["obs"]
        self.frame = self.builder.observe(self.obs)
        self.metrics = Counter()
        self.events = Counter()
        self.trace = []
        self.primitive_steps = 0
        self.mode_counts = np.zeros(5, np.int64)
        self.taus = []
        self.goal_age_sum = self.goal_age_count = 0
        self.done = False
        return self.packet()

    def packet(self):
        f = self.frame
        units = unit_vectors(
            f,
            self.manager.goals,
            self.manager.modes,
            self.manager.ages,
            self.manager.valid,
            self.env_cfg,
        )
        points = np.asarray(self.obs["team_points"])
        wins = np.asarray(self.obs["team_wins"])
        team = np.array(
            [
                points[self.side] / 1000,
                points[1 - self.side] / 1000,
                (wins[self.side] - wins[1 - self.side]) / 5,
                f["rank"] / 576,
            ],
            np.float32,
        )
        return dict(
            history=torch.from_numpy(f["history"][None]),
            positions=torch.from_numpy(f["positions"][None].astype(np.int64)),
            units=torch.from_numpy(units[None]),
            alive=torch.from_numpy(f["alive"][None]),
            team_context=torch.from_numpy(team[None]),
        )

    def step(self, modes, goals):
        cfg = self.config
        start_alive = self.frame["alive"].copy()
        self.manager.assign(modes, goals, start_alive)
        self.mode_counts += np.bincount(np.asarray(modes)[start_alive], minlength=5)
        start_step = self.frame["step"]
        discounted = 0.0
        components = Counter()
        triggers = []
        for k in range(cfg["tactical"]["macro_horizon"]):
            actions, tactical_info = self.tactical.act(self.frame, self.manager)
            before = self.frame
            old_wins = np.asarray(self.obs["team_wins"])
            old_points = np.asarray(self.obs["team_points"])
            old_entropy = float(entropy_map(before["base"][11]).sum())
            response = self.bridge.request(op="step", actions=actions.tolist())
            self.obs = response["obs"]
            self.frame = self.builder.observe(self.obs)
            new_wins = np.asarray(self.obs["team_wins"])
            new_points = np.asarray(self.obs["team_points"])
            wins = (new_wins - old_wins)[self.side] - (new_wins - old_wins)[
                1 - self.side
            ]
            points = 0.0
            if self.frame["match_step"] > before["match_step"]:
                points = (new_points - old_points)[self.side] - (
                    new_points - old_points
                )[1 - self.side]
            info = max(
                0, old_entropy - float(entropy_map(self.frame["base"][11]).sum())
            )
            round_reset, reused, lost, deaths = lifecycle_events(before, self.frame)
            observed_deaths = int(deaths.sum())
            raw = dict(
                win=float(wins),
                points=float(points),
                information=info,
                death=-observed_deaths,
                energy_waste=-int((actions[:, 0] == 5).sum())
                * self.env_cfg["unit_sap_cost"]
                / 400,
            )
            score_scale = self.env_cfg["max_units"] * self.env_cfg["max_steps_in_match"]
            reward = (
                cfg["rl"]["win_reward"] * raw["win"]
                + cfg["rl"]["point_shaping"] * raw["points"] / score_scale
                + cfg["rl"]["information_shaping"] * raw["information"]
                + cfg["rl"]["death_shaping"] * raw["death"]
                + cfg["rl"]["energy_waste_shaping"] * raw["energy_waste"]
            )
            for key, value in raw.items():
                components[key] += value
            discounted += cfg["rl"]["gamma"] ** k * reward
            self.metrics["undiscounted_reward"] += reward
            for name, value in response["metrics"].items():
                if name == "engine_score_gain":
                    self.metrics["score"] += value[0]
                    self.metrics["opponent_score"] += value[1]
                elif name == "alive_units":
                    self.metrics["alive_unit_steps"] += value
                else:
                    self.metrics[name] += value
            for uid in np.flatnonzero(deaths):
                self.tactical.stats["retreat_failure"] += self.tactical.states[uid] == 3
            # A new life cannot inherit a dead unit's tactical state or goal.
            cleared = lost | reused
            if round_reset:
                cleared[:] = True
            self.tactical.states[cleared] = 0
            self.tactical.state_age[cleared] = 0
            self.manager.valid[cleared] = False
            self.manager.ages[cleared] = 0
            self.primitive_steps += 1
            self.manager.advance(self.frame)
            self.goal_age_sum += int(self.manager.ages[self.manager.valid].sum())
            self.goal_age_count += int(self.manager.valid.sum())
            self.done = response["done"]
            critical = []
            if reused.any() or not np.array_equal(before["alive"], self.frame["alive"]):
                critical.append("unit_death_or_spawn")
            if round_reset:
                critical.append("round_reset")
            triggers = list(critical)
            if self.event_driven and k + 1 >= cfg["tactical"]["minimum_macro_steps"]:
                if self.manager.invalidated(self.frame):
                    triggers.append("goal_invalidated")
                if any(
                    self.manager.valid[u]
                    and np.array_equal(
                        self.frame["positions"][u], self.manager.goals[u]
                    )
                    for u in np.flatnonzero(self.frame["alive"])
                ):
                    triggers.append("goal_reached")
                if tactical_info["severe_threat"]:
                    triggers.append("severe_threat")
                if tactical_info["path_stall"]:
                    triggers.append("router_stall")
                if points != 0 or wins != 0:
                    triggers.append("score_event")
                if info >= cfg["labeler"]["belief_entropy_threshold"]:
                    triggers.append("belief_update")
            if self.done or triggers:
                break
        tau = k + 1
        if not triggers and not self.done:
            triggers = ["max_macro_horizon"]
        self.events.update(triggers)
        self.taus.append(tau)
        self.trace.append(
            dict(
                start=start_step,
                end=self.frame["step"],
                tau=tau,
                reward=discounted,
                terminal=self.done,
                events=triggers,
                mode=np.asarray(modes).tolist(),
                goals=np.asarray(goals).tolist(),
                alive=start_alive.tolist(),
                components=dict(components),
            )
        )
        return (
            self.packet(),
            discounted,
            self.done,
            dict(tau=tau, components=dict(components), events=triggers),
        )

    def summary(self):
        wins = np.asarray(self.obs["team_wins"])
        assignments = max(1, self.manager.stats["assignments"])
        spent = self.metrics["sap_energy_spent"]
        return dict(
            completed=bool(self.done),
            primitive_steps=self.primitive_steps,
            match_wins=int(wins[self.side]),
            opponent_match_wins=int(wins[1 - self.side]),
            full_game_win=bool(wins[self.side] > wins[1 - self.side])
            if self.done
            else None,
            survival=self.metrics["alive_unit_steps"]
            / max(1, 16 * self.primitive_steps),
            death_rate_per_100_steps=100
            * self.metrics["own_deaths"]
            / max(1, self.primitive_steps),
            goal_persistence=self.manager.stats["unchanged_assignments"] / assignments,
            replanning_frequency=len(self.taus) / max(1, self.primitive_steps),
            goal_change_frequency=self.manager.stats["changes"]
            / max(1, self.primitive_steps),
            mean_goal_age=self.goal_age_sum / max(1, self.goal_age_count),
            mean_tau=float(np.mean(self.taus)) if self.taus else 0.0,
            mode_distribution=self.mode_counts.tolist(),
            tactical_states={
                name: int(self.tactical.stats[name])
                for name in ("NAVIGATE", "RALLY", "ENGAGE", "RETREAT", "RECHARGE")
            },
            tactical_diagnostics={
                k: (v.item() if isinstance(v, np.generic) else v)
                for k, v in self.tactical.stats.items()
            },
            events=dict(self.events),
            sap_energy_efficiency=self.metrics["sap_effective_damage"] / spent
            if spent
            else 0.0,
            metrics=dict(self.metrics),
            metric_privilege="engine state for diagnostics only; never Actor, critic or reward shaping",
        )

    def close(self):
        self.bridge.close()
