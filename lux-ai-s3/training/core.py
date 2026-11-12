"""Single official-engine collection and PPO implementation for A/B/C."""
import hashlib
import importlib
import json
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.nn import functional as F

from agents.mappo.shared.credit import OptionLedger, primitive_targets
from agents.mappo.shared.model import PrefixCritic
from agents.mappo.shared.policy import decide, environment_action, pack_policy


def models(name, device):
    # Variant-specific parameter counts cannot shift the common Critic or rollout RNG.
    with torch.random.fork_rng(devices=[]):
        seed = torch.initial_seed()
        actor = importlib.import_module(f"agents.mappo.model_{name}").Actor()
        torch.random.default_generator.manual_seed((seed + 1000) % (2**63))
        critic = PrefixCritic()
    return actor.to(device), critic.to(device)


def batch(rows, device):
    return {key: torch.from_numpy(np.stack([r[key] for r in rows])).to(device)
            for key in rows[0]}


def batches(size, chunk, shuffle=True):
    indices = torch.randperm(size).tolist() if shuffle else list(range(size))
    for start in range(0, size, chunk):
        yield indices[start:start+chunk]


@dataclass
class Decision:
    state: dict
    commands: np.ndarray
    logp: np.ndarray
    mask: np.ndarray
    timestep: int


@dataclass
class Rollout:
    states: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    trajectory: list = field(default_factory=list)
    options: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    final_state: dict = field(default_factory=dict)
    seconds: float = 0.0


@torch.no_grad()
def play(env, actor, root, side, device, opponent="v2", limit=None, greedy=False,
         audit_actions=False):
    begin = time.monotonic()
    env.reset(root, side=side, opponent=opponent)
    roll = Rollout()
    ledger = OptionLedger()

    def after(bridge, reward, actions, reset):
        ledger.after_step(bridge, reset, getattr(bridge.tactical, "last_targets", None))
        roll.rewards.append(reward)
        if audit_actions:
            obs_hash = hashlib.sha256(json.dumps(bridge.obs, sort_keys=True).encode()).hexdigest()
            roll.trajectory.append({"t": bridge.steps, "actions": actions.tolist(),
                                    "public_obs_sha256": obs_hash,
                                    "team_points": bridge.obs["team_points"],
                                    "team_wins": bridge.obs["team_wins"],
                                    "terminal": bridge.terminal, "reset": bool(reset)})
        roll.states.append(bridge.physics.critic_view(bridge))

    roll.states.append(env.physics.critic_view(env))
    while not (env.terminal or env.truncated):
        state = pack_policy(env, actor.uses_memory)
        tensors = batch([state], device)
        commands, logp, mask, _, _ = decide(actor, tensors, greedy=greedy)
        action = environment_action(tensors, commands, side, mask)
        row = commands[0].cpu().numpy()
        ledger.start(env.steps, row, state["review"], side)
        if mask.any():
            roll.decisions.append(Decision(state, row, logp[0].cpu().numpy(), mask[0].cpu().numpy(), env.steps))
        env.step(action, max_steps=limit, on_primitive=after)
    roll.final_state = roll.states.pop()
    roll.summary = env.summary()
    roll.options = ledger.finish(env.steps, roll.rewards, env.config["ppo"]["gamma"])
    roll.seconds = time.monotonic() - begin
    if roll.summary["complete"] and (len(roll.rewards) != 505 or sum(roll.rewards) != roll.summary["game_margin"]):
        raise RuntimeError("505-step official reward reconstruction failed")
    if roll.summary["goal_conflicts"] or not roll.states:
        raise RuntimeError("committed Goal conflict or empty rollout")
    return roll


def grad_step(loss, optimizer, clip, params):
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(params, clip, error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def delta_norm(before, module):
    total = sum((p.detach() - old).square().sum() for old, p in zip(before, module.parameters(), strict=True))
    return float(total.sqrt())


def learn(rolls, actor, critic, actor_opt, critic_opt, cfg, device):
    begin = time.monotonic()
    params = list(actor.parameters())
    critparams = list(critic.parameters())
    before = [p.detach().clone() for p in params]
    before_c = [p.detach().clone() for p in critparams]
    decisions = [d for roll in rolls for d in roll.decisions]
    primitive = [s for roll in rolls for s in roll.states]
    rows = [d.state for d in decisions]
    if not rows:
        raise RuntimeError("no stochastic decision factors in rollout")
    commands = torch.from_numpy(np.stack([d.commands for d in decisions])).to(device)
    old = torch.from_numpy(np.stack([d.logp for d in decisions])).to(device)
    mask = torch.from_numpy(np.stack([d.mask for d in decisions])).to(device)
    targets = []
    value_targets = []
    max_replay_error = 0.0
    prior = torch.zeros_like(old)
    with torch.no_grad():
        for roll in rolls:
            x = [critic(batch([roll.states[i] for i in ids], device)).cpu().numpy()
                 for ids in batches(len(roll.states), cfg["critic_minibatch"], shuffle=False)]
            terminal = roll.summary["complete"]
            bootstrap = 0.0 if terminal else float(critic(batch([roll.final_state], device))[0])
            value_targets.extend(primitive_targets(roll.rewards, np.concatenate(x), bootstrap,
                                                   terminal, cfg["gamma"], cfg["lambda"]))
            total = bootstrap
            returns = []
            for r in reversed(roll.rewards):
                total = r + cfg["gamma"] * total
                returns.append(total)
            returns.reverse()
            targets.extend(returns[d.timestep] for d in roll.decisions)
        for ids in batches(len(rows), cfg["actor_minibatch"], shuffle=False):
            s = batch([rows[i] for i in ids], device)
            _, logp, active, _, baseline = decide(actor, s, commands[ids],
                                                   critic=critic, encoded_value=critic.encode(s))
            if not torch.equal(active, mask[ids]):
                raise RuntimeError("stored and replayed branch masks differ")
            max_replay_error = max(max_replay_error, float((logp - old[ids]).abs().max()))
            prior[ids] = baseline
    if max_replay_error > 1e-4:
        raise RuntimeError(f"behavior likelihood mismatch {max_replay_error}")
    target = torch.tensor(targets, dtype=torch.float32, device=device)
    v_target = torch.tensor(value_targets, dtype=torch.float32, device=device)
    advantages = target[:, None, None] - prior
    for k in range(3):
        active = mask[..., k]
        samples = advantages[..., k][active]
        if samples.numel() > 1:
            advantages[..., k] = (advantages[..., k] - samples.mean()) / samples.std(unbiased=False).clamp_min(.01)
    actor_metrics, critic_metrics = [], []
    actor_updates = critic_updates = 0
    for epoch in range(cfg["epochs"]):
        for ids in batches(len(rows), cfg["actor_minibatch"]):
            s = batch([rows[i] for i in ids], device)
            _, logp, active, entropy, _ = decide(actor, s, commands[ids])
            dif = logp - old[ids]
            ratio = dif[active].clamp(-20, 20).exp()
            adv = advantages[ids][active]
            bounded = ratio.clamp(1-cfg["clip"], 1+cfg["clip"])
            policy_loss = -torch.minimum(ratio*adv, bounded*adv).mean()
            h = entropy[active].mean()
            joint = dif.sum((1, 2))
            kl = (joint.clamp(-20, 20).exp() - 1 - joint).mean()
            if not torch.isfinite(kl):
                raise RuntimeError("nonfinite joint KL")
            # Avoid further updates once the policy drifts on an observed minibatch.
            if float(kl.detach()) > cfg["target_joint_kl"]:
                break
            norm = grad_step(policy_loss - cfg["entropy_weight"]*h, actor_opt, cfg["grad_clip"], params)
            actor_updates += 1
            actor_metrics.append((float(policy_loss.detach()), float(h.detach()), float(kl.detach()),
                                  float(((ratio-1).abs() > cfg["clip"]).float().mean()), norm))
        else:
            continue
        break
    for _ in range(cfg["epochs"]):
        for ids in batches(len(primitive), cfg["critic_minibatch"]):
            prediction = critic(batch([primitive[i] for i in ids], device))
            value_loss = F.mse_loss(prediction, v_target[ids])
            norm = grad_step(cfg["value_weight"]*value_loss, critic_opt, cfg["grad_clip"], critparams)
            critic_updates += 1
            critic_metrics.append((float(value_loss.detach()), norm))
        for ids in batches(len(rows), cfg["actor_minibatch"]):
            s = batch([rows[i] for i in ids], device)
            with torch.no_grad():
                prefix = prefix_inputs(s, commands[ids])
            enc = critic.encode(s)
            pred = []
            for uid, claims in enumerate(prefix):
                if not bool(mask[ids, uid].any()):
                    pred.append(torch.zeros((len(ids), 3), device=device))
                    continue
                vr, vm, vg = critic.conditional(enc, uid, claims)
                mode = commands[ids, uid, 1]
                pred.append(torch.stack((vr, vm, vg.gather(1, mode[:, None]).squeeze(1)), -1))
            p = torch.stack(pred, 1)
            goal_value_loss = F.mse_loss(p[mask[ids]], target[ids, None, None].expand_as(p)[mask[ids]])
            grad_step(cfg["value_weight"]*goal_value_loss, critic_opt, cfg["grad_clip"], critparams)
            critic_updates += 1
    a = np.mean(actor_metrics, axis=0) if actor_metrics else [0.0]*5
    c = np.mean(critic_metrics, axis=0)
    return dict(zip(("policy_loss", "entropy", "approx_joint_kl", "clip_fraction", "actor_grad_norm"),
                    map(float, a), strict=True),
                value_loss=float(c[0]), critic_grad_norm=float(c[1]),
                actor_delta_l2=delta_norm(before, actor), critic_delta_l2=delta_norm(before_c, critic),
                actor_updates=actor_updates, critic_updates=critic_updates, max_replay_error=max_replay_error,
                update_seconds=time.monotonic()-begin)


def prefix_inputs(state, commands):
    """Conditional value prefixes are identical to the policy's recorded commitments."""
    claims = torch.where(state["valid"], state["previous"], -1)
    for uid in range(16):
        before = claims.clone()
        before[:, uid] = -1
        yield before
        replan = state["review"][:, uid] & commands[:, uid, 0].bool()
        claims[:, uid] = torch.where(replan, commands[:, uid, 2], claims[:, uid])
