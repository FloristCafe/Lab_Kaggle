"""Small paired IL-only / Random MAPPO / IL-initialized MAPPO experiment matrix."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.strategic.environment import GoalLevelEnv
from agents.strategic.experiment import digest, provenance, save_json, setup
from agents.strategic.model import (
    CentralizedCritic,
    StrategicActor,
    action_statistics,
    actor_input,
    biased_logits,
    choose_actions,
)
from agents.strategic.semimdp import clipped_actor_loss, smdp_gae


def state_digest(state):
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def device_batch(packet, device="cuda"):
    return {k: v.to(device) for k, v in packet.items()}


def critic_value(critic, b):
    return critic(*actor_input(b), b["team_context"])


def policy(actor, b, config, greedy=False):
    ml, gl = actor(*actor_input(b))
    ml, gl = biased_logits(ml, gl, b["units"], config["tactical"]["hysteresis_bias"])
    mode, goal, lp = choose_actions(ml, gl, greedy=greedy)
    _, entropy = action_statistics(ml, gl, mode, goal)
    alive = b["alive"]
    entropy_value = float(entropy[alive].mean()) if alive.any() else 0.0
    conditional = (
        gl.float()
        .flatten(-2)
        .gather(2, mode[:, :, None, None].expand(-1, -1, 1, 576))
        .squeeze(2)
    )
    logp = conditional.log_softmax(-1)
    goal_entropy = -(logp.exp() * logp).sum(-1)
    return (
        mode,
        goal,
        lp,
        entropy_value,
        float(goal_entropy[alive].mean()) if alive.any() else 0.0,
    )


def update(actor, critic, optimizer, buffer, config):
    rl = config["rl"]
    b = {k: torch.cat([x["obs"][k] for x in buffer]).cuda() for k in buffer[0]["obs"]}
    rewards = np.array([x["reward"] for x in buffer])
    tau = np.array([x["tau"] for x in buffer])
    values = np.array([x["value"] for x in buffer])
    next_values = np.array([x["next_value"] for x in buffer])
    terminal = np.array([x["terminal"] for x in buffer])
    continues = np.r_[~terminal[:-1], False]
    adv, returns = smdp_gae(
        rewards,
        tau,
        values,
        next_values,
        terminal,
        continues,
        rl["gamma"],
        rl["gae_lambda"],
        rl["gae_lambda_timebase"] == "primitive",
    )
    advantages = torch.tensor(adv, dtype=torch.float32, device="cuda")
    returns = torch.tensor(returns, dtype=torch.float32, device="cuda")
    advantages = (advantages - advantages.mean()) / advantages.std(
        unbiased=False
    ).clamp_min(1e-6)
    modes = torch.cat([x["mode"] for x in buffer]).cuda()
    goals = torch.cat([x["goal"] for x in buffer]).cuda()
    oldlp = torch.cat([x["logprob"] for x in buffer]).cuda()
    actor.train()
    critic.train()
    initial_error = 0.0
    for epoch in range(rl["update_epochs"]):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=False):
            ml, gl = actor(*actor_input(b))
            ml, gl = biased_logits(
                ml, gl, b["units"], config["tactical"]["hysteresis_bias"]
            )
            value = critic_value(critic, b).float()
        lp, entropy = action_statistics(ml, gl, modes, goals)
        if epoch == 0 and b["alive"].any():
            initial_error = float((lp - oldlp)[b["alive"]].abs().max())
            if initial_error > 1e-4:
                raise RuntimeError(
                    "behavior policy/logprob mismatch before first update"
                )
        actor_loss, clip_fraction = clipped_actor_loss(
            lp, oldlp, advantages, b["alive"], rl["clip_ratio"]
        )
        count = b["alive"].sum(1).clamp_min(1)
        h = (
            ((entropy * b["alive"]).sum(1) / count)[b["alive"].any(1)].mean()
            if b["alive"].any()
            else entropy.sum() * 0
        )
        value_loss = F.mse_loss(value, returns)
        loss = actor_loss + rl["value_weight"] * value_loss - rl["entropy_weight"] * h
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite MAPPO loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(actor.parameters()) + list(critic.parameters()), rl["max_grad_norm"]
        )
        optimizer.step()
    return dict(
        actor_loss=float(actor_loss),
        value_loss=float(value_loss),
        clip_fraction=float(clip_fraction),
        first_update_logprob_error=initial_error,
        options=len(buffer),
        primitive_steps=int(tau.sum()),
        mean_tau=float(tau.mean()),
        reward_mean=float(rewards.mean()),
        reward_std=float(rewards.std()),
        optimizer_steps=rl["update_epochs"],
    )


def run_arm(arm, args, base_config):
    config = json.loads(json.dumps(base_config))
    setup(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = args.output_dir / arm
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"use a new output directory; {out} already exists")
    out.mkdir(parents=True)
    actor = StrategicActor().cuda()
    il = torch.load(args.il_checkpoint, weights_only=False, map_location="cpu")
    reference_actor_digest = state_digest(il["actor_state_dict"])
    if arm in ("il_mappo", "il_only"):
        actor.load_state_dict(il["actor_state_dict"])
    actor_initial = state_digest(actor.state_dict())
    if arm != "random_mappo" and actor_initial != reference_actor_digest:
        raise RuntimeError(
            "IL actor was not initialized exactly from Stage1 checkpoint"
        )
    torch.manual_seed(args.seed + 1000)
    critic = CentralizedCritic().cuda()
    critic_initial = state_digest(critic.state_dict())
    torch.manual_seed(args.seed)
    from ml_workspace.resource_guard import cuda_memory_guard, cuda_memory_stats
    from ml_workspace.tracking import tracked_run
    from dataclasses import asdict
    import mlflow

    started = time.perf_counter()
    with tracked_run(
        arm,
        experiment_name="lux-s3-strategic-baseline",
        params={
            "seed": args.seed,
            "arm": arm,
            "opponent": "v0",
            "train_episodes": 0
            if arm == "il_only"
            else len(config["rl"]["train_seeds"]),
        },
    ) as run:
        metadata = dict(
            provenance=provenance(),
            il_checkpoint_sha256=digest(args.il_checkpoint),
            initial_actor_digest=actor_initial,
            reference_il_actor_digest=reference_actor_digest,
            initial_critic_digest=critic_initial,
            critic_initialization="random seed+1000, identical across arms",
            actor_boundary="legal EnvObs <=T only",
            critic_boundary="legal team context, no privileged map",
            diagnostic_boundary="engine-state/counterfactual statistics isolated in CPU worker",
            precision="FP32 rollout/update likelihoods, TF32 disabled; avoid BF16 batch-shape ratio drift",
            point_reward_normalization="public max_units * max_steps_in_match; shaping<=0.01 per match",
            bootstrap="official complete 5-match game terminal=1; rollout cuts bootstrap without reset",
            ppo_surrogate="per-unit clipping, average units within decision epochs; start-active masks",
            config=config,
            event_driven=not args.fixed_horizon,
            mlflow_run_id=run.info.run_id,
        )
        save_json(out / "manifest.json", metadata)
        (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        env = GoalLevelEnv(
            config, out / "env_worker.log", event_driven=not args.fixed_horizon
        )
        training, updates, evaluations = [], [], []
        optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()), lr=config["rl"]["lr"]
        )
        try:
            with cuda_memory_guard(f"{arm}-rollout-update"):
                if arm != "il_only":
                    for train_seed in config["rl"]["train_seeds"]:
                        packet = env.reset(train_seed, 0)
                        buffer = []
                        while not env.done:
                            actor.eval()
                            critic.eval()
                            b = device_batch(packet)
                            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                                mode, goal, lp, _, _ = policy(actor, b, config)
                                value = float(critic_value(critic, b)[0])
                            next_packet, reward, terminal, info = env.step(
                                mode[0].cpu().numpy(), goal[0].cpu().numpy()
                            )
                            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                                next_value = (
                                    0.0
                                    if terminal
                                    else float(
                                        critic_value(critic, device_batch(next_packet))[
                                            0
                                        ]
                                    )
                                )
                            buffer.append(
                                dict(
                                    obs=packet,
                                    mode=mode.cpu(),
                                    goal=goal.cpu(),
                                    logprob=lp.cpu(),
                                    reward=reward,
                                    tau=info["tau"],
                                    value=value,
                                    next_value=next_value,
                                    terminal=terminal,
                                )
                            )
                            packet = next_packet
                            if (
                                len(buffer) >= config["rl"]["rollout_options"]
                                or terminal
                            ):
                                metrics = update(
                                    actor, critic, optimizer, buffer, config
                                )
                                updates.append(metrics)
                                mlflow.log_metrics(
                                    {f"update_{k}": v for k, v in metrics.items()},
                                    step=len(updates),
                                )
                                print(
                                    json.dumps(
                                        dict(arm=arm, update=len(updates), **metrics)
                                    ),
                                    flush=True,
                                )
                                save_json(out / "train_log.json", updates)
                                buffer = []
                        summary = env.summary()
                        summary["seed"] = train_seed
                        training.append(summary)
                        save_json(out / f"train_trace_{train_seed}.json", env.trace)
                torch.save(
                    dict(
                        actor_state_dict=actor.state_dict(),
                        critic_state_dict=critic.state_dict(),
                        config=config,
                        arm=arm,
                        optimizer_state_dict=optimizer.state_dict(),
                    ),
                    out / "checkpoint.pth",
                )
                actor.eval()
                for seed in config["rl"]["evaluation_seeds"]:
                    for side in config["rl"]["evaluation_sides"]:
                        packet = env.reset(seed, side)
                        joint_entropies = []
                        goal_entropies = []
                        while not env.done:
                            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                                m, g, _, entropy, goal_entropy = policy(
                                    actor, device_batch(packet), config, greedy=True
                                )
                            joint_entropies.append(entropy)
                            goal_entropies.append(goal_entropy)
                            packet, _, _, _ = env.step(
                                m[0].cpu().numpy(), g[0].cpu().numpy()
                            )
                        result = env.summary()
                        result.update(
                            seed=seed,
                            side=side,
                            joint_entropy=float(np.mean(joint_entropies)),
                            goal_entropy=float(np.mean(goal_entropies)),
                        )
                        evaluations.append(result)
                        save_json(out / f"eval_trace_{seed}_p{side}.json", env.trace)
                        save_json(out / "validation_log.json", evaluations)
                        print(json.dumps(dict(arm=arm, evaluation=result)), flush=True)
        finally:
            env.close()
        summary = dict(
            arm=arm,
            training=training,
            evaluations=evaluations,
            win_rate=float(np.mean([r["full_game_win"] for r in evaluations])),
            mean_score=float(np.mean([r["metrics"]["score"] for r in evaluations])),
            mean_opponent_score=float(
                np.mean([r["metrics"]["opponent_score"] for r in evaluations])
            ),
            primitive_training_steps=sum(r["primitive_steps"] for r in training),
            optimizer_updates=sum(r["optimizer_steps"] for r in updates),
            initial_actor_digest=actor_initial,
            final_actor_digest=state_digest(actor.state_dict()),
            initial_critic_digest=critic_initial,
            updates=updates,
            wall_seconds=time.perf_counter() - started,
            cuda_memory=asdict(cuda_memory_stats()),
            limitation="one training seed, one paired evaluation seed vs v0: smoke-scale, no strength claim",
            mlflow_run_id=run.info.run_id,
        )
        save_json(out / "summary.json", summary)
        mlflow.log_metrics(
            {k: v for k, v in summary.items() if isinstance(v, (int, float))}
        )
        mlflow.log_artifacts(str(out))
        return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", type=Path, default=ROOT / "configs/strategic_baseline.yaml"
    )
    p.add_argument(
        "--il-checkpoint",
        type=Path,
        default=ROOT / "experiments/B_strategic_goal_il/best_model.pth",
    )
    p.add_argument(
        "--output-dir", type=Path, default=ROOT / "experiments/CDE_goal_mappo"
    )
    p.add_argument(
        "--arms",
        nargs="+",
        choices=["random_mappo", "il_mappo", "il_only"],
        default=["random_mappo", "il_mappo", "il_only"],
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fixed-horizon", action="store_true")
    args = p.parse_args()
    config = yaml.safe_load(args.config.read_text())
    summaries = {arm: run_arm(arm, args, config) for arm in args.arms}
    save_json(args.output_dir / "summary.json", summaries)
    print(
        json.dumps(
            {
                arm: {
                    k: v
                    for k, v in summary.items()
                    if k
                    in (
                        "win_rate",
                        "mean_score",
                        "mean_opponent_score",
                        "primitive_training_steps",
                        "optimizer_updates",
                    )
                }
                for arm, summary in summaries.items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
