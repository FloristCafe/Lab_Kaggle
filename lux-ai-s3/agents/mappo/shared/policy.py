"""Exact conditional likelihood of a sampled fleet action."""
import torch
from torch.distributions import Categorical


def pack_policy(env, memory):
    state = env.physics.policy_view(env, memory)
    frame = env.frame
    manager = env.manager
    blocked = ((frame["base"][5] > .5) & frame["sensor"]).flatten()
    false_score = (frame["base"][19] > .5).flatten()
    goals = manager.goals @ [24, 1]
    if env.side == 1:
        blocked, false_score, goals = blocked[::-1], false_score[::-1], 575 - goals
    state.update(review=(manager.review_mask & frame["alive"]).copy(),
                 previous=goals.copy(), valid=(manager.valid & frame["alive"]).copy(),
                 legal=(~blocked).copy(), false_score=false_score.copy())
    return state


def decide(actor, state, recorded=None, greedy=False, critic=None, encoded_value=None):
    """Replay the same conditionals for PPO. Branch bits R, Mode, Goal."""
    encoded = actor(state)
    b = state["alive"].shape[0]
    device = state["alive"].device
    claims = torch.where(state["valid"], state["previous"], -1)
    out, logprobs, masks, entropies, values = [], [], [], [], []
    legality = torch.ones(b, device=device, dtype=torch.bool)
    grid = torch.arange(576, device=device)
    for uid in range(16):
        review = state["review"][:, uid]
        if not bool(review.any()):
            out.append(torch.zeros((b, 3), dtype=torch.long, device=device))
            logprobs.append(torch.zeros((b, 3), device=device))
            entropies.append(torch.zeros((b, 3), device=device))
            masks.append(torch.zeros((b, 3), dtype=torch.bool, device=device))
            if critic is not None:
                values.append(torch.zeros((b, 3), device=device))
            continue
        before = claims.clone()
        before[:, uid] = -1
        occupied = ((before[:, :, None] == grid) & (before[:, :, None] >= 0)).any(1)
        previous = state["previous"][:, uid]
        forced = ~state["valid"][:, uid] | occupied.gather(1, previous[:, None]).squeeze(1)
        r, m, g = actor.prefix_adjustment(encoded, uid, before)
        rdist = Categorical(logits=r.float(), validate_args=False)
        if recorded is None:
            r_choice = r.argmax(-1) if greedy else rdist.sample()
            r_choice = torch.where(review, (r_choice.bool() | forced).long(), 0)
        else:
            r_choice = recorded[:, uid, 0]
        replan = review & r_choice.bool()
        legality &= ~review | ~forced | replan
        legal = (state["legal"] & ~occupied)[:, None].expand(-1, 5, -1).clone()
        legal[:, 0] &= ~state["false_score"]
        allowed_mode = legal.any(-1)
        legality &= ~replan | allowed_mode.any(1)
        # Dummies are used exclusively for evaluating inactive branches or reporting an error.
        safe_modes = allowed_mode.clone()
        safe_modes[:, 0] |= ~safe_modes.any(-1)
        safe = legal.clone()
        safe[:, :, 0] |= ~allowed_mode
        mdist = Categorical(logits=m.float().masked_fill(~safe_modes, -torch.inf),
                            validate_args=False)
        gdists = Categorical(logits=g.float().masked_fill(~safe, -torch.inf), validate_args=False)
        if recorded is not None:
            mode, goal = recorded[:, uid, 1], recorded[:, uid, 2]
        elif greedy:
            joint = mdist.logits[:, :, None] + gdists.logits
            joint = joint.masked_fill(~safe_modes[:, :, None], -torch.inf)
            best = joint.flatten(1).argmax(1)
            mode, goal = best // 576, best % 576
        else:
            mode = mdist.sample()
            goal = Categorical(logits=gdists.logits[torch.arange(b, device=device), mode],
                               validate_args=False).sample()
        rows = torch.arange(b, device=device)
        legality &= ~replan | legal[rows, mode, goal]
        goal_dist = Categorical(logits=gdists.logits[rows, mode], validate_args=False)
        active = torch.stack((review & ~forced, replan, replan), -1)
        logs = torch.stack((rdist.log_prob(r_choice), mdist.log_prob(mode), goal_dist.log_prob(goal)), -1)
        entropy = torch.stack((rdist.entropy(), mdist.entropy(), goal_dist.entropy()), -1)
        masks.append(active)
        logprobs.append(logs.masked_fill(~active, 0))
        entropies.append(entropy.masked_fill(~active, 0))
        out.append(torch.stack((r_choice, torch.where(replan, mode, 0),
                                torch.where(replan, goal, 0)), -1))
        if critic is not None:
            vr, vm, vg = critic.conditional(encoded_value, uid, before)
            values.append(torch.stack((vr, vm, vg[rows, mode]), -1))
        claims = claims.clone()
        claims[:, uid] = torch.where(replan, goal, claims[:, uid])
    if not bool(legality.all()):
        raise RuntimeError("illegal recorded or sampled fleet action")
    commands = torch.stack(out, 1)
    results = tuple(torch.stack(items, 1) for items in (logprobs, masks, entropies))
    return commands, *results, torch.stack(values, 1) if critic is not None else None


def environment_action(state, commands, side, branch):
    replan, mode, goal = commands.unbind(-1)
    return {"r": replan.bool(), "mode": mode, "goal": goal if side == 0 else 575 - goal,
            "decision_mask": state["review"], "branch_mask": branch,
            "forced_action_mask": state["review"] & ~branch[..., 0]}
