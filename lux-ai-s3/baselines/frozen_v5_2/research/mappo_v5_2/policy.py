"""v5.2 conditional policy. Forced R is a rule; its MG are Actor samples.

UID-ordered reservation masks match Legacy. No network or physical rule changes.
"""
import torch
from torch.distributions import Bernoulli, Categorical

from agents.mappo_v5.policy import (
    masked,
    optimizer,
    policy_outputs,
    previous,
    statistics,
)
from agents.v3.dataset import CAUSAL_CURRENT_CHANNELS

__all__ = ['optimizer', 'sample']


@torch.no_grad()
def sample(actor, state, config, greedy=False):
    outputs = policy_outputs(actor, state, config['model'])
    rlog, ml, gl = outputs
    batch, count = rlog.shape
    where = rlog.device
    pm, pg = previous(state['units'])
    valid = (state['units'][..., 11] > .5) & state['alive']
    action = {
        'r': torch.zeros_like(valid), 'mode': pm.clone(), 'goal': pg.clone(),
        'decision_mask': state['review'].clone(), 'forced_action_mask': torch.zeros_like(valid),
        'branch_mask': torch.zeros((batch, count, 3), dtype=torch.bool, device=where),
        'component_forced': torch.zeros((batch, count, 3), dtype=torch.bool, device=where),
        'r_mask': torch.ones((batch, count, 2), dtype=torch.bool, device=where),
        'mode_mask': torch.ones((batch, count, 5), dtype=torch.bool, device=where),
        'goal_mask': torch.ones((batch, count, 5, 576), dtype=torch.bool, device=where),
        'order': torch.arange(count, device=where).expand(batch, -1).clone(),
        'stochastic': torch.full((batch,), not greedy, dtype=torch.bool, device=where),
        'scheduler_id': torch.full((batch,), int(config['scheduler'] == 'B1'), dtype=torch.uint8, device=where),
    }
    draws = {'R': 0, 'Mode': 0, 'Goal': 0}
    for b in range(batch):
        owners = {}
        for u in range(count):
            if valid[b, u]:
                owners.setdefault(int(pg[b, u]), set()).add(u)
        for u in action['order'][b].tolist():
            if not state['review'][b, u] or not state['alive'][b, u]:
                continue
            for cell in list(owners):
                owners[cell].discard(u)
                if not owners[cell]:
                    del owners[cell]
            forced = not bool(valid[b, u]) or int(pg[b, u]) in owners
            action['forced_action_mask'][b, u] = forced
            if forced:
                replan = True
                action['r_mask'][b, u] = torch.tensor([False, True], device=where)
                action['component_forced'][b, u, 0] = True
            elif config['policy']['gate'] == 'deterministic':
                replan = bool(state['gate_replan'][b, u])
                action['r_mask'][b, u] = torch.tensor([not replan, replan], device=where)
                action['component_forced'][b, u, 0] = True
            else:
                if greedy:
                    replan = bool(rlog[b, u] >= 0)
                else:
                    replan = bool(Bernoulli(logits=rlog[b, u].float()).sample())
                    draws['R'] += 1
                action['branch_mask'][b, u, 0] = not greedy
            action['r'][b, u] = replan
            if not replan:
                owners.setdefault(int(pg[b, u]), set()).add(u)
                continue
            support = state['legal_goals'][b].clone()
            if owners:
                support[list(owners)] = False
            if not support.any():
                raise ValueError('empty conditional support; stop instead of relaxing a physical mask')
            mode_goal_support = support[None].expand(5, -1).clone()
            # Exact existing TeamMemory hard invalidity predicate for SCORE.
            # Only causal public solver conclusions in the current history frame.
            false_score = state['history'][b, 0, CAUSAL_CURRENT_CHANNELS.index(19)].flatten() > .5
            mode_goal_support[0] &= ~false_score
            action['goal_mask'][b, u] = mode_goal_support
            action['mode_mask'][b, u] = action['goal_mask'][b, u].any(-1)
            mode_logits = masked(ml[b, u], action['mode_mask'][b, u])
            if greedy:
                # Inactive modes are excluded by mode_mask. A dummy conditional
                # normalization prevents NaN, without adding a legal sampled goal.
                safe_support = mode_goal_support.clone()
                safe_support[~action['mode_mask'][b, u], 0] = True
                goal_logits = masked(gl[b, u].flatten(-2), safe_support)
                joint = mode_logits.log_softmax(-1)[:, None] + goal_logits.log_softmax(-1)
                choice = int(joint.flatten().argmax())
                mode, goal = choice // 576, choice % 576
            else:
                mode = int(Categorical(logits=mode_logits).sample())
                goal_logits = masked(gl[b, u, mode].flatten(), mode_goal_support[mode])
                goal = int(Categorical(logits=goal_logits).sample())
                draws['Mode'] += 1
                draws['Goal'] += 1
                action['branch_mask'][b, u, 1:] = True
            action['mode'][b, u], action['goal'][b, u] = mode, goal
            owners.setdefault(goal, set()).add(u)
    action['policy_mask'] = action['branch_mask'].any(-1)
    action['component_sampled'] = action['branch_mask'].clone()
    action['log_prob_valid'] = action['branch_mask'].clone()
    lp, entropy, _, factors = statistics(outputs, state['units'], action, return_factors=True)
    action['old_logprob'] = lp.detach().clone()
    action['old_logprob_factors'] = factors.detach().clone()
    action['draw_counts'] = draws
    return action, entropy
