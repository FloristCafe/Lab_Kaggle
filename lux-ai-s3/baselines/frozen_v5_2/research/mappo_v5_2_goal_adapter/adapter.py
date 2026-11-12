"""Goal-only substitution AFTER the entire original B0 joint greedy decode."""

import copy
import numbers
from collections import Counter

import numpy as np
import torch

from agents.v3.dataset import CAUSAL_CURRENT_CHANNELS
from research.mappo_v5_2.policy import sample
from research.mappo_v5_2_learning_curve.store import tree_hash


def adapt_goals(state, baseline, proposal, *, side, variant):
    if variant not in ("C", "Bprime"):
        raise ValueError("unknown arm")
    if state["alive"].shape != (1, 16) or baseline["goal"].shape != (1, 16):
        raise ValueError("one controlled team of16 required")
    if baseline["stochastic"].any() or baseline["branch_mask"].any():
        raise ValueError(
            "evaluation-only adapter accepts original greedy B0 actions, never PPO samples"
        )
    step = int(state["primitive_step"][0])
    if proposal.step != step or proposal.side != side:
        raise ValueError("stale/future or opposite-side proposal")
    a = copy.deepcopy(baseline)
    alive = state["alive"][0].cpu().numpy().astype(bool)
    review = state["review"][0].cpu().numpy().astype(bool)
    if not np.array_equal(review, baseline["decision_mask"][0].cpu().numpy()):
        raise ValueError("stale B0 review mask")
    replan = baseline["r"][0].cpu().numpy().astype(bool)
    old = baseline["goal"][0].cpu().numpy().astype(int)
    modes = baseline["mode"][0].cpu().numpy().astype(int)
    held = alive & ((state["units"][0, :, 11].cpu().numpy() > 0.5) | (review & replan))
    owners = Counter(int(old[u]) for u in np.flatnonzero(held))
    proposals = {}
    invalid_entries = []
    for uid, target in proposal.targets.items():
        if (
            isinstance(uid, (bool, np.bool_))
            or not isinstance(uid, numbers.Integral)
            or not 0 <= int(uid) < 16
        ):
            invalid_entries.append({"uid": repr(uid), "reason": "invalid_uid"})
            continue
        proposals[int(uid)] = target
    rows = []
    for uid in range(16):
        mode = int(modes[uid])
        goal = int(old[uid])
        present = uid in proposals
        xy = proposals.get(uid)
        eligible = bool(alive[uid] and review[uid] and replan[uid])
        record = {
            "step": step,
            "side": side,
            "uid": uid,
            "node_id": f"{step}:{uid}",
            "alive": bool(alive[uid]),
            "reviewed": bool(review[uid]),
            "eligible_replan": eligible,
            "rule_forced_replan": bool(baseline["forced_action_mask"][0, uid]),
            "B0_mode": mode,
            "B0_goal": goal,
            "V2_proposal_present": present,
            "V2_goal": None,
            "V2_goal_raw_repr": repr(xy),
            "B0_mode_enabled": bool(
                0 <= mode < 5 and baseline["mode_mask"][0, uid, mode]
            ),
            "accepted": False,
            "effective_goal_change": False,
            "attempted": bool(eligible and variant == "Bprime"),
            "selected_goal": goal,
            "mode_compatible": None,
            "conditional_goal_supported": None,
            "reservation_conflict": False,
            "coordinate_valid": False,
            "checks_reasons": [],
        }
        if not alive[uid]:
            reasons = ["unit_not_alive"]
        elif not review[uid]:
            reasons = ["not_reviewed"]
        elif not replan[uid]:
            reasons = ["continue_unchanged"]
        elif not present:
            reasons = ["proposal_missing"]
        else:
            reasons = []
            coordinate = (
                isinstance(xy, (list, tuple, np.ndarray))
                and np.ndim(xy) == 1
                and len(xy) == 2
                and all(
                    not isinstance(x, (bool, np.bool_))
                    and isinstance(x, numbers.Integral)
                    and 0 <= int(x) < 24
                    for x in xy
                )
            )
            record["coordinate_valid"] = bool(coordinate)
            if not coordinate:
                reasons.append("invalid_coordinate")
            elif not 0 <= mode < 5 or not baseline["mode_mask"][0, uid, mode]:
                raise ValueError(
                    "invalid original B0 Mode; do not hide baseline corruption with Goal fallback"
                )
            else:
                record["V2_goal"] = [int(xy[0]), int(xy[1])]
                candidate = int(xy[0]) * 24 + int(xy[1])
                record["V2_goal_flat"] = candidate
                public_ok = bool(state["legal_goals"][0, candidate])
                score_ok = not (
                    mode == 0
                    and state["history"][
                        0, 0, CAUSAL_CURRENT_CHANNELS.index(19)
                    ].flatten()[candidate]
                    > 0.5
                )
                supported = bool(baseline["goal_mask"][0, uid, mode, candidate])
                record["mode_compatible"] = bool(score_ok)
                record["conditional_goal_supported"] = supported
                if not public_ok:
                    visible = (
                        state["history"][
                            0, 0, CAUSAL_CURRENT_CHANNELS.index(0)
                        ].flatten()[candidate]
                        > 0.5
                    )
                    asteroid = (
                        state["history"][
                            0, 0, CAUSAL_CURRENT_CHANNELS.index(5)
                        ].flatten()[candidate]
                        > 0.5
                    )
                    reasons.append(
                        "known_obstacle" if visible and asteroid else "base_goal_mask"
                    )
                if not score_ok:
                    reasons.append("SCORE_known_zero")
                if not supported:
                    reasons.append("original_conditional_support")
                # Reserve ALL remaining original final targets so no accepted early
                # replacement can make a later exact fallback conflict. No swaps,
                # backtracking, resampling, projection or relaxation is introduced.
                conflict = (
                    owners[candidate] - (int(held[uid] and goal == candidate)) > 0
                )
                record["reservation_conflict"] = bool(conflict)
                if conflict:
                    reasons.append("final_target_reserved")
                # A saved mask may reserve an earlier/previous intent even when
                # that cell is now free. Keep that original rejection too.
                if not supported and public_ok and score_ok:
                    record["reservation_conflict"] = True
                if not reasons and variant == "Bprime":
                    if held[uid]:
                        owners[goal] -= 1
                    owners[candidate] += 1
                    held[uid] = True
                    a["goal"][0, uid] = candidate
                    record["accepted"] = True
                    record["effective_goal_change"] = candidate != goal
                    record["selected_goal"] = candidate
        record["checks_reasons"] = reasons
        record["outcome"] = (
            "control_unchanged"
            if variant == "C"
            else ("accepted" if record["accepted"] else "fallback")
        )
        record["reason"] = (
            "C_no_intervention"
            if variant == "C"
            else (
                "accepted_same_goal"
                if record["accepted"] and not record["effective_goal_change"]
                else ("accepted" if record["accepted"] else reasons[0])
            )
        )
        rows.append(record)
    for key in baseline:
        if key == "goal":
            continue
        x, y = baseline[key], a[key]
        if isinstance(x, torch.Tensor):
            if not torch.equal(x, y):
                raise RuntimeError("adapter changed non-Goal B0 field " + key)
        elif x != y:
            raise RuntimeError("adapter changed B0 metadata")
    return a, {
        "schema": "v2_goal_adapter_decision_v1",
        "step": step,
        "side": side,
        "variant": variant,
        "rows": rows,
        "invalid_proposal_entries": invalid_entries,
        "proposal_input_sha256": proposal.input_sha256,
        "decode": "entire original B0 joint greedy decode first; R/Mode fixed on this state; no independent Mode argmax",
        "support": "unchanged original saved per-UID conditional mask AND conflict-free final allocation",
        "behavior_likelihood": "NOT APPLICABLE: deterministic external Goal proposal, not a PPO behavior sample",
    }


@torch.no_grad()
def decode(actor, state, config, shadow, *, variant):
    if actor.training or any(p.requires_grad for p in actor.parameters()):
        raise ValueError("explicit frozen eval Actor required")
    if config["scheduler"] != "B0" or config["policy"]["gate"] != "learned":
        raise ValueError("only original learned-R B0 supported")
    if any(key in config for key in ("pbrs", "critic_bound", "il_reference_kl")):
        raise ValueError("original B0 only")
    identity = tree_hash(actor.state_dict())
    inputs = tree_hash(state)
    baseline, entropy = sample(actor, state, config, greedy=True)
    action, record = adapt_goals(
        state,
        baseline,
        shadow.proposals(int(state["primitive_step"][0])),
        side=shadow.side,
        variant=variant,
    )
    if identity != tree_hash(actor.state_dict()) or inputs != tree_hash(state):
        raise RuntimeError("frozen Actor/input mutated")
    record["actor_state_sha256"] = identity
    record["config_sha256"] = tree_hash(config)
    return action, entropy, record
