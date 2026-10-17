"""Project-Python CPU Lux worker. Only legal EnvObs crosses the Actor boundary.

Raw engine states/unknown physics are used ONLY for isolated diagnostic counts
and a same-key no-our-SAP counterfactual, never for actors, labels or shaping.
"""

import json
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map  # adapter only, no pinned vendor edits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.v0.agent import Agent as V0Agent
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams, env_params_ranges


def encode(x):
    if isinstance(x, dict):
        return {k: encode(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [encode(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def public(obs):
    return encode(flax.serialization.to_state_dict(obs))


def movement_snapshot(state, actions, params):
    pos = np.array(state.units.position)
    energy = np.array(state.units.energy)[..., 0]
    mask = np.array(state.units_mask) & (energy >= 0)
    if int(state.match_steps) == 0:
        mask[:] = False
    directions = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
    tile = np.asarray(state.map_features.tile_type)
    for team in range(2):
        for uid in np.flatnonzero(mask[team]):
            action = int(actions[team][uid, 0])
            if 1 <= action <= 4 and energy[team, uid] >= int(params.unit_move_cost):
                xy = pos[team, uid] + directions[action]
                # Actions produced by adapters stay inside bounds.
                xy = xy.clip(0, 23)
                if tile[tuple(xy)] != 2:
                    pos[team, uid] = xy
                    energy[team, uid] -= int(params.unit_move_cost)
    return pos, energy, mask


def diagnostics(before, after, actions, params, side, cf_state=None):
    pos, energy, mask = movement_snapshot(before, actions, params)
    other = 1 - side
    sap_cost = int(params.unit_sap_cost)
    damage = np.zeros(16, np.int32)
    fired = 0
    for uid in np.flatnonzero(mask[side]):
        a = actions[side][uid]
        target = pos[side, uid] + a[1:]
        if (
            a[0] == 5
            and energy[side, uid] >= sap_cost
            and np.max(np.abs(a[1:])) <= int(params.unit_sap_range)
            and (target >= 0).all()
            and (target < 24).all()
        ):
            fired += 1
            distance = np.abs(pos[other] - target).max(-1)
            damage += (
                np.where(
                    distance == 0,
                    sap_cost,
                    np.where(
                        distance == 1,
                        int(sap_cost * float(params.unit_sap_dropoff_factor)),
                        0,
                    ),
                )
                * mask[other]
            )
    # With multiple simultaneous adjacent hits, the official engine floors the
    # summed dropoff damage once. Recompute exactly rather than summing floors.
    centers = np.zeros(16, int)
    adjacent = np.zeros(16, int)
    for uid in np.flatnonzero(mask[side]):
        a = actions[side][uid]
        target = pos[side, uid] + a[1:]
        if (
            a[0] == 5
            and energy[side, uid] >= sap_cost
            and np.max(np.abs(a[1:])) <= int(params.unit_sap_range)
            and (target >= 0).all()
            and (target < 24).all()
        ):
            d = np.abs(pos[other] - target).max(-1)
            centers += (d == 0) & mask[other]
            adjacent += (d == 1) & mask[other]
    damage = centers * sap_cost + np.floor(
        adjacent * sap_cost * float(params.unit_sap_dropoff_factor)
    ).astype(int)
    effective = np.minimum(damage, np.maximum(0, energy[other] + 1)) * mask[other]
    new_energy = np.asarray(after.units.energy)[..., 0]
    new_mask = np.asarray(after.units_mask)
    sap_kills = 0
    if cf_state is not None:
        cf_energy = np.asarray(cf_state.units.energy)[..., 0]
        sap_kills = int(
            (
                mask[other]
                & new_mask[other]
                & (new_energy[other] < 0)
                & (cf_energy[other] >= 0)
                & (damage > 0)
            ).sum()
        )
    own_lost = np.zeros(16, bool)
    collision_win = collision_loss = collision_tie = 0
    own_cells = {tuple(p) for p in pos[side, mask[side]]}
    enemy_cells = {tuple(p) for p in pos[other, mask[other]]}
    for cell in own_cells & enemy_cells:
        ours = mask[side] & np.all(pos[side] == cell, axis=1)
        theirs = mask[other] & np.all(pos[other] == cell, axis=1)
        a, b = energy[side, ours].sum(), energy[other, theirs].sum()
        collision_win += a > b
        collision_loss += a <= b
        collision_tie += a == b
        if a <= b:
            own_lost |= ours
    # Negative-energy deaths are counted once at the first negative-energy obs.
    death = int(
        own_lost.sum()
        + ((mask[side] & ~own_lost) & new_mask[side] & (new_energy[side] < 0)).sum()
    )
    gains = []
    weights = np.asarray(after.relic_nodes_map_weights)
    active_limit = int(np.asarray(after.relic_nodes_mask).sum()) // 2
    for team in range(2):
        occupied = {
            tuple(p)
            for p, e, m in zip(
                np.asarray(after.units.position)[team],
                new_energy[team],
                new_mask[team],
                strict=True,
            )
            if m and e >= 0
        }
        gains.append(sum(0 < weights[cell] <= active_limit for cell in occupied))
    return dict(
        sap_fired=fired,
        sap_hit=int(((damage > 0) & mask[other]).sum()),
        sap_kill=sap_kills,
        sap_effective_damage=int(effective.sum()),
        sap_energy_spent=fired * sap_cost,
        friendly_fire=0,
        collision_win=int(collision_win),
        collision_loss=int(collision_loss),
        collision_tie=int(collision_tie),
        own_deaths=death,
        alive_units=int((new_mask[side] & (new_energy[side] >= 0)).sum()),
        engine_score_gain=[int(gains[side]), int(gains[other])],
        counterfactual_steps=int(cf_state is not None),
    )


def main():
    env = LuxAIS3Env(auto_reset=False)
    state = obs = params = key = opponent = None
    side = 0
    for line in sys.stdin:
        request = json.loads(line)
        if request["op"] == "close":
            break
        if request["op"] == "reset":
            seed, side = int(request["seed"]), int(request.get("side", 0))
            rng = np.random.default_rng(seed)
            sampled = {
                name: rng.choice(values).item()
                for name, values in env_params_ranges.items()
            }
            params = jax.tree_util.tree_map(jnp.asarray, EnvParams(**sampled))
            key = jax.random.PRNGKey(seed)
            key, subkey = jax.random.split(key)
            obs, state = env.reset(subkey, params)
            public_keys = (
                "map_width",
                "map_height",
                "max_units",
                "max_steps_in_match",
                "match_count_per_episode",
                "unit_move_cost",
                "unit_sap_cost",
                "unit_sap_range",
                "unit_sensor_range",
            )
            cfg = {k: int(getattr(params, k)) for k in public_keys}
            opponent = V0Agent(f"player_{1 - side}", cfg)
            response = dict(
                obs=public(obs[f"player_{side}"]),
                env_cfg=cfg,
                opponent="v0",
                jax_version=jax.__version__,
            )
        else:
            ours = np.asarray(request["actions"], np.int32)
            opponent_obs = public(obs[f"player_{1 - side}"])
            theirs = opponent.act(int(opponent_obs["steps"]), opponent_obs)
            actions = [ours, theirs] if side == 0 else [theirs, ours]
            packed = {f"player_{t}": jnp.asarray(actions[t]) for t in range(2)}
            key, step_key = jax.random.split(key)
            before = state
            obs, state, reward, terminated, truncated, _ = env.step(
                step_key, state, packed, params
            )
            cf_state = None
            if (ours[:, 0] == 5).any():
                counter = ours.copy()
                counter[counter[:, 0] == 5] = 0
                cf_actions = dict(packed)
                cf_actions[f"player_{side}"] = jnp.asarray(counter)
                _, cf_state, _, _, _, _ = env.step(step_key, before, cf_actions, params)
            metrics = diagnostics(before, state, actions, params, side, cf_state)
            response = dict(
                obs=public(obs[f"player_{side}"]),
                done=bool(terminated[f"player_{side}"] | truncated[f"player_{side}"]),
                official_team_wins=[
                    int(reward[f"player_{side}"]),
                    int(reward[f"player_{1 - side}"]),
                ],
                metrics=metrics,
            )
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
