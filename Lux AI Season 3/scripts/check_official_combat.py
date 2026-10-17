"""Pinned-engine fixtures for the isolated diagnostic counters, not policy inputs."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.strategic_env_worker import EnvParams, LuxAIS3Env, diagnostics, jax, jnp, np


def main():
    env = LuxAIS3Env(auto_reset=False)
    params = jax.tree_util.tree_map(jnp.asarray, EnvParams(
        unit_sap_cost=31, unit_sap_dropoff_factor=.25, unit_energy_void_factor=0.,
        min_energy_per_tile=0, max_energy_per_tile=0, nebula_tile_energy_reduction=0,
        nebula_tile_drift_speed=0., energy_node_drift_speed=0.))
    _, initial = env.reset(jax.random.PRNGKey(15002), params)
    results = []

    def state(units):
        positions = np.zeros((2,16,2), np.int16)
        energies = np.zeros((2,16,1), np.int16)
        mask = np.zeros((2,16), bool)
        for team, uid, xy, energy in units:
            positions[team, uid], energies[team, uid, 0], mask[team, uid] = xy, energy, True
        return initial.replace(units=initial.units.replace(position=jnp.array(positions), energy=jnp.array(energies)),
                               units_mask=jnp.array(mask), match_steps=jnp.array(2), steps=jnp.array(2),
                               map_features=initial.map_features.replace(tile_type=jnp.zeros((24,24),jnp.int16)))

    def step(before, actions, name):
        packed = {f'player_{t}': jnp.array(actions[t]) for t in (0,1)}
        key = jax.random.PRNGKey(15003)
        _, after, *_ = env.step(key, before, packed, params)
        cf = actions[0].copy()
        cf[cf[:,0] == 5] = 0
        _, counter, *_ = env.step(key, before, {**packed, 'player_0': jnp.array(cf)}, params)
        result = diagnostics(before, after, actions, params, 0, counter)
        results.append({'name': name, 'metrics': result})
        return after, result

    # Center hit kills the opposing unit; a friendly unit in the adjacent blast
    # footprint takes no damage. SAP cost is charged exactly once.
    before = state([(0,0,(7,8),100), (0,1,(10,9),100), (1,0,(10,8),20)])
    actions = [np.zeros((16,3),np.int16), np.zeros((16,3),np.int16)]
    actions[0][0] = (5,3,0)
    after, metrics = step(before, actions, 'center_kill_no_friendly_fire')
    assert int(after.units.energy[0,0,0]) == 69
    assert int(after.units.energy[0,1,0]) == 100
    assert int(after.units.energy[1,0,0]) < 0
    assert metrics['sap_kill'] == 1 and metrics['sap_fired'] == 1
    # Two adjacent dropoff hits floor the combined 15.5 damage once.
    before = state([(0,0,(7,8),100), (0,1,(7,9),100), (1,0,(10,8),100)])
    actions[0][0] = (5,3,1)
    actions[0][1] = (5,4,-1)
    after, metrics = step(before, actions, 'summed_dropoff_floor')
    assert int(after.units.energy[1,0,0]) == 85 and metrics['sap_effective_damage'] == 15
    # Collision uses energy saved BEFORE SAP, despite the stronger unit paying
    # SAP and having less current energy by collision-resolution time.
    before = state([(0,0,(8,8),40), (1,0,(8,8),50)])
    actions = [np.zeros((16,3),np.int16), np.zeros((16,3),np.int16)]
    actions[1][0] = (5,4,4)
    after, metrics = step(before, actions, 'pre_sap_collision_energy')
    assert not bool(after.units_mask[0,0]) and bool(after.units_mask[1,0])
    assert metrics['collision_loss'] == 1 and metrics['own_deaths'] == 1
    before = state([(0,0,(8,8),40), (1,0,(8,8),40)])
    actions[1][0] = 0
    after, metrics = step(before, actions, 'collision_tie_both_removed')
    assert not bool(after.units_mask[0,0]) and not bool(after.units_mask[1,0])
    assert metrics['collision_tie'] == 1
    out = ROOT/'experiments/interface_contract/combat_fixtures.json'
    out.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps({'fixtures':len(results),'passed':True}),flush=True)


if __name__ == '__main__':
    main()
