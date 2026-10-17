"""Pinned official CPU engine. Only legal observations/rewards cross IPC.

No replay, hidden-map diagnostics, counterfactual simulation or hidden params.
"""
import json
import os
import sys
from pathlib import Path

os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false')
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

if not hasattr(jax,'tree_map'):jax.tree_map=jax.tree_util.tree_map
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'agents/v2'))
from agent import Agent as V2Agent
from agents.v0.agent import Agent as V0Agent
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams,env_params_ranges


def public(x):
    if isinstance(x,dict):return {k:public(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [public(v) for v in x]
    return x.tolist() if hasattr(x,'tolist') else x


def main():
    engine_source=Path(sys.modules[LuxAIS3Env.__module__].__file__).resolve()
    if engine_source!=(ROOT/'vendor/lux-design-s3/src/luxai_s3/env.py').resolve():
        raise RuntimeError('project Python imported a different official engine')
    env=LuxAIS3Env(auto_reset=False);state=obs=params=key=opponent=None;side=0
    for line in sys.stdin:
        request=json.loads(line)
        if request['op']=='close':break
        if request['op']=='reset':
            seed,side=int(request['seed']),int(request['side']);rng=np.random.default_rng(seed)
            params=jax.tree_util.tree_map(jnp.asarray,EnvParams(**{k:rng.choice(v).item() for k,v in env_params_ranges.items()}))
            key,sub=jax.random.split(jax.random.PRNGKey(seed));obs,state=env.reset(sub,params)
            keys=('map_width','map_height','max_units','max_steps_in_match','match_count_per_episode',
                  'unit_move_cost','unit_sap_cost','unit_sap_range','unit_sensor_range')
            cfg={k:int(getattr(params,k)) for k in keys}
            opponent_name=request['opponent']
            np.random.seed(seed+103*(1-side))
            opponent={'v2':V2Agent,'v0':V0Agent}[opponent_name](f'player_{1-side}',cfg)
            response={'obs':public(flax.serialization.to_state_dict(obs[f'player_{side}'])),'env_cfg':cfg,'opponent':opponent_name}
        elif request['op']=='step':
            enemy=public(flax.serialization.to_state_dict(obs[f'player_{1-side}']))
            ours=np.asarray(request['actions'],np.int32);theirs=opponent.act(int(enemy['steps']),enemy)
            actions=[ours,theirs] if side==0 else [theirs,ours]
            key,sub=jax.random.split(key)
            obs,state,reward,terminated,truncated,_=env.step(sub,state,{f'player_{i}':jnp.asarray(actions[i]) for i in range(2)},params)
            observation=public(flax.serialization.to_state_dict(obs[f'player_{side}']))
            response={'obs':observation,'official_reward':[float(reward[f'player_{side}']),float(reward[f'player_{1-side}'])],
                'engine_terminal':bool(terminated[f'player_{side}']),'engine_truncated':bool(truncated[f'player_{side}']),
                'task_complete':int(observation['steps']) >= (cfg['max_steps_in_match']+1)*cfg['match_count_per_episode']}
        else:raise ValueError('unknown IPC operation')
        print(json.dumps(response,allow_nan=False),flush=True)


if __name__=='__main__':main()
