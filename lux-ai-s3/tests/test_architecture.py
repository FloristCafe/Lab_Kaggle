"""CPU-only structural checks for the retained historical prototypes."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agents.mappo.model_a import Actor as A
from agents.mappo.model_b import Actor as B
from agents.mappo.model_c import Actor as C
from agents.mappo.shared.model import PrefixCritic
from agents.mappo.shared.policy import decide
from agents.mappo.shared.physics import canonical,relic_mirror,world_index
from agents.v2.equation_solver import EquationSolver
from agents.v2.obs_parser import ObsParser


def sample():
    return {"grid":torch.zeros(1,19,24,24),
            "units":torch.zeros(1,16,22),"team":torch.zeros(1,6),
            "reach":torch.zeros(1,16,5,576),
            "alive":(torch.arange(16)<3)[None],"review":(torch.arange(16)<2)[None],
            "valid":(torch.arange(16)==2)[None],"previous":torch.zeros(1,16,dtype=torch.long),
            "legal":torch.ones(1,576,dtype=torch.bool),"false_score":torch.zeros(1,576,dtype=torch.bool),
            "memory":torch.zeros(1,5,4,24,24),"memory_count":torch.tensor([2])}


@pytest.mark.parametrize("actor_type,parameters",[(A,57220),(B,61956),(C,63824)])
def test_forward_and_conditional_replay(actor_type,parameters):
    torch.set_num_threads(2);torch.manual_seed(17)
    actor=actor_type();state=sample()
    assert sum(p.numel() for p in actor.parameters())==parameters
    r,mode,goal,tiles,units=actor(state)
    assert r.shape==(1,16,2) and mode.shape==(1,16,5) and goal.shape==(1,16,5,576)
    assert all(torch.isfinite(x).all() for x in (r,mode,goal,tiles,units))
    command,logp,active,_,_=decide(actor,state)
    replay=decide(actor,state,recorded=command)
    assert torch.equal(logp,replay[1]) and torch.equal(active,replay[2])
    chosen=command[0,:2,2].tolist()
    assert len(set(chosen))==2 and 0 not in chosen
    assert not active[0,2:].any()


def test_causal_memory_boundary():
    torch.set_num_threads(2);torch.manual_seed(17)
    actor=B();state=sample();base=actor(state)[2]
    state["memory"][:,2:]=1
    assert torch.equal(base,actor(state)[2])
    state["memory"][:,0]=1
    assert not torch.equal(base,actor(state)[2])


def test_assignment_context_is_used():
    torch.set_num_threads(2);torch.manual_seed(17)
    actor=C();encoded=actor(sample());claims=torch.full((1,16),-1)
    before=actor.prefix_adjustment(encoded,1,claims)[2]
    claims[0,0]=100
    assert not torch.equal(before,actor.prefix_adjustment(encoded,1,claims)[2])


def test_critic_forward():
    torch.set_num_threads(2)
    critic=PrefixCritic();value=critic(sample())
    assert value.shape==(1,) and torch.isfinite(value).all()


def test_geometry_and_public_belief_imports():
    assert relic_mirror((2,7))==(16,21)
    assert canonical(canonical([2,7],1),1).tolist()==[2,7]
    assert world_index(world_index(55,1),1)==55
    solver=EquationSolver(24,24)
    parser=ObsParser(solver=solver,team_id=0,max_steps_in_match=100)
    assert parser is not None and np.isfinite(np.zeros((24,24))).all()
