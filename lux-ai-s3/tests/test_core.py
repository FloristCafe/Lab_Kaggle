import numpy as np
import pytest
import torch

from agents.mappo.shared.credit import OptionLedger, primitive_targets
from agents.mappo.shared.physics import (TerrainBelief, canonical, drift_offset, reachability,
                                        relic_mirror, route_distances, terrain_symmetry, world_index)
from agents.mappo.shared.policy import decide, environment_action
from training.contracts import config, map_pool, setup, training_slot
from training.core import batch, models, prefix_inputs


def sample(model):
    state={"grid": np.zeros((19,24,24),np.float16),
           "units": np.zeros((16,22),np.float16),
           "reach":np.zeros((16,5,576),np.float16),
           "team":np.zeros(6,np.float16),
           "alive":np.arange(16)<3,"review":np.arange(16)<2,
           "legal":np.ones(576,bool),"false_score":np.zeros(576,bool),
           "valid":np.arange(16)==2,"previous":np.zeros(16,np.int64)}
    state["legal"][5]=False
    state["false_score"][:]=True
    if model=="b":
        state["memory"]=np.zeros((5,4,24,24),np.float16)
        state["memory_count"]=np.array(2,np.int64)
    return batch([state],"cpu")


@pytest.mark.parametrize("model",list("abc"))
def test_joint_replay(model):
    setup("cpu", 17)
    actor,critic=models(model,"cpu")
    state=sample(model)
    command,logp,mask,_,base=decide(actor,state,critic=critic,encoded_value=critic.encode(state))
    assert not mask[0,0,0] and mask[0,0,1:].all()
    assert not mask[0,2:].any()
    assert command[0,0,1]>0 and command[0,1,1]>0
    goals=command[0,:2,2].tolist()
    assert len(set(goals))==2 and not any(x in (0,5) for x in goals)
    replay=decide(actor,state,command,critic=critic,encoded_value=critic.encode(state))
    assert torch.equal(logp,replay[1]) and torch.equal(mask,replay[2])
    assert torch.equal(base,replay[-1]) and torch.isfinite(logp).all()
    assert torch.equal(environment_action(state,command,1,mask)["goal"],575-command[...,2])
    (-logp[mask].mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())
    bad=command.clone()
    bad[0,1,2]=bad[0,0,2]
    with pytest.raises(RuntimeError,match="illegal"):
        decide(actor,state,bad)


@pytest.mark.parametrize("model",list("abc"))
def test_continue_probability(model):
    actor,_=models(model,"cpu")
    state=sample(model)
    state["valid"][0,0]=True
    state["previous"][0,0]=17
    state["review"][0,1]=False
    commands=torch.zeros((1,16,3),dtype=torch.long)
    _, logs, masks, _,_=decide(actor,state,commands)
    assert masks[0,0].tolist()==[True,False,False]
    assert logs[0,0,0] < 0 and not logs[0,0,1:].any()
    prefixes=list(prefix_inputs(state,commands))
    assert prefixes[1][0,0] == 17


def test_memory_and_prefix_sensitivity():
    s=sample("b")
    b,_=models("b","cpu")
    a=b(s)[2]
    s["memory"][:,0]=1
    assert not torch.equal(a,b(s)[2])
    a=b(s)[2]
    s["memory"][:,2:]=1
    assert torch.equal(a,b(s)[2])
    c,_=models("c","cpu")
    state=sample("c")
    encoded=c(state)
    p=torch.full((1,16),-1)
    g=c.prefix_adjustment(encoded,1,p)[2]
    p[0,0]=100
    assert not torch.equal(g,c.prefix_adjustment(encoded,1,p)[2])


def test_drift_geometry():
    assert relic_mirror((2,7))==(16,21)
    assert canonical(canonical([2,7],1),1).tolist()==[2,7]
    assert world_index(world_index(55,1),1)==55
    known=np.full((24,24),-1,np.int8)
    known[5,7]=2
    inferred=terrain_symmetry(known,0)
    assert inferred[7,5]==2 and inferred[16,18]==-1
    assert drift_offset(np.float32(.05),0)==0
    tiles=np.zeros((24,24),np.int8)
    tiles[5,7]=tiles[7,5]=2
    belief=TerrainBelief()
    speed=np.float32(.05)
    for i in range(70):
        n=drift_offset(speed,i)
        belief.observe(i,np.roll(tiles,(n,-n),(0,1)))
        assert speed in belief.grids
    future=drift_offset(speed,72)
    assert np.array_equal(belief.forecast(3),np.roll(tiles,(future,-future),(0,1)))


def test_reach_and_primitive_value():
    blocked=np.zeros((24,24),bool)
    blocked[:,1]=True
    assert np.isinf(route_distances(blocked,np.array([[0,0]]))[0,2])
    units=np.zeros((16,22),np.float32)
    p=reachability(np.where(blocked,2,0), units, np.arange(16)==0, 3, np.zeros((24,24)))
    assert p[0,0,2]==1 and p[0,0,24]>0 and not p[1:].any()
    assert primitive_targets([0,3],[1,2],88,True,1,1).tolist()==[3,3]
    assert primitive_targets([0,0],[1,2],3,False,1,1).tolist()==[3,3]
    x=OptionLedger()
    x.active[3]={"start":0}
    x.end(3,1,"match_reset")
    event=x.finish(2,[0,3],1)[0]
    assert event["lifetime_reward"]==0 and event["continuation_reward"]==3


def test_map_protocol():
    pool=map_pool()
    assert not set(pool["training"]) & set(pool["development"])
    assert training_slot(0,pool["training"])[0]==training_slot(900,pool["training"])[0]
    assert training_slot(0,pool["training"])[1]!=training_slot(900,pool["training"])[1]
    assert config("a")["ppo"]==config("b")["ppo"]==config("c")["ppo"]


def test_paired_initialization():
    setup("cpu", 42)
    before = torch.get_rng_state()
    a, ac = models("a", "cpu")
    b, bc = models("b", "cpu")
    c, cc = models("c", "cpu")
    assert torch.equal(before, torch.get_rng_state())
    for k, value in ac.state_dict().items():
        assert torch.equal(value, bc.state_dict()[k]) and torch.equal(value, cc.state_dict()[k])
    for k, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[k]) and torch.equal(value, c.state_dict()[k])


def test_checkpoint_commit_authority(tmp_path):
    from training.contracts import commit, committed

    attempt = tmp_path / "attempt"
    attempt.mkdir()
    staged = attempt / "staging"
    staged.mkdir()
    assert committed(tmp_path) is None
    path = commit(tmp_path, staged, 0, {"state": torch.arange(3)}, {"games": 1})
    assert committed(tmp_path) == path
    (attempt / "interrupted_stage").mkdir()
    assert committed(tmp_path) == path
    with (path / "metrics.json").open("a") as out:
        out.write("tampered")
    with pytest.raises(RuntimeError, match="receipt"):
        committed(tmp_path)


def test_match_boundary_does_not_recount_old_units():
    from types import SimpleNamespace
    from agents.mappo.shared.physics import PhysicsState

    state = PhysicsState(0, {"max_steps_in_match": 100})
    obs = {"map_features": {"tile_type": np.zeros((24,24),np.int8)}}
    seen = SimpleNamespace(energy_seen=np.zeros((24,24)), energy_field=np.zeros((24,24)), known_relics={})
    base = np.zeros((21,24,24),np.float32)
    base[13:17,0,0] = 1
    base[18,0,0] = 1
    sensor = np.ones((24,24),bool)
    for step in range(102):
        state.observe(obs, {"step":step,"match_step":step%101,"base":base,"sensor":sensor}, seen)
    assert state.match_index == 1 and state.completed[0,2,0,0] == 1
    assert state.last_spatial[2,0,0] == 0
    assert not state.spatial[15:19].any()
