import copy

import numpy as np

import pytest
import torch

from training.contracts import json_write,sha,source_lock
from training.core import models
from training.expert.initialize import load_bc
from training.expert.replays import observation
from training.opponents.pool import Pool


def test_public_replay_view_excludes_hidden_info():
    public={k:None for k in ("steps","match_steps","units","units_mask","sensor_mask",
                             "map_features","relic_nodes","relic_nodes_mask","team_points","team_wins")}
    raw=copy.deepcopy(public)
    raw["energy_nodes"]=[1,2]
    source={"steps":[[{"observation":{"obs":raw},"info":{"replay":{"hidden":"secret"}}}]]}
    assert observation(source,0,0)==public


def test_pfsp_recovery_and_anchor():
    entries={"agents":{"expert":{"enabled":True}}}
    pool=Pool(3101,entries)
    assert pool.weights().sum()==pytest.approx(1)
    assert pool.weights()[pool.ids.index("v2")]>=.35
    pool.observe("expert",-5)
    state=copy.deepcopy(pool.state_dict())
    sequence=[pool.sample() for _ in range(30)]
    other=Pool(999,entries)
    other.load_state_dict(state)
    assert sequence==[other.sample() for _ in range(30)]
    with pytest.raises(RuntimeError):
        Pool(0,{"agents":{}})


@pytest.mark.parametrize("model",list("abc"))
def test_bc_load_is_exact_and_optimizer_not_transferred(tmp_path,model):
    actor,_=models(model,"cpu")
    altered={k:v.clone() for k,v in actor.state_dict().items()}
    key=next(iter(altered))
    altered[key]+=1
    directory=tmp_path/"batch_000000";directory.mkdir()
    state={"schema":"macro_bc_v1","identity":{"model":model,"smoke":True,
           "sources":source_lock(),"teacher":"synthetic_engineering","dataset_sha256":"test"},
           "actor":altered}
    path=directory/"checkpoint.pt"
    torch.save(state,path)
    json_write(directory/"commit.json",{"checkpoint.pt":sha(path)})
    metadata=load_bc(actor,model,path,smoke=True)
    assert torch.equal(actor.state_dict()[key],altered[key]) and metadata["engineering"]
    with pytest.raises(RuntimeError,match="engineering BC"):
        load_bc(actor,model,path,smoke=False)
    wrong="b" if model=="a" else "a"
    with pytest.raises(RuntimeError,match="architecture"):
        load_bc(actor,wrong,path,smoke=True)


def test_game_shuffle_is_complete_and_reproducible():
    from types import SimpleNamespace
    from training.pretrain import shuffled_game_order

    data=SimpleNamespace(rows=[(0,i) for i in range(4)]+[(1,i) for i in range(3)])
    torch.manual_seed(77)
    a=shuffled_game_order(data)
    torch.manual_seed(77)
    b=shuffled_game_order(data)
    assert a==b and sorted(a)==list(range(7))
    groups=[data.rows[i][0] for i in a]
    assert sum(x!=y for x,y in zip(groups,groups[1:]))==1


@pytest.mark.parametrize("model",list("abc"))
@pytest.mark.parametrize("dtype",[np.int32,np.int64])
def test_primitive_ce_accepts_storage_integer_dtypes(model,dtype):
    from training.pretrain import primitive_loss

    torch.manual_seed(3101)
    actor,_=models(model,"cpu")
    auxiliary=torch.nn.Linear(37,6)
    state={"grid":np.zeros((19,24,24),np.float32),
           "position_energy":np.zeros((16,3),np.float32),
           "team":np.zeros(8,np.float32),"alive":np.array([True]+[False]*15)}
    target=np.zeros(16,dtype=dtype)
    target[0]=5
    data=[(state,target)]
    value=primitive_loss(actor,auxiliary,data,[0],"cpu")
    reference=primitive_loss(actor,auxiliary,[(state,target.astype(np.int64))],[0],"cpu")
    assert torch.isfinite(value) and torch.equal(value,reference)
    value.backward()
    assert auxiliary.weight.grad is not None
    assert torch.isfinite(auxiliary.weight.grad).all()
