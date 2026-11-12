"""Synthetic, CPU-only checks: no fitting, datasets, workers or trajectories."""
import numpy as np
import pytest
import torch
from torch import nn

from research.representation_diagnostics.goal_heads import Effective, GoalHead, Nonlinear
from research.representation_diagnostics.metrics import ranking
from research.representation_diagnostics.oracle import solve
from research.representation_diagnostics.ranker import Ranker, node_loss
from research.representation_diagnostics.utility_features import (
    analytic_from_RAW, feature_views, pairs_for_node, public_row,
)

def fixture_actor():
    actor=nn.Module()
    actor.mode_queries=nn.Linear(32,160)
    actor.goal_keys=nn.Conv2d(32,32,1)
    actor.goal_bias=nn.Conv2d(32,5,1)
    actor.cost_weight=nn.Linear(32,10)
    x,y=torch.meshgrid(torch.arange(24.),torch.arange(24.),indexing="ij")
    actor.register_buffer("gx",x)
    actor.register_buffer("gy",y)
    return actor

def inputs():
    return (torch.randn(2,32,24,24),torch.randn(2,1,32),
            torch.tensor([[[2.,4.]],[[20.,21.]]]),torch.ones(2,1)*.2,torch.tensor([[1],[3]]))

def test_head_parameter_counts_and_nested_init():
    actor=fixture_actor();base=GoalHead(actor);nonlinear=Nonlinear(actor)
    assert sum(p.numel() for p in base.parameters())==6831
    assert sum(p.numel() for p in nonlinear.parameters())==32688
    batch=inputs()
    assert torch.equal(base.conditional(*batch),nonlinear.conditional(*batch))

def test_effective_parameterization_preserves_conditional_distribution():
    head=GoalHead(fixture_actor()).double()
    batch=[x.double() if x.is_floating_point() else x for x in inputs()]
    original=head.conditional(*batch)[:,0]
    effective=Effective(head)(dict(zip(("spatial","query","positions","move_cost","mode"),batch)))
    # Raw logits may differ by a per-node constant.
    assert torch.allclose(original.log_softmax(-1),effective.log_softmax(-1),atol=1e-12,rtol=1e-12)

@pytest.mark.parametrize("mode",range(5))
def test_team_nonlinear_matches_individual_queries(mode):
    model=Nonlinear(fixture_actor());spatial,query,pos,cost,_=inputs()
    query=query.expand(-1,2,-1);pos=pos.expand(-1,2,-1);cost=cost.expand(-1,2)
    modes=torch.full((2,2),mode)
    team=model.conditional(spatial,query,pos,cost,modes)
    single=model.conditional(spatial,query[:,:1],pos[:,:1],cost[:,:1],modes[:,:1])
    # Different GEMM batch shapes may change FP32 accumulation order.
    assert torch.allclose(team[:,:1],single,atol=2e-6,rtol=1e-6)

def test_raw_reconstructs_public_utility_without_teacher_features():
    rng=np.random.default_rng(7)
    history=rng.random((3,18,24,24))
    history[:,1]=(history[:,1]>.5);history[:,7]=(history[:,7]>.5)
    history[0,8]=rng.choice([0.,.25,.5,1.],(24,24))
    raw,enc=feature_views(history,np.zeros((32,24,24)),(2,20),np.zeros(22),1,178,77,2)
    assert raw.shape==enc.shape==(576,165)
    assert np.allclose(analytic_from_RAW(raw),public_row(history,(2,20),77,100,2),atol=2e-6,rtol=2e-6)

def test_fixed_pair_slots_reproducible_and_ties_not_invented():
    row=np.arange(576,dtype=float);mask=np.zeros(576,bool);mask[[1,2,4,8,9,15]]=True
    pair,valid=pairs_for_node(row,mask,[1,2,3])
    again,v=pairs_for_node(row,mask,[1,2,3])
    assert np.array_equal(pair,again) and np.array_equal(valid,v)
    assert pair.shape==(32,2) and valid.all() and mask[pair].all()
    _,tied=pairs_for_node(np.zeros(576),mask,[1,2,3])
    assert not tied.any()

def test_ranker_parameter_count_and_equal_node_reduction():
    assert sum(p.numel() for p in Ranker().parameters())==5345
    scores=torch.zeros(2,32,2,requires_grad=True)
    signs=torch.ones(2,32)
    valid=torch.zeros(2,32);valid[0,0]=1;valid[1,:]=1
    losses=node_loss(scores,signs,valid)
    assert torch.allclose(losses,torch.full((2,),np.log(2),dtype=losses.dtype))
    losses.mean().backward()
    assert torch.isfinite(scores.grad).all()

def test_ranking_ties_and_empty_support():
    support=np.ones(3,bool)
    metric=ranking(np.array([1.,1.,0.]),np.array([3.,3.,0.]),support)
    assert metric["best_hit"] and metric["pair_acc"]==1 and metric["regret"]==0
    assert ranking(np.ones(3),np.ones(3),support)["pair_acc"] is None
    assert ranking(np.ones(3),np.ones(3),~support)["n"]==0

def test_per_node_oracle_is_relaxed_not_shared():
    spatial=np.zeros((32,24,24));mask=np.zeros(576,bool);mask[[1,24]]=True
    tied=solve(spatial,np.array([0,0]),mask,1)
    assert not tied["positive_margin"]
    spatial[0,0,1]=1
    assert solve(spatial,np.array([0,0]),mask,1)["positive_margin"]
