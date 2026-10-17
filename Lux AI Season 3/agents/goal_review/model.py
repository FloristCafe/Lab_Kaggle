"""Capacity-only spatial variants and a separate Bernoulli macro review head."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from agents.strategic.model import StrategicActor, action_statistics, choose_actions, conv


class ResidualBlock(nn.Module):
    def __init__(self, width, dilation=1):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(width,width,3,padding=dilation,dilation=dilation),
            nn.GroupNorm(4,width),nn.SiLU(),nn.Conv2d(width,width,3,padding=dilation,dilation=dilation),nn.GroupNorm(4,width))

    def forward(self,x):
        return F.silu(x+self.layers(x))


class ResidualHistoryEncoder(nn.Module):
    """Retain the EXACT grouped encoders/ConvGRU; replace spatial path only."""
    def __init__(self, original, width, dilations):
        super().__init__()
        self.groups = original.groups
        self.encoders,self.gates,self.candidate = original.encoders,original.gates,original.candidate
        self.register_buffer('coordinates',original.coordinates.clone())
        self.stem = conv(34,width)
        self.blocks = nn.Sequential(*(ResidualBlock(width,d) for d in dilations))
        self.project = nn.Conv2d(width,32,1)

    def forward(self,history):
        if tuple(history.shape[1:]) != (3,18,24,24):
            raise ValueError('require causal [B,3,18,24,24]')
        hidden=history.new_zeros((len(history),32,24,24))
        for t in (2,1,0):
            encoded=torch.cat([m(history[:,t,list(g)]) for m,g in zip(self.encoders,self.groups,strict=True)],1)
            reset,update=self.gates(torch.cat((encoded,hidden),1)).sigmoid().chunk(2,1)
            candidate=self.candidate(torch.cat((encoded,reset*hidden),1)).tanh()
            hidden=(1-update)*hidden+update*candidate
        x=self.stem(torch.cat((hidden,self.coordinates.expand(len(history),-1,-1,-1)),1))
        return self.project(self.blocks(x))


class ReviewActor(StrategicActor):
    def __init__(self, cfg):
        super().__init__()
        self.cfg=dict(cfg)
        self.use_replan=bool(cfg.get('replan',False))
        arch=cfg.get('architecture','current')
        variants={'res6':(64,[1]*6),'res8':(128,[1]*8),'dilated6':(64,[1,2,3,5,8,4])}
        if arch in variants:
            self.encoder=ResidualHistoryEncoder(self.encoder,*variants[arch])
        elif arch!='current':
            raise ValueError('unknown architecture')
        self.replan_head=nn.Linear(32,1) if self.use_replan else None
        self.legacy_previous_bias=bool(cfg.get('legacy_previous_bias',True))
        if not self.legacy_previous_bias:
            self.previous_weight.requires_grad_(False)

    def forward(self,history,positions,units,alive):
        spatial=self.encoder(history)
        B,K,_=positions.shape
        pos=positions.clamp(0,23)
        ids=pos[:,:,0]*24+pos[:,:,1]
        local=spatial.flatten(2).gather(2,ids[:,None].expand(-1,32,-1)).transpose(1,2)
        query=self.unit_embedding(torch.cat((local,units),-1))
        ml=self.mode_head(query)
        q=self.mode_queries(query).reshape(B,K,5,32)
        gl=torch.einsum('bkmd,bdxy->bkmxy',q,self.goal_keys(spatial))/math.sqrt(32)
        gl=gl+self.goal_bias(spatial)[:,None]
        distance=(self.gx-pos[:,:,0,None,None]).abs()+(self.gy-pos[:,:,1,None,None]).abs()
        costs=torch.stack((distance/46,distance*(units[:,:,19,None,None]*5)/400),2)
        gl=gl+torch.einsum('bkmc,bkcxy->bkmxy',self.cost_weight(query).reshape(B,K,5,2),costs)
        if self.legacy_previous_bias:
            previous=(units[:,:,8:10]*23).round().long().clamp(0,23)
            mask=F.one_hot(previous[:,:,0]*24+previous[:,:,1],576).reshape(B,K,24,24)*units[:,:,11,None,None]
            gl=gl+mask[:,:,None]*self.previous_weight[None,None,:,None,None]
        r=self.replan_head(query).squeeze(-1) if self.use_replan else ml[...,0]*0
        return r,ml,gl


def bounded_joint_bias(ml,gl,units,cfg):
    eta=min(max(float(cfg.get('hysteresis',0)),0),float(cfg.get('hysteresis_cap',.5)))
    if not eta:
        return ml,gl
    previous=(units[:,:,8:10]*23).round().long().clamp(0,23)
    pm=units[:,:,12:17].argmax(-1)
    goal=F.one_hot(previous[:,:,0]*24+previous[:,:,1],576).reshape(*pm.shape,24,24)
    pair=F.one_hot(pm,5)[:,:,:,None,None]*goal[:,:,None]
    decay=(1-units[:,:,10]*100/cfg.get('hysteresis_max_age',80)).clamp(0,1)
    shift=eta*pair*decay[:,:,None,None,None]*units[:,:,11,None,None,None]
    updated=gl+shift
    # Compensate conditional normalization: this is exactly a bounded bias on
    # ONE (mode,goal) pair in the full joint distribution, not all five modes.
    correction=torch.logsumexp(updated.float().flatten(-2),-1)-torch.logsumexp(gl.float().flatten(-2),-1)
    return ml+correction,updated


def review_statistics(r_logits,ml,gl,replan,modes,goals,units,alive,use_replan=True):
    lp,conditional_entropy=action_statistics(ml,gl,modes,goals)
    valid_previous=(units[:,:,11]>.5)&alive
    if use_replan and (alive & ~valid_previous & ~replan.bool()).any():
        raise ValueError('a forced Replan cannot have a Continue behavior logprob')
    if use_replan:
        r_lp=-F.binary_cross_entropy_with_logits(r_logits.float(),replan.float(),reduction='none')
        probability=r_logits.float().sigmoid()
        h=-(probability*F.logsigmoid(r_logits.float())+(1-probability)*F.logsigmoid(-r_logits.float()))
        # Forced Replan when no valid goal is a deterministic constraint, P=1.
        r_lp=torch.where(valid_previous,r_lp,torch.zeros_like(r_lp))
        p=torch.where(valid_previous,probability,torch.ones_like(probability))
        entropy=torch.where(valid_previous,h,torch.zeros_like(h))+p*conditional_entropy
        lp=r_lp+torch.where(replan,lp,torch.zeros_like(lp))
    else:
        entropy=conditional_entropy
    return torch.where(alive,lp,torch.zeros_like(lp)),torch.where(alive,entropy,torch.zeros_like(entropy))


def select_review_actions(outputs,units,alive,cfg,greedy=False):
    r_logits,ml,gl=outputs
    ml,gl=bounded_joint_bias(ml,gl,units,cfg)
    enabled=bool(cfg.get('replan',False))
    if enabled:
        r=(r_logits.sigmoid()>=cfg.get('replan_threshold',.5)) if greedy else torch.distributions.Bernoulli(logits=r_logits.float()).sample().bool()
        r=r | ~(units[:,:,11]>.5)
    else:
        r=torch.ones_like(alive)
    modes,goals,_=choose_actions(ml,gl,greedy=greedy)
    previous=(units[:,:,8:10]*23).round().long().clamp(0,23)
    pm=units[:,:,12:17].argmax(-1)
    modes=torch.where(r,modes,pm)
    goals=torch.where(r,goals,previous[:,:,0]*24+previous[:,:,1])
    lp,entropy=review_statistics(r_logits,ml,gl,r,modes,goals,units,alive,enabled)
    return r,modes,goals,lp,entropy
