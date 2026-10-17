"""Conditional branch sampling with immutable autoregressive behavior masks."""
import torch
from torch.distributions import Bernoulli, Categorical

from agents.goal_review.model import bounded_joint_bias


def neutralize_r(actor):
    """Only R's affine output is reset; the original checkpoint is untouched."""
    with torch.no_grad():
        actor.replan_head.weight.zero_()
        actor.replan_head.bias.zero_()


def previous(units):
    xy=(units[...,8:10]*23).round().long().clamp(0,23)
    return units[...,12:17].argmax(-1),xy[...,0]*24+xy[...,1]


def masked(logits,mask):
    if not mask.any(-1).all():
        raise ValueError('empty conditional support')
    return logits.float().masked_fill(~mask,-1e9)


def statistics(outputs,units,actions,return_factors=False):
    """Recompute the sampled action under EXACT rollout conditional supports.

    Branch entropy is a sampled-branch regularizer, not full joint entropy.
    No Mode/Goal gradient is introduced for Continue or fully forced actions.
    """
    r,ml,gl=outputs
    shape=r.shape
    result=ml.sum((-1))*0
    entropy=result.clone();parts=[];factors=[]
    for branch,logits,name,maskname in (
        (0,torch.stack((torch.zeros_like(r),r),-1),'r','r_mask'),
        (1,ml,'mode','mode_mask')):
        support=actions[maskname]
        dist=Categorical(logits=masked(logits,support))
        active=actions['branch_mask'][...,branch]
        lp=dist.log_prob(actions[name].long())
        result=result+torch.where(active,lp,torch.zeros_like(lp))
        factors.append(torch.where(active,lp,torch.zeros_like(lp)))
        entropy=entropy+torch.where(active,dist.entropy(),torch.zeros_like(lp))
        parts.append(torch.where(active,dist.entropy(),torch.zeros_like(lp)))
    flat=gl.float().flatten(-2)
    selected=flat.gather(2,actions['mode'][...,None,None].expand(*shape,1,576)).squeeze(2)
    support=actions['goal_mask'].gather(2,actions['mode'][...,None,None].expand(*shape,1,576)).squeeze(2)
    dist=Categorical(logits=masked(selected,support))
    active=actions['branch_mask'][...,2];lp=dist.log_prob(actions['goal'].long())
    result=result+torch.where(active,lp,torch.zeros_like(lp))
    factors.append(torch.where(active,lp,torch.zeros_like(lp)))
    entropy=entropy+torch.where(active,dist.entropy(),torch.zeros_like(lp))
    parts.append(torch.where(active,dist.entropy(),torch.zeros_like(lp)))
    values=(result,entropy,torch.stack(parts,-1))
    return (*values,torch.stack(factors,-1)) if return_factors else values


def policy_outputs(actor,state,model_cfg):
    outputs=actor(state['history'],state['positions'],state['units'],state['alive'])
    ml,gl=bounded_joint_bias(outputs[1],outputs[2],state['units'],model_cfg)
    return outputs[0],ml,gl


@torch.no_grad()
def sample(actor,state,config,greedy=False):
    """UID order; all existing intents reserve cells until explicitly replaced.

    Invalid intent => rule-forced entire action: deterministic masked joint
    argmax, zero probability factors/PPO mask. A free Gate-Replan samples MG;
    a free learned-R Replan samples R then M then G. Continue never draws MG.
    """
    outputs=policy_outputs(actor,state,config['model']);rlog,ml,gl=outputs
    B,K=rlog.shape;device=rlog.device
    pm,pg=previous(state['units']);valid=(state['units'][...,11]>.5)&state['alive']
    result={'r':torch.zeros((B,K),dtype=torch.bool,device=device),'mode':pm.clone(),'goal':pg.clone(),
        'decision_mask':state['review'].clone(),'forced_action_mask':torch.zeros_like(valid),
        'branch_mask':torch.zeros((B,K,3),dtype=torch.bool,device=device),
        'r_mask':torch.ones((B,K,2),dtype=torch.bool,device=device),
        'mode_mask':torch.ones((B,K,5),dtype=torch.bool,device=device),
        'goal_mask':torch.ones((B,K,5,576),dtype=torch.bool,device=device),
        'order':torch.arange(K,device=device).expand(B,-1).clone()}
    result['stochastic']=torch.full((B,),not greedy,dtype=torch.bool,device=device)
    draws={'R':0,'Mode':0,'Goal':0}
    for b in range(B):
        owners={}
        for u in range(K):
            if valid[b,u]:owners.setdefault(int(pg[b,u]),set()).add(u)
        for u in result['order'][b].tolist():
            if not state['review'][b,u] or not state['alive'][b,u]:continue
            for cell in list(owners):
                owners[cell].discard(u)
                if not owners[cell]:del owners[cell]
            forced=not bool(valid[b,u])
            # An inherited conflict is a deterministic protocol interruption.
            conflict=bool(valid[b,u] and int(pg[b,u]) in owners)
            forced=forced or conflict
            result['forced_action_mask'][b,u]=forced
            result['r_mask'][b,u]=torch.tensor([not forced,True],device=device)
            if forced:
                replan=True
            elif config['policy']['gate']=='deterministic':
                replan=bool(state['gate_replan'][b,u])
                result['r_mask'][b,u]=torch.tensor([not replan,replan],device=device)
            else:
                if greedy:replan=bool(rlog[b,u]>=0)
                else:replan=bool(Bernoulli(logits=rlog[b,u].float()).sample());draws['R']+=1
                result['branch_mask'][b,u,0]=True
            result['r'][b,u]=replan
            if not replan:
                owners.setdefault(int(pg[b,u]),set()).add(u)
                continue
            support=state['legal_goals'][b].clone()
            if owners:support[list(owners)]=False
            if not support.any():raise ValueError('no unreserved legal goal; never silently relax mask')
            result['goal_mask'][b,u]=support[None].expand(5,-1)
            mode_support=result['goal_mask'][b,u].any(-1)
            result['mode_mask'][b,u]=mode_support
            mode_logits=masked(ml[b,u],mode_support)
            goal_logits=masked(gl[b,u].flatten(-2),result['goal_mask'][b,u])
            if forced or greedy:
                joint=mode_logits.log_softmax(-1)[:,None]+goal_logits.log_softmax(-1)
                choice=int(joint.flatten().argmax());m,g=choice//576,choice%576
            else:
                m=int(Categorical(logits=mode_logits).sample());draws['Mode']+=1
                g=int(Categorical(logits=goal_logits[m]).sample());draws['Goal']+=1
            result['mode'][b,u]=m;result['goal'][b,u]=g
            if not forced:result['branch_mask'][b,u,1:]=True
            owners.setdefault(g,set()).add(u)
    result['policy_mask']=result['branch_mask'].any(-1)
    lp,entropy,_,factors=statistics(outputs,state['units'],result,return_factors=True)
    result['old_logprob']=lp.detach().clone()
    result['old_logprob_factors']=factors.detach().clone()
    result['draw_counts']=draws
    return result,entropy


def optimizer(actor,critic,config):
    lr=config['ppo'];groups={'encoder':[],'mode_goal':[],'R':[]}
    for name,p in actor.named_parameters():
        if not p.requires_grad:continue
        key='R' if name.startswith('replan_head.') else 'encoder' if name.startswith(('encoder.','unit_embedding.')) else 'mode_goal'
        if key=='R' and config['policy']['gate']=='deterministic':p.requires_grad_(False);continue
        groups[key].append(p)
    specs=[{'params':ps,'lr':lr[key+'_lr'],'name':key} for key,ps in groups.items() if ps]
    specs.append({'params':list(critic.parameters()),'lr':lr['critic_lr'],'name':'critic'})
    ids=[id(p) for group in specs for p in group['params']]
    if len(ids)!=len(set(ids)):raise ValueError('parameter appears in multiple optimizer groups')
    return torch.optim.Adam(specs,eps=1e-5)
