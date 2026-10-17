"""Independent public-state Critic, CPU team buffer and duration-aware PPO."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from agents.strategic.model import HistoryEncoder
from agents.strategic.semimdp import smdp_gae,clipped_actor_loss
from .policy import policy_outputs,statistics


class TeamCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=HistoryEncoder()  # independently initialized; never copy Actor
        self.context=nn.Sequential(nn.Conv2d(12,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,16,3,stride=2,padding=1),nn.SiLU())
        self.units=nn.Sequential(nn.Linear(30,32),nn.SiLU())
        self.value=nn.Sequential(nn.Linear(32+16+64+12,64),nn.SiLU(),nn.Linear(64,1))

    def forward(self,state):
        spatial=self.encoder(state['history']).mean((2,3))
        context=self.context(state['critic_maps']).mean((2,3))
        mask=state['alive'];x=self.units(state['critic_units'])
        mean=(x*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        maximum=x.masked_fill(~mask[...,None],-1e9).max(1).values
        maximum=torch.where(mask.any(1)[:,None],maximum,torch.zeros_like(maximum))
        return self.value(torch.cat((spatial,context,mean,maximum,state['team']),-1)).squeeze(-1)


def cpu_copy(data):
    return {k:v.detach().cpu().clone() for k,v in data.items() if isinstance(v,torch.Tensor)}


def device(data,where='cuda'):
    return {k:v.to(where) for k,v in data.items()}


@dataclass
class Transition:
    state:dict
    next_state:dict
    action:dict
    reward:float
    tau:int
    value:float
    next_value:float
    terminal:bool
    truncated:bool
    episode:int
    engine_terminal:bool=False
    engine_truncated:bool=False
    task_complete:bool=False
    primitive_rewards:tuple=()

    def __post_init__(self):
        if self.tau<1 or not np.isfinite([self.reward,self.value,self.next_value]).all():raise ValueError('invalid SMDP transition')
        if self.terminal and self.truncated:raise ValueError('task terminal and external truncation are distinct')
        if self.primitive_rewards and len(self.primitive_rewards)!=self.tau:raise ValueError('primitive reward duration mismatch')
        self.state=cpu_copy(self.state);self.next_state=cpu_copy(self.next_state);self.action=cpu_copy(self.action)


def returns(buffer,gamma,lam):
    terminal=np.array([t.terminal for t in buffer]);truncated=np.array([t.truncated for t in buffer])
    links=np.array([i+1<len(buffer) and t.episode==buffer[i+1].episode and not t.terminal and not t.truncated
                    for i,t in enumerate(buffer)])
    return smdp_gae([t.reward for t in buffer],[t.tau for t in buffer],[t.value for t in buffer],
                    [t.next_value for t in buffer],terminal,links&~truncated,gamma,lam,True)


def save_rollout(path,buffer,gamma):
    """Lossless float32 state journals, bit-packed masks and ZLIB; no raw JSON."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('immutable behavior journal already exists')
    arrays={}
    for prefix,attribute in (('state','state'),('next','next_state'),('action','action')):
        for key in getattr(buffer[0],attribute):
            value=torch.cat([getattr(t,attribute)[key] for t in buffer]).numpy()
            arrays[f'{prefix}__{key}']=np.packbits(value,axis=-1) if key=='goal_mask' else value
    for name in ('reward','tau','value','next_value','terminal','truncated','episode','engine_terminal','engine_truncated','task_complete'):
        arrays[name]=np.asarray([getattr(t,name) for t in buffer])
    primitive=np.zeros((len(buffer),max(t.tau for t in buffer)),np.float32)
    for i,t in enumerate(buffer):
        if t.primitive_rewards:primitive[i,:t.tau]=t.primitive_rewards
    arrays['primitive_rewards']=primitive
    arrays['metadata']=np.asarray(json.dumps({'schema':'mappo_v5_rollout_v1','gamma':gamma,
        'goal_mask_encoding':'packbits last axis, 576 bits','phase':'stochastic training only',
        'state_precision':'lossless original float32, no new feature quantization'}))
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as file:np.savez_compressed(file,**arrays)
    tmp.replace(path);return path.stat().st_size


def load_rollout(path):
    with np.load(path,allow_pickle=False) as z:data={key:z[key] for key in z.files}
    data['action__goal_mask']=np.unpackbits(data['action__goal_mask'],axis=-1,count=576).astype(bool)
    return data


def update(actor,critic,opt,buffer,config):
    p=config['ppo'];adv,ret=returns(buffer,p['gamma'],p['lambda'])
    raw_adv=adv.copy()
    state={k:torch.cat([t.state[k] for t in buffer]) for k in buffer[0].state}
    actions={k:torch.cat([t.action[k] for t in buffer]) for k in buffer[0].action}
    if 'stochastic' in actions and not actions['stochastic'].all():raise ValueError('greedy evaluation density is not PPO behavior')
    N=len(buffer);old=actions['old_logprob'];valid=actions['policy_mask']
    adv=torch.as_tensor(adv,dtype=torch.float32);ret=torch.as_tensor(ret,dtype=torch.float32)
    if valid.any(1).sum()>1:
        selected=adv[valid.any(1)];adv=(adv-selected.mean())/selected.std(unbiased=False).clamp_min(1e-6)
    actor.train();critic.train();logs=[];error=0.;where=next(actor.parameters()).device
    # Before ANY update, recompute all original behavior likelihoods.
    with torch.no_grad():
        for start in range(0,N,p['minibatch']):
            ids=slice(start,start+p['minibatch']);b=device({k:v[ids] for k,v in state.items()},where)
            a=device({k:v[ids] for k,v in actions.items()},where)
            lp,_,_=statistics(policy_outputs(actor,b,config['model']),b['units'],a)
            error=max(error,float((lp.cpu()-old[ids]).abs().max()))
    if error>1e-4:raise RuntimeError(f'old behavior probability mismatch BEFORE update: {error}')
    for _ in range(p['epochs']):
        for ids in torch.randperm(N).split(p['minibatch']):
            b=device({k:v[ids] for k,v in state.items()},where);a=device({k:v[ids] for k,v in actions.items()},where)
            lp,h,parts=statistics(policy_outputs(actor,b,config['model']),b['units'],a)
            loss_actor,clip=clipped_actor_loss(lp,old[ids].to(where),adv[ids].to(where),a['policy_mask'],p['clip'])
            count=a['policy_mask'].sum(1).clamp_min(1)
            hmean=((h*a['policy_mask']).sum(1)/count)[a['policy_mask'].any(1)].mean() if a['policy_mask'].any() else h.sum()*0
            value=critic(b);loss_value=F.mse_loss(value,ret[ids].to(where))
            loss=loss_actor+p['value_weight']*loss_value-p['entropy_weight']*hmean
            if not torch.isfinite(loss):raise RuntimeError('nonfinite PPO loss')
            opt.zero_grad(set_to_none=True);loss.backward()
            for group in opt.param_groups:
                inactive=(group['name']=='R' and not a['branch_mask'][...,0].any()
                    or group['name']=='mode_goal' and not a['branch_mask'][...,1:].any()
                    or group['name']=='encoder' and not a['policy_mask'].any())
                if inactive:
                    # Skip Adam momentum as well as gradients for absent branches.
                    for parameter in group['params']:parameter.grad=None
            grads={g['name']:float(torch.sqrt(sum((x.grad.float().square().sum() for x in g['params'] if x.grad is not None),torch.zeros((),device=where)))) for g in opt.param_groups}
            for branch,prefixes in (('Mode',('mode_head.',)),('Goal',('mode_queries.','goal_keys.','goal_bias.','cost_weight.'))):
                grads[branch]=float(torch.sqrt(sum((x.grad.float().square().sum() for name,x in actor.named_parameters()
                    if name.startswith(prefixes) and x.grad is not None),torch.zeros((),device=where))))
            norm=nn.utils.clip_grad_norm_([x for g in opt.param_groups for x in g['params']],p['grad_clip'],error_if_nonfinite=True)
            opt.step()
            if any(not torch.isfinite(x).all() for g in opt.param_groups for x in g['params']):raise RuntimeError('nonfinite parameter')
            logratio=(lp-old[ids].to(where))[a['policy_mask']]
            kl=float((logratio.exp()-1-logratio).mean().detach()) if logratio.numel() else 0.
            logs.append({'actor_loss':float(loss_actor.detach()),'value_loss':float(loss_value.detach()),
                'clip_fraction':float(clip.detach()),'approx_kl':kl,'entropy':float(hmean.detach()),'grad_norm':float(norm),
                **{f'grad_{k}':v for k,v in grads.items()},
                **{f'entropy_{name}':float(parts[...,i][a['branch_mask'][...,i]].mean().detach()) if a['branch_mask'][...,i].any() else 0.
                   for i,name in enumerate(('R','Mode','Goal'))}})
    v=np.array([t.value for t in buffer]);y=ret.numpy()
    with torch.no_grad():
        after=torch.cat([critic(device({k:x[start:start+p['minibatch']] for k,x in state.items()},where)).cpu()
                         for start in range(0,N,p['minibatch'])]).numpy()
    return {'options':N,'primitive_steps':sum(t.tau for t in buffer),'first_logprob_error':error,
        'optimizer_steps':len(logs),'reward_nonzero':sum(t.reward!=0 for t in buffer),
        'reward_mean':float(np.mean([t.reward for t in buffer])),'return_mean':float(y.mean()),
        'advantage_mean':float(adv.mean()),'advantage_std':float(adv.std(unbiased=False)),
        'raw_advantage_mean':float(raw_adv.mean()),'raw_advantage_std':float(raw_adv.std()),
        'raw_advantage_q05':float(np.quantile(raw_adv,.05)),'raw_advantage_q95':float(np.quantile(raw_adv,.95)),
        'critic_mse_before':float(np.mean((v-y)**2)),'critic_mse_after':float(np.mean((after-y)**2)),
        'explained_variance_before':float(1-np.var(y-v)/np.var(y)) if np.var(y)>1e-12 else None,
        'explained_variance_after':float(1-np.var(y-after)/np.var(y)) if np.var(y)>1e-12 else None,
        'free_R_decisions':int(actions['branch_mask'][...,0].sum()),
        'MG_decisions':int(actions['branch_mask'][...,1].sum()),
        **{k:float(np.mean([log[k] for log in logs])) for k in logs[0]}}
