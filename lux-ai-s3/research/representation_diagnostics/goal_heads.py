"""Historical score families; device-neutral extraction, no fitting entry."""
import copy

import torch
from torch import nn

class GoalHead(nn.Module):
    names=("mode_queries","goal_keys","goal_bias","cost_weight")
    def __init__(self,actor):
        super().__init__()
        for n in self.names:setattr(self,n,copy.deepcopy(getattr(actor,n)))
        self.register_buffer("gx",actor.gx.detach().clone())
        self.register_buffer("gy",actor.gy.detach().clone())
        self.requires_grad_(True)
    def forward(self,spatial,query,positions,move_cost):
        b,k,_=positions.shape
        q=self.mode_queries(query).reshape(b,k,5,32)
        gl=torch.einsum("bkmd,bdxy->bkmxy",q,self.goal_keys(spatial))/(32**.5)
        gl=gl+self.goal_bias(spatial)[:,None]
        pos=positions.clamp(0,23)
        distance=(self.gx-pos[:,:,0,None,None]).abs()+(self.gy-pos[:,:,1,None,None]).abs()
        costs=torch.stack((distance/46,distance*(move_cost[:,:,None,None]*5)/400),2)
        gl=gl+torch.einsum("bkmc,bkcxy->bkmxy",self.cost_weight(query).reshape(b,k,5,2),costs)
        return gl.flatten(-2)
    def conditional(self,spatial,query,positions,move_cost,mode):
        all_logits=self(spatial,query,positions,move_cost)
        return all_logits.gather(2,mode[:,:,None,None].expand(-1,-1,1,576)).squeeze(2)

def candidate_features(spatial,query,positions,move_cost,mode):
    if query.shape[1]!=1:raise ValueError("one unit for residual features")
    axis=torch.arange(24,device=spatial.device,dtype=spatial.dtype);gx,gy=torch.meshgrid(axis,axis,indexing="ij")
    pos=positions[:,0].clamp(0,23)
    dist=(gx.flatten()[None]-pos[:,0,None]).abs()+(gy.flatten()[None]-pos[:,1,None]).abs()
    costs=torch.stack((dist/46,dist*move_cost[:,0,None]*5/400),-1)
    return torch.cat((spatial.flatten(2).transpose(1,2),query[:,0,None].expand(-1,576,-1),
       torch.nn.functional.one_hot(mode[:,0],5).float()[:,None].expand(-1,576,-1),costs),-1)

class Nonlinear(nn.Module):
    """Exact D02 32688 architecture, generalized to team inference."""
    def __init__(self,actor):
        super().__init__();self.base=GoalHead(actor)
        self.residual=nn.Sequential(nn.Linear(71,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
    def conditional(self,spatial,query,positions,move_cost,mode):
        base=self.base.conditional(spatial,query,positions,move_cost,mode)
        residual=[]
        for u in range(query.shape[1]):
            f=candidate_features(spatial,query[:,u:u+1],positions[:,u:u+1],move_cost[:,u:u+1],mode[:,u:u+1])
            residual.append(self.residual(f).squeeze(-1))
        return base+torch.stack(residual,1)

class Effective(nn.Module):
    def __init__(self,head):
        super().__init__()
        q=head.mode_queries.weight.detach().double().reshape(5,32,32)
        qb=head.mode_queries.bias.detach().double().reshape(5,32)
        key=head.goal_keys.weight.detach().double()[:,:,0,0]
        self.A=nn.Parameter(torch.einsum("mdi,dc->mic",q,key)/(32**.5))
        self.B=nn.Parameter(qb@key/(32**.5)+head.goal_bias.weight.detach().double()[:,:,0,0])
        self.C=nn.Parameter(head.cost_weight.weight.detach().double().reshape(5,2,32).clone())
        self.D=nn.Parameter(head.cost_weight.bias.detach().double().reshape(5,2).clone())
        self.register_buffer("gx",head.gx.detach().double().clone())
        self.register_buffer("gy",head.gy.detach().double().clone())
    def forward(self,b):
        mode=b["mode"][:,0];query=b["query"][:,0].double()
        coeff=torch.einsum("ni,nic->nc",query,self.A[mode])+self.B[mode]
        scores=torch.einsum("nc,ncxy->nxy",coeff,b["spatial"].double()).flatten(1)
        pos=b["positions"][:,0].clamp(0,23)
        distance=(self.gx-pos[:,0,None,None]).abs()+(self.gy-pos[:,1,None,None]).abs()
        costs=torch.stack((distance/46,distance*b["move_cost"][:,0,None,None].double()*5/400),1).flatten(2)
        cw=torch.einsum("ni,nji->nj",query,self.C[mode])+self.D[mode]
        return scores+torch.einsum("nj,njg->ng",cw,costs)
