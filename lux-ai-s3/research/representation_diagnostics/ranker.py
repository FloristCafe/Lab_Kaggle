"""Fixed 5345-parameter D08 scorer and node-equal pair loss; no optimizer."""
import torch
from torch import nn

class Ranker(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(165,32),nn.SiLU(),nn.Linear(32,1))
 def forward(self,x):return self.net(x).squeeze(-1)

def node_loss(scores,sign,valid):
 pair=scores.reshape(len(scores),32,2)
 losses=torch.nn.functional.softplus(-sign*(pair[:,:,0]-pair[:,:,1]))
 return (losses*valid).sum(1)/valid.sum(1).clamp_min(1)
