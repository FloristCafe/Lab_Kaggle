import torch
from torch.nn import functional as F


def objective(outputs,batch,config,weights=None):
    r,ml,gl=outputs
    valid=batch['valid'] & batch['alive']
    if config['model']['replan']:
        valid=valid & batch['replan_valid'] & (batch['replan']>.5)
    zero=ml.sum()*0+gl.sum()*0+r.sum()*0
    mode_loss,goal_loss=zero,zero
    if valid.any():
        mode=batch['mode'][valid]
        confidence=batch['confidence'][valid].float()
        mw=confidence if weights is None else confidence*weights[mode]
        ce=F.cross_entropy(ml[valid].float(),mode,reduction='none')
        mode_loss=(ce*mw).sum()/mw.sum().clamp_min(1e-8)
        logits=gl[valid].float()[torch.arange(len(mode),device=mode.device),mode].flatten(1)
        regions=batch['regions'][valid].flatten(1)
        if not regions.any(1).all():raise ValueError('valid goal without anchor region')
        nll=-torch.logsumexp(logits.log_softmax(1).masked_fill(~regions,-torch.inf),1)
        goal_loss=(nll*confidence).sum()/confidence.sum().clamp_min(1e-8)
    rm=batch['replan_valid'] & batch['alive'] & (batch['units'][:,:,11]>.5)
    r_loss=zero
    if config['model']['replan'] and rm.any():
        r_loss=F.binary_cross_entropy_with_logits(r[rm].float(),batch['replan'][rm].float())
    loss=mode_loss+goal_loss+config['study']['replan_weight']*r_loss
    return loss,{'mode_loss':float(mode_loss.detach()),'goal_loss':float(goal_loss.detach()),
                 'replan_loss':float(r_loss.detach()),'goal_supervised_units':int(valid.sum()),
                 'free_replan_supervised_units':int(rm.sum()) if config['model']['replan'] else 0}
