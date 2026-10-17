import numpy as np
import torch

from agents.strategic.features import MODES
from .model import select_review_actions


def classification(confusion):
    c=np.asarray(confusion)
    precision=c.diagonal()/np.maximum(c.sum(0),1)
    recall=c.diagonal()/np.maximum(c.sum(1),1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    return {'accuracy':float(c.trace()/max(c.sum(),1)),'macro_f1':float(f1.mean()),
            'balanced_accuracy':float(recall.mean()),'precision':precision.tolist(),
            'recall':recall.tolist(),'f1':f1.tolist(),'support':c.sum(1).tolist(),
            'predicted_count':c.sum(0).tolist(),'confusion_matrix':c.tolist()}


class Metrics:
    def __init__(self):
        self.head=np.zeros((5,5),int);self.executed=np.zeros((5,5),int)
        self.r=np.zeros((2,2),int);self.free_r=np.zeros((2,2),int)
        self.goal_n=0;self.totals=np.zeros(8,float)
        self.by_mode=np.zeros((5,5),float)
        self.last={};self.pairs=self.persist=self.switches=self.mode_switches=0
        self.predicted_distance=self.expert_region_distance=0.
        self.segment_pairs=self.segment_persist=0

    @torch.no_grad()
    def update(self,outputs,b,cfg):
        rr,mm,gg,_,_=select_review_actions(outputs,b['units'],b['alive'],cfg,greedy=True)
        raw=outputs[1].argmax(-1)
        rv=b['replan_valid']&b['sampled']&b['alive']
        if rv.any():
            y=b['replan'][rv].long().cpu().numpy();p=rr[rv].long().cpu().numpy()
            np.add.at(self.r,(y,p),1)
            free=(b['units'][:,:,11]>.5)[rv].cpu().numpy()
            np.add.at(self.free_r,(y[free],p[free]),1)
        v=b['valid']&b['sampled']&b['alive']
        if not v.any():return
        y=b['mode'][v]
        region=b['regions'][v].flatten(1)
        goal_logits=outputs[2][v].float()[torch.arange(len(y),device=y.device),y].flatten(1)
        oracle=goal_logits.argmax(1)
        ranked=goal_logits.topk(10,1).indices
        hit=region.gather(1,oracle[:,None]).squeeze(1)
        executed_hit=region.gather(1,gg[v][:,None]).squeeze(1)
        logp=goal_logits.log_softmax(1);h=-(logp.exp()*logp).sum(1)
        grid=torch.arange(576,device=y.device)
        distance=((grid[None]//24-oracle[:,None]//24).abs()+(grid[None]%24-oracle[:,None]%24).abs()).float()
        distance=distance.masked_fill(~region,torch.inf).min(1).values
        anchor=b['positions'][v][:,0]*24+b['positions'][v][:,1]
        executed_distance=(gg[v]//24-anchor//24).abs()+(gg[v]%24-anchor%24).abs()
        from_anchor=((grid[None]//24-anchor[:,None]//24).abs()+(grid[None]%24-anchor[:,None]%24).abs()).float()
        self.predicted_distance+=float(executed_distance.sum())
        self.expert_region_distance+=float(from_anchor.masked_fill(~region,torch.inf).min(1).values.sum())
        numbers=torch.stack((hit.float(),region.gather(1,ranked[:,:5]).any(1).float(),
            region.gather(1,ranked).any(1).float(),distance,h,logp.exp().max(1).values,
            (executed_hit&(mm[v]==y)).float(),(oracle==anchor).float()),1)
        self.totals+=numbers.sum(0).cpu().numpy();self.goal_n+=len(y)
        yc=y.cpu().numpy()
        np.add.at(self.head,(yc,raw[v].cpu().numpy()),1)
        np.add.at(self.executed,(yc,mm[v].cpu().numpy()),1)
        for mode in range(5):
            mask=y==mode
            self.by_mode[mode]+=np.array([int(mask.sum()),float(hit[mask].sum()),float(distance[mask].sum()),
                                         float(h[mask].sum()),float(((mm[v]==y)&executed_hit)[mask].sum())])
        for row,uid in v.nonzero().cpu().tolist():
            key=(int(b['shard'][row]),uid)
            step,ms=int(b['step'][row]),int(b['match_step'][row])
            segment=int(b['segment_id'][row,uid])
            previous=self.last.get(key)
            if previous is not None and 0<step-previous[0]<=32 and ms>previous[1]:
                self.pairs+=1
                self.persist+=int(mm[row,uid])==previous[2] and int(gg[row,uid])==previous[3]
                self.switches+=int(gg[row,uid])!=previous[3]
                self.mode_switches+=int(mm[row,uid])!=previous[2]
                if segment>=0 and segment==previous[4]:
                    self.segment_pairs+=1
                    self.segment_persist+=int(mm[row,uid])==previous[2] and int(gg[row,uid])==previous[3]
            self.last[key]=(step,ms,int(mm[row,uid]),int(gg[row,uid]),segment)

    def finish(self):
        head,executed=classification(self.head),classification(self.executed)
        values=self.totals/max(1,self.goal_n)
        result={'mode':head,'executed_mode':executed,'valid_goal_units':self.goal_n,
                **dict(zip(('goal_top1','goal_top5','goal_top10','goal_manhattan','goal_entropy',
                            'goal_peak','joint_mode_goal_accuracy','anchor_rate'),map(float,values))),
                'goal_geodesic_error':None,'geodesic_note':'not identified for drifting/unknown passability',
                'replan':classification(self.r),'free_replan':classification(self.free_r),
                'false_replan_rate':float(self.r[0,1]/max(1,self.r[0].sum())),
                'goal_switch_rate':self.switches/max(1,self.pairs),'mode_switch_rate':self.mode_switches/max(1,self.pairs),
                'goal_persistence':self.persist/max(1,self.pairs),'persistence_pairs':self.pairs,
                'offline_persistence_scope':'adjacent sampled replay states, causal scripted/corrupted context; not actual rollout',
                'goal_completion_rate':None,'invalid_goal_persistence':None,'per_mode':{}}
        result.update(predicted_goal_distance=self.predicted_distance/max(1,self.goal_n),
                      expert_region_distance=self.expert_region_distance/max(1,self.goal_n),
                      within_segment_prediction_consistency=self.segment_persist/max(1,self.segment_pairs),
                      within_segment_pairs=self.segment_pairs,
                      goal_topk_scope='expert-mode conditional raw goal head; joint accuracy uses resolved executed intent')
        for i,name in enumerate(MODES):
            n,hit,dist,h,joint=self.by_mode[i]
            result['per_mode'][name]={'precision':head['precision'][i],'recall':head['recall'][i],
                'f1':head['f1'][i],'label_count':int(n),'predicted_count':head['predicted_count'][i],
                'executed_recall':executed['recall'][i],'executed_predicted_count':executed['predicted_count'][i],
                'goal_top1':hit/max(n,1),'goal_error':dist/max(n,1),'goal_entropy':h/max(n,1),
                'joint_mode_goal_accuracy':joint/max(n,1)}
        return result


@torch.no_grad()
def evaluate(model,loader,config,device='cuda',max_units=0):
    model.eval();metrics=Metrics()
    for cpu in loader:
        b={k:v.to(device) for k,v in cpu.items()}
        with torch.autocast(device_type=device,enabled=device=='cuda',dtype=torch.bfloat16):
            outputs=model(b['history'],b['positions'],b['units'],b['alive'])
        metrics.update(outputs,b,config['model'])
        if max_units and metrics.goal_n>=max_units:break
    return metrics.finish()
