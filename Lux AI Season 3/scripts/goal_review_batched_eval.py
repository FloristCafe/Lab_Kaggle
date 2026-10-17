"""Batch independent replay streams; each game's policy memory remains causal."""
import hashlib
import json

import numpy as np
import torch

from agents.goal_review.data import corrupt_context,unpack_review
from agents.goal_review.memory import frame_at
from agents.goal_review.runtime import RuntimeReviewMemory
from agents.goal_review.metrics import Metrics
from agents.goal_review.model import select_review_actions
from scripts.evaluate_goal_review import replay_packet


@torch.no_grad()
def batched_replay_evaluation(actor,config,paths,condition,batch_games=16,max_frames=0,corruption=.5):
    if batch_games<1:raise ValueError('positive number of independent streams required')
    metrics=Metrics();frames=reviews=alive_count=covered=prev_valid=0
    prev_modes=np.zeros(5,int);ages=[];memory_sources=[]
    held_totals=np.zeros((5,6),np.float64)
    for offset in range(0,len(paths),batch_games):
        chunk=paths[offset:offset+batch_games];streams=[]
        for i,path in enumerate(chunk):
            with np.load(path,allow_pickle=False) as z:d=unpack_review(z)
            public=json.loads(str(d['metadata']))['env_cfg']
            held={'mode':np.full(16,3,np.int64),'goal':np.zeros((16,2),np.int64),'valid':np.zeros(16,bool)}
            streams.append((offset+i,path,d,RuntimeReviewMemory(config['review'],public),held))
        limit=max(min(len(d['steps']),max_frames or len(d['steps'])) for _,_,d,_,_ in streams)
        for t in range(limit):
            packets=[];opened_streams=[];current_frames=[]
            for shard,path,d,memory,held in streams:
                if t>=min(len(d['steps']),max_frames or len(d['steps'])):continue
                f=frame_at(d,t);frames+=1;alive_count+=int(f['alive'].sum());covered+=int(d['label_valid'][t].sum())
                held['valid']&=f['alive'];current_frames.append((d,f,held))
                if condition=='policy_replay':
                    opened=memory.observe(f);units=memory.vectors(f)
                else:
                    opened=bool(d['review_frames'][t]);units=d['unit_vectors'][t].astype(np.float32)
                if condition=='corrupted':
                    seed=int.from_bytes(hashlib.sha256(f'42:{path.name}:{t}'.encode()).digest()[:8],'little')
                    units=corrupt_context(units,seed,corruption)
                previous=(units[:,11]>.5)&f['alive'];prev_valid+=int(previous.sum())
                prev_modes+=np.bincount(units[previous,12:17].argmax(1),minlength=5)
                ages.extend((units[previous,10]*100).tolist())
                if not opened:continue
                reviews+=1;packets.append(replay_packet(d,t,units,shard));opened_streams.append((memory,f,held))
            if packets:
                batch={k:torch.cat([packet[k] for packet in packets]) for k in packets[0]}
                outputs=actor(batch['history'],batch['positions'],batch['units'],batch['alive'])
                metrics.update(outputs,batch,config['model'])
                r,m,g,_,_=select_review_actions(outputs,batch['units'],batch['alive'],config['model'],greedy=True)
                rr,mm,gg=r.cpu().numpy(),m.cpu().numpy(),g.cpu().numpy()
                for i,(memory,f,held) in enumerate(opened_streams):
                    if condition=='policy_replay':memory.assign_choices(rr[i],mm[i],gg[i],f['alive'])
                    active=f['alive'];held['mode'][active]=mm[i,active]
                    held['goal'][active,0]=gg[i,active]//24;held['goal'][active,1]=gg[i,active]%24
                    held['valid'][active]=True
            # Score the emitted/held intent at EVERY labelled physical frame,
            # not only policy-dependent review times. Targets are scoring-only.
            for d,f,held in current_frames:
                valid=d['label_valid'][t]&f['alive'];u=np.flatnonzero(valid)
                if not len(u):continue
                y=d['mode'][t,u];sid=d['segment_id'][t,u];xy=held['goal'][u]
                hit=d['goal_regions'][sid,xy[:,0],xy[:,1]].astype(bool)&held['valid'][u]
                mode_hit=(held['mode'][u]==y)&held['valid'][u]
                at_anchor=(xy==f['positions'][u]).all(1)&held['valid'][u]
                values=np.column_stack((np.ones(len(u)),hit&mode_hit,mode_hit,hit,at_anchor,held['valid'][u]))
                np.add.at(held_totals,y,values)
        memory_sources.extend(dict(memory.stats) if condition=='policy_replay' else {} for _,_,_,memory,_ in streams)
    result=metrics.finish()
    total=held_totals.sum(0);denom=max(1,total[0])
    result['all_frame_held_intent']={'units':int(total[0]),'joint':total[1]/denom,
        'mode_accuracy':total[2]/denom,'goal_hit_without_mode':total[3]/denom,
        'anchor_prediction_rate':total[4]/denom,'assigned_intent_fraction':total[5]/denom,
        'per_mode_counts':held_totals[:,0].astype(int).tolist(),
        'per_mode_joint':(held_totals[:,1]/np.maximum(held_totals[:,0],1)).tolist(),
        'scope':'emitted intent persists between reviews; all labelled replay frames, fixed denominator across conditions; expert physical state path is fixed'}
    result.update(condition=condition,environment_frames=frames,macro_review_epochs=reviews,
        macro_review_frequency=reviews/max(1,frames),alive_unit_states=alive_count,valid_goal_unit_states=covered,
        goal_label_coverage=covered/max(1,alive_count),previous_goal_valid_fraction=prev_valid/max(1,alive_count),
        previous_mode_counts=prev_modes.tolist(),previous_age_mean=float(np.mean(ages)) if ages else 0.,
        memory_sources=memory_sources,batch_games=batch_games,
        causal_boundary='each stream advances T in order and updates only its own memory after acting at T; different games batch independently',
        interpretation='policy_replay is replay-conditioned memory feedback, NOT an interactive environment rollout')
    return result
