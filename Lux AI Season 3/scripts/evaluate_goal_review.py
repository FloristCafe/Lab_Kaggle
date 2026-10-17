"""Frozen-weight scripted/corrupted/policy-memory evaluation and short online probes."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.goal_review.data import corrupt_context,paths_for_split,unpack_review
from agents.goal_review.environment import ReviewGoalEnv
from agents.goal_review.labels import replan_targets
from agents.goal_review.memory import ReviewMemory,frame_at
from agents.goal_review.metrics import Metrics
from agents.goal_review.model import ReviewActor,select_review_actions
from agents.strategic.data import history_at
from agents.strategic.experiment import digest,save_json,setup


def replay_packet(d,t,units,shard):
    regions=np.zeros((16,24,24),bool)
    for u in np.flatnonzero(d['semantic_valid'][t]):regions[u]=d['goal_regions'][d['segment_id'][t,u]].astype(bool)
    r,rv,compatible=replan_targets(d['mode'][t],regions,d['semantic_valid'][t],d['boundary_replan'][t],
                                  d['bridge_continue'][t],units,d['label_valid'][t])
    arrays={'history':history_at(d['features'],d['steps'],d['match_steps'],t),'positions':d['positions'][t].astype(np.int64),
        'units':units,'alive':d['active'][t],'mode':d['mode'][t].astype(np.int64),'regions':regions,
        'valid':d['label_valid'][t],'confidence':d['label_confidence'][t].astype(np.float32),
        'replan':r.astype(np.float32),'replan_valid':rv,'sampled':d['active'][t],
        'shard':np.asarray(shard),'step':np.asarray(d['steps'][t]),'match_step':np.asarray(d['match_steps'][t]),
        'segment_id':d['segment_id'][t].astype(np.int64)}
    return {k:torch.from_numpy(np.asarray(v)[None]).cuda() for k,v in arrays.items()}


@torch.no_grad()
def replay_evaluation(actor,config,paths,condition,max_frames=0,corruption=.5):
    metrics=Metrics();frames=reviews=alive_count=covered=prev_valid=0;prev_modes=np.zeros(5,int);ages=[];memory_sources=[]
    for shard,path in enumerate(paths):
        with np.load(path,allow_pickle=False) as z:d=unpack_review(z)
        public=json.loads(str(d['metadata']))['env_cfg'];memory=ReviewMemory(config['review'],public)
        for t in range(min(len(d['steps']),max_frames or len(d['steps']))):
            f=frame_at(d,t);frames+=1;alive_count+=int(f['alive'].sum());covered+=int(d['label_valid'][t].sum())
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
            reviews+=1
            b=replay_packet(d,t,units,shard)
            outputs=actor(b['history'],b['positions'],b['units'],b['alive'])
            metrics.update(outputs,b,config['model'])
            if condition=='policy_replay':
                r,m,g,_,_=select_review_actions(outputs,b['units'],b['alive'],config['model'],greedy=True)
                memory.assign_choices(r[0].cpu().numpy(),m[0].cpu().numpy(),g[0].cpu().numpy(),f['alive'])
        memory_sources.append(dict(memory.stats) if condition=='policy_replay' else {})
    result=metrics.finish()
    result.update(condition=condition,environment_frames=frames,macro_review_epochs=reviews,
        macro_review_frequency=reviews/max(1,frames),alive_unit_states=alive_count,valid_goal_unit_states=covered,
        goal_label_coverage=covered/max(1,alive_count),previous_goal_valid_fraction=prev_valid/max(1,alive_count),
        previous_mode_counts=prev_modes.tolist(),previous_age_mean=float(np.mean(ages)) if ages else 0.,
        memory_sources=memory_sources,
        causal_boundary='policy outputs update future MEMORY only; recorded expert trajectory remains fixed',
        interpretation='policy_replay is replay-conditioned memory feedback, NOT an interactive environment rollout')
    return result


@torch.no_grad()
def online_probe(actor,config,out,steps=64):
    env=ReviewGoalEnv(config,out/'env_worker.log')
    try:
        packet=env.reset(15001,0)  # already-used interface/dev seed, never final holdout
        while not env.done and env.primitive_steps<steps:
            b={k:v.cuda() for k,v in packet.items()}
            outputs=actor(b['history'],b['positions'],b['units'],b['alive'])
            r,m,g,_,_=select_review_actions(outputs,b['units'],b['alive'],config['model'],greedy=True)
            packet,_,_,_=env.step(r[0].cpu().numpy(),m[0].cpu().numpy(),g[0].cpu().numpy())
        result=env.summary();result.update(requested_step_budget=steps,truncated_for_interface_audit=not env.done,seed=15001)
        save_json(out/'online_trace.json',env.review_trace)
        return result
    finally:env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/dataset_v3_goal_review')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--conditions',nargs='+',default=['scripted','corrupted','policy_replay'],choices=['scripted','corrupted','policy_replay'])
    p.add_argument('--max-games',type=int,default=0);p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--closed-loop-steps',type=int,default=64);args=p.parse_args()
    setup();checkpoint=torch.load(args.checkpoint,weights_only=False,map_location='cpu');config=checkpoint['config']
    actor=ReviewActor(config['model']).cuda().eval();actor.load_state_dict(checkpoint['actor_state_dict'])
    paths=paths_for_split(args.data_dir,'validation')
    if args.max_games:paths=paths[:args.max_games]
    if not paths:raise ValueError('no validation games')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if (args.output_dir/'summary.json').exists():raise ValueError('existing evaluation summary; use fresh output')
    from ml_workspace.tracking import tracked_run
    import mlflow
    with tracked_run(args.output_dir.name,experiment_name='lux-s3-goal-review',params={'operation':'frozen_evaluation_no_training','seed':42}) as run:
        result={c:replay_evaluation(actor,config,paths,c,args.max_frames) for c in args.conditions}
        if args.closed_loop_steps:result['closed_loop']=online_probe(actor,config,args.output_dir,args.closed_loop_steps)
        result.update(checkpoint_sha256=digest(args.checkpoint),mlflow_run_id=run.info.run_id,
                      smoke_only=checkpoint.get('smoke_only',False),training_updates=0)
        save_json(args.output_dir/'summary.json',result);mlflow.log_artifacts(str(args.output_dir))
        print(json.dumps({'output':str(args.output_dir),'training_updates':0}),flush=True)


if __name__=='__main__':main()
