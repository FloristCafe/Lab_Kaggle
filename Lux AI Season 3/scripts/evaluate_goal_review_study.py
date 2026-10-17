"""Frozen-weight memory/hysteresis study, full validation replays and dev prefixes."""
import argparse
import copy
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.goal_review.data import paths_for_split
from agents.goal_review.runtime import RuntimeGoalEnv
from agents.goal_review.model import ReviewActor,select_review_actions
from agents.strategic.experiment import digest,provenance,save_json,setup
from scripts.goal_review_batched_eval import batched_replay_evaluation


@torch.no_grad()
def dev_prefix(actor,config,out,side,steps):
    env=RuntimeGoalEnv(config,out/f'env_worker_p{side}.log')
    try:
        packet=env.reset(15001,side)
        while not env.done and env.primitive_steps<steps:
            b={k:v.cuda() for k,v in packet.items()}
            outputs=actor(b['history'],b['positions'],b['units'],b['alive'])
            r,m,g,_,_=select_review_actions(outputs,b['units'],b['alive'],config['model'],greedy=True)
            packet,_,_,_=env.step(r[0].cpu().numpy(),m[0].cpu().numpy(),g[0].cpu().numpy())
        result=env.summary()
        result.update(seed=15001,side=side,requested_step_budget=steps,truncated_for_study=not env.done,
                      protocol='short dev prefix; option finishes before artificial truncation; no full-game win claim')
        save_json(out/f'online_trace_p{side}.json',env.review_trace)
        return result
    finally:env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/dataset_v3_goal_review')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--hysteresis-values',nargs='+',type=float,default=[0.,.25])
    p.add_argument('--conditions',nargs='+',default=['scripted','corrupted','policy_replay'],choices=['scripted','corrupted','policy_replay'])
    p.add_argument('--max-games',type=int,default=0)
    p.add_argument('--closed-loop-steps',type=int,default=128)
    p.add_argument('--sides',nargs='+',type=int,default=[0,1],choices=[0,1])
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--batch-games',type=int,default=16)
    args=p.parse_args();setup();started=time.perf_counter()
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    if args.smoke_only:
        args.max_games=1;args.closed_loop_steps=16;args.sides=[0]
    checkpoint=torch.load(args.checkpoint,weights_only=False,map_location='cpu');original=checkpoint['config']
    if checkpoint.get('smoke_only') and not args.smoke_only:raise ValueError('formal memory study refuses smoke weights')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if (args.output_dir/'summary.json').exists():raise ValueError('immutable finished evaluation; choose a new directory')
    if len(set(args.hysteresis_values))!=len(args.hysteresis_values):raise ValueError('duplicate hysteresis conditions')
    actor=ReviewActor(original['model']).cuda().eval();actor.load_state_dict(checkpoint['actor_state_dict'])
    paths=paths_for_split(args.data_dir,'validation')
    if args.max_games:paths=paths[:args.max_games]
    if not paths:raise ValueError('empty validation')
    manifest=json.loads((args.data_dir/'manifest.json').read_text())
    assert checkpoint['dataset_manifest_sha256']==digest(args.data_dir/'manifest.json')
    for path in paths:
        if digest(path)!=manifest['files'][path.name]['npz_sha256']:raise ValueError('validation artifact changed')
    from ml_workspace.resource_guard import cuda_memory_stats
    from ml_workspace.tracking import tracked_run
    import mlflow
    result={}
    with tracked_run(args.output_dir.name,experiment_name='lux-s3-goal-review',params={
        'operation':'frozen_memory_study','seed':42,'checkpoint':args.checkpoint.parent.name,
        'validation_games':len(paths),'training_updates':0,'precision':'FP32','closed_loop_steps':args.closed_loop_steps}) as run:
        plan={'provenance':provenance(),'checkpoint_sha256':digest(args.checkpoint),
              'checkpoint_training_config':original,'hysteresis_values':args.hysteresis_values,
              'validation_files':[path.name for path in paths],'dataset_manifest_sha256':digest(args.data_dir/'manifest.json'),
              'conditions':args.conditions,'dev_seed':15001,'sides':args.sides,'prefix_steps':args.closed_loop_steps,
              'training_updates':0,'smoke_only':args.smoke_only,'comparison':'within-checkpoint inference intervention, no model retraining',
              'replay_boundary':'policy_replay updates memory only, expert state trajectories remain fixed',
              'actual_closed_loop':'public-observation environment on previously used dev seed, short prefixes',
              'batch_games':args.batch_games,'tf32':False,
              'source_sha256':digest(Path(__file__)),
              'evaluation_sources':{name:digest(ROOT/'scripts'/name) for name in (
                  'goal_review_batched_eval.py','evaluate_goal_review.py')},
              'runtime_memory_sha256':digest(ROOT/'agents/goal_review/runtime.py'),
              'goal_age_definition':'identity persistence age, preserved for same-pair Replan; dataset scripted contexts unchanged',
              'mlflow_run_id':run.info.run_id}
        existing=args.output_dir/'plan.json'
        if existing.exists():
            old=json.loads(existing.read_text())
            for key in ('checkpoint_sha256','hysteresis_values','validation_files','dataset_manifest_sha256',
                        'conditions','dev_seed','sides','prefix_steps','source_sha256','evaluation_sources','batch_games','tf32','runtime_memory_sha256'):
                if old[key]!=plan[key]:raise ValueError(f'partial evaluation plan changed: {key}')
        save_json(existing,plan)
        for eta in args.hysteresis_values:
            if not 0<=eta<=original['model'].get('hysteresis_cap',.5):raise ValueError('outside preregistered bounded hysteresis')
            config=copy.deepcopy(original);config['model']['hysteresis']=eta
            folder=args.output_dir/f'eta_{eta:g}';folder.mkdir(exist_ok=True)
            values={}
            for condition in args.conditions:
                path=folder/f'{condition}.json'
                if path.exists():
                    values[condition]=json.loads(path.read_text())
                else:
                    values[condition]=batched_replay_evaluation(actor,config,paths,condition,batch_games=args.batch_games)
                    save_json(path,values[condition])
                print(json.dumps({'eta':eta,'condition':condition,'complete':True,
                      'games':len(paths),'macro_f1':values[condition]['mode']['macro_f1'],
                      'joint':values[condition]['joint_mode_goal_accuracy'],'elapsed_seconds':time.perf_counter()-started}),flush=True)
            values['closed_loop']=[]
            if args.closed_loop_steps:
                for side in args.sides:
                    path=folder/f'online_summary_p{side}.json'
                    prefix=json.loads(path.read_text()) if path.exists() else dev_prefix(actor,config,folder,side,args.closed_loop_steps)
                    save_json(path,prefix);values['closed_loop'].append(prefix)
            result[str(eta)]=values
            save_json(args.output_dir/'partial_summary.json',result)
        summary={'conditions':result,'plan':plan,'inference_precision':'FP32','training_updates':0,
                 'wall_seconds':time.perf_counter()-started,'cuda_memory':asdict(cuda_memory_stats()),
                 'status':'FINISHED','smoke_only':args.smoke_only,'mlflow_run_id':run.info.run_id}
        save_json(args.output_dir/'summary.json',summary);mlflow.log_artifacts(str(args.output_dir))
        print(json.dumps({'output':str(args.output_dir),'status':'FINISHED','wall_seconds':summary['wall_seconds']}),flush=True)


if __name__=='__main__':main()
