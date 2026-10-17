"""Independent E0/E1/E2 pilot entry. Never automatically starts another arm."""
import argparse
import copy
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.goal_review.config import load_config
from agents.goal_review.model import ReviewActor
from agents.mappo_v5.environment import TeamEnv
from agents.mappo_v5.learning import TeamCritic,Transition,device,update,save_rollout
from agents.mappo_v5.policy import neutralize_r,optimizer,sample
from agents.strategic.experiment import digest,provenance,save_json,setup


def state_hash(state):
    h=hashlib.sha256()
    for key,x in sorted(state.items()):h.update(key.encode());h.update(x.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def initialize(config,seed,where='cuda'):
    path=ROOT/config['il_checkpoint']
    if digest(path)!=config['il_checkpoint_sha256']:raise ValueError('IL checkpoint hash changed')
    checkpoint=torch.load(path,weights_only=False,map_location='cpu')
    if checkpoint.get('smoke_only') or checkpoint['config']['model']!=config['model']:
        raise ValueError('wrong IL model/config or smoke checkpoint')
    actor=ReviewActor(config['model']);actor.load_state_dict(checkpoint['actor_state_dict'],strict=True)
    original=state_hash(actor.state_dict())
    if config['policy']['r_initialization']=='neutral':neutralize_r(actor)
    elif config['policy']['r_initialization']!='original':raise ValueError('unknown R initialization')
    changed=[k for k,x in actor.state_dict().items() if not torch.equal(x,checkpoint['actor_state_dict'][k])]
    if any(not k.startswith('replan_head.') for k in changed):raise AssertionError('IL weights changed outside R')
    torch.manual_seed(seed+1000);critic=TeamCritic()
    torch.manual_seed(seed)
    actor=actor.to(where);critic=critic.to(where)
    info={'il_checkpoint_sha256':digest(path),'original_actor_hash':original,'initial_actor_hash':state_hash(actor.state_dict()),
          'initial_critic_hash':state_hash(critic.state_dict()),'modified_IL_tensors':changed,
          'R_initialization':config['policy']['r_initialization'],'critic_initialization':'independent seed+1000'}
    return actor,critic,info


def sources():
    paths=[*ROOT.joinpath('agents/mappo_v5').glob('*.py'),*ROOT.joinpath('agents/goal_review').glob('*.py'),
        *ROOT.joinpath('agents/strategic').glob('*.py'),*ROOT.joinpath('agents/v2').glob('*.py'),
        ROOT/'scripts/train_mappo_v5.py',ROOT/'scripts/mappo_v5_worker.py',ROOT/'scripts/replay_parser.py',
        ROOT/'agents/v3/dataset.py']
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def rng_state():
    return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'numpy':np.random.get_state(),'python':random.getstate()}


def restore_rng(rng):
    torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda'])
    np.random.set_state(rng['numpy']);random.setstate(rng['python'])


def save_checkpoint(path,actor,critic,opt,config,progress,manifest,smoke=False):
    tmp=path.with_suffix('.tmp')
    torch.save({'actor_state_dict':actor.state_dict(),'critic_state_dict':critic.state_dict(),
        'optimizer_state_dict':opt.state_dict(),'config':config,'progress':progress,'rng':rng_state(),
        'IL_sha256':manifest['initialization']['il_checkpoint_sha256'],'source_sha256':manifest['source_sha256'],
        'resume_boundary':'complete game only, no environment state serialization','smoke_only':smoke},tmp)
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--resume',action='store_true')
    p.add_argument('--run-pilot',action='store_true',help='explicit launch, no implicit long training')
    args=p.parse_args()
    if not args.run_pilot:p.error('pilot requires --run-pilot; no training was started')
    config=load_config(args.config);setup(args.seed);random.seed(args.seed)
    vendor=ROOT/'vendor/lux-design-s3'
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=vendor,text=True).strip()!='c6d1e665d12a467b299fb586b3f876e47847746b':
        raise ValueError('official environment pin changed')
    if subprocess.check_output(['git','status','--porcelain'],cwd=vendor,text=True).strip():
        raise ValueError('official environment has local modifications')
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if not 0<=config['games']<=100:raise ValueError('this implementation handoff permits pilot <=100 games only')
    if config['policy']['r_initialization']=='original' and config['ppo']['R_lr']>3e-5:
        raise ValueError('high R LR requires explicitly neutral initialization')
    if config['model']['hysteresis']!=0:raise ValueError('pilot isolates RL, no new hysteresis intervention')
    out=args.output_dir;exists=out.exists() and any(out.iterdir())
    if exists and not args.resume:raise ValueError('preserve previous run; use --resume or a fresh output directory')
    if args.resume and not (out/'last.pth').exists():raise ValueError('resume requires a complete-game checkpoint')
    if (out/'summary.json').exists():raise ValueError('completed immutable run; choose a new directory')
    out.mkdir(parents=True,exist_ok=True)
    actor,critic,initial=initialize(config,args.seed);opt=optimizer(actor,critic,config)
    actual_sources=sources()
    manifest={'config':config,'seed':args.seed,'provenance':provenance(),'source_sha256':actual_sources,
        'initialization':initial,'reward':'terminal difference of public official cumulative team wins; all intermediate rewards zero',
        'precision':'FP32 likelihoods/update, TF32 off; small batches avoid behavior ratio precision drift',
        'feature_boundary':'legal observations <=T; actual previous intents <T, no expert labels in RL',
        'evaluation_boundary':'paired previously designated dev seeds, both sides; no final holdout',
        'resume_boundary':'complete-game checkpoints only; midgame failure restarts from previous saved game',
        'opponent_hashes':{name:actual_sources.get(f'agents/{name}/agent.py') or digest(ROOT/f'agents/{name}/agent.py') for name in config['opponents']}}
    progress={'next_game':0,'training':[],'updates':[],'evaluations':[],'primitive_steps':0,'resume_attempt':0,'rollout_bytes':0}
    if args.resume:
        original_manifest=json.loads((out/'manifest.json').read_text())
        if original_manifest['seed']!=args.seed:raise ValueError('cannot resume with a different training seed')
        checkpoint=torch.load(out/'last.pth',weights_only=False,map_location='cpu')
        if checkpoint.get('smoke_only'):raise ValueError('cannot resume pilot from software smoke')
        if checkpoint['config']!=config or checkpoint['source_sha256']!=actual_sources or checkpoint['IL_sha256']!=initial['il_checkpoint_sha256']:
            raise ValueError('resume config, code or IL hash mismatch')
        actor.load_state_dict(checkpoint['actor_state_dict']);critic.load_state_dict(checkpoint['critic_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict']);progress=checkpoint['progress'];restore_rng(checkpoint['rng'])
        progress['resume_attempt']+=1
        manifest=original_manifest
    else:
        for name in actual_sources:
            target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        save_json(out/'manifest.json',manifest);(out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        torch.save({'actor_state_dict':actor.state_dict(),'critic_state_dict':critic.state_dict(),'initialization':initial},out/'initial.pth')
    # Fixed, auditable map seeds. Cross-check against IL source seeds as well.
    data=json.loads((ROOT/'data/dataset_v3_goal_review/manifest.json').read_text())
    il_seeds={v['source_seed'] for v in data['files'].values() if v['source_seed'] is not None}
    train_seeds=[config['train_seed_base']+args.seed*1000+i for i in range(config['games'])]
    eval_seeds=config['evaluation']['seeds']
    if set(train_seeds)&(set(eval_seeds)|il_seeds) or set(eval_seeds)&il_seeds:
        raise ValueError('RL train/dev and IL map seed overlap')
    save_json(out/'seed_plan.json',{'train':train_seeds,'dev':eval_seeds,'sides':[0,1],'opponents':config['opponents'],
        'dev_note':'not final test; same fixed seeds across all arms; some dev seeds used by prior studies'})
    # Include preserved orphan journals from interrupted games in the disk cap.
    progress['rollout_bytes']=sum(p.stat().st_size for p in (out/'rollouts').glob('**/*.npz'))
    from ml_workspace.resource_guard import cuda_memory_guard,cuda_memory_stats
    from ml_workspace.tracking import tracked_run
    import mlflow
    started=time.perf_counter()
    with tracked_run(out.name+f"_attempt{progress['resume_attempt']}",experiment_name='lux-s3-mappo-v5',
        params={'arm':config['arm'],'seed':args.seed,'training_games':config['games'],
                'R_initialization':config['policy']['r_initialization'],'gate':config['policy']['gate']}) as run:
        save_json(out/f"attempt_{progress['resume_attempt']}.json",{'mlflow_run_id':run.info.run_id,'resumed':args.resume})
        env=TeamEnv(config,out/'worker.log')
        try:
            with cuda_memory_guard('mappo-v5-pilot'):
                for episode in range(progress['next_game'],config['games']):
                    opponent=config['opponents'][episode%len(config['opponents'])]
                    packet=env.reset(train_seeds[episode],episode%2,opponent);buffer=[];last_rollout=None
                    while not env.terminal:
                        actor.eval();critic.eval();b=device(packet)
                        with torch.no_grad():
                            action,_=sample(actor,b,config);value=float(critic(b)[0])
                        next_packet,reward,terminal,truncated,info=env.step(action)
                        with torch.no_grad():next_value=0. if terminal else float(critic(device(next_packet))[0])
                        buffer.append(Transition(packet,next_packet,action,reward,info['tau'],value,next_value,terminal,truncated,episode,
                            info['engine_terminal'],info['engine_truncated'],info['task_complete'],tuple(info['primitive_rewards'])))
                        packet=next_packet;progress['primitive_steps']+=info['tau']
                        if len(buffer)>=config['ppo']['rollout_options'] or terminal or truncated:
                            if shutil.disk_usage(out).free<config['storage']['min_free_mib']*2**20:
                                raise RuntimeError('disk reserve reached; completed-game checkpoint retained')
                            if progress['rollout_bytes']>=config['storage']['max_rollout_mib']*2**20:
                                raise RuntimeError('rollout artifact budget reached; do not fill the research disk')
                            journal=out/f"rollouts/attempt{progress['resume_attempt']}/update_{len(progress['updates'])+1:06d}.npz"
                            progress['rollout_bytes']+=save_rollout(journal,buffer,config['ppo']['gamma'])
                            # Persist a small real behavior sample, not every game-sized raw observation.
                            if last_rollout is None:last_rollout=copy.deepcopy(buffer[:2])
                            metrics=update(actor,critic,opt,buffer,config);progress['updates'].append(metrics)
                            step=len(progress['updates']);mlflow.log_metrics({k:v for k,v in metrics.items() if isinstance(v,(int,float))},step=step)
                            print(json.dumps({'episode':episode,'update':step,**metrics}),flush=True)
                            save_json(out/'update_metrics.json',progress['updates']);buffer=[]
                        if truncated:raise RuntimeError('unexpected external truncation in complete-game training')
                    result=env.summary();result['episode']=episode;progress['training'].append(result);progress['next_game']=episode+1
                    save_json(out/'training_metrics.json',progress['training'])
                    if episode in (0,config['games']-1):torch.save(last_rollout,out/f"rollout_sample_game{episode}_attempt{progress['resume_attempt']}.pth")
                    save_checkpoint(out/'last.pth',actor,critic,opt,config,progress,manifest)
                    print(json.dumps({'training_game_complete':episode+1,'total':config['games'],'result':result}),flush=True)
                done={(r['seed'],r['side'],r['opponent']) for r in progress['evaluations']}
                for seed in eval_seeds:
                    for opponent in config['opponents']:
                        for side in (0,1):
                            if (seed,side,opponent) in done:continue
                            packet=env.reset(seed,side,opponent);actor.eval();trace=[]
                            while not env.terminal:
                                with torch.no_grad():action,_=sample(actor,device(packet),config,greedy=True)
                                trace.append({'step':int(packet['primitive_step'][0]),
                                    'previous_intent':packet['previous_intent'][0].tolist(),
                                    **{k:action[k][0].cpu().tolist() for k in ('r','mode','goal','decision_mask','forced_action_mask','branch_mask','order')},
                                    'model_log_density':action['old_logprob'][0].cpu().tolist(),
                                    'probability_note':'deterministic greedy evaluation; density under latent stochastic policy, NEVER PPO old probability'})
                                packet,_,_,truncated,_=env.step(action)
                                if truncated:raise RuntimeError('dev evaluation unexpectedly truncated')
                            result=env.summary();progress['evaluations'].append(result)
                            save_json(out/f'evaluation_trace_{seed}_p{side}_{opponent}.json',trace)
                            save_json(out/'evaluation_metrics.json',progress['evaluations'])
                            save_checkpoint(out/'last.pth',actor,critic,opt,config,progress,manifest)
                            print(json.dumps({'evaluation':result}),flush=True)
        finally:env.close()
        summary={'status':'FINISHED','arm':config['arm'],'seed':args.seed,'training_games':len(progress['training']),
            'training':progress['training'],'evaluations':progress['evaluations'],
            'optimizer_steps':sum(r['optimizer_steps'] for r in progress['updates']),
            'primitive_training_steps':progress['primitive_steps'],'initialization':initial,
            'final_actor_hash':state_hash(actor.state_dict()),'final_critic_hash':state_hash(critic.state_dict()),
            'win_rate':float(np.mean([r['win'] for r in progress['evaluations']])),
            'evaluation_mean_margin':float(np.mean([r['game_margin'] for r in progress['evaluations']])),
            'wall_seconds_current_attempt':time.perf_counter()-started,'cuda_memory':asdict(cuda_memory_stats()),
            'mlflow_run_id':run.info.run_id,'pilot_verdict':'PENDING_RESEARCH_REVIEW; no automatic 1000-game launch',
            'config':config,'source_sha256':actual_sources,'checkpoint_sha256':digest(out/'last.pth')}
        save_json(out/'summary.json',summary)
        # Store compact lineage/metrics in MLflow. Weights/source remain ext4 at manifest path.
        for name in ('summary.json','manifest.json','config.yaml','seed_plan.json'):
            mlflow.log_artifact(str(out/name))
        print(json.dumps({'status':'FINISHED','training_games':summary['training_games'],'win_rate':summary['win_rate']}),flush=True)


if __name__=='__main__':main()
