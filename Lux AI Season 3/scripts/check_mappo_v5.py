"""Bounded software checks (<115s), no complete games or pilot training."""
import argparse
import copy
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.goal_review.config import load_config
from agents.goal_review.data import unpack_review
from agents.mappo_v5.environment import TeamEnv
from agents.mappo_v5.learning import Transition,device,update,save_rollout,load_rollout
from agents.mappo_v5.policy import neutralize_r,optimizer,policy_outputs,sample
from agents.strategic.experiment import digest,provenance,save_json,setup
from scripts.train_mappo_v5 import initialize,state_hash,save_checkpoint,restore_rng,sources


def probe_r(actor,packet):
    with torch.no_grad():logits=actor(packet['history'],packet['positions'],packet['units'],packet['alive'])[0].float()
    active=packet['alive'];free=active&(packet['units'][...,11]>.5);p=logits.sigmoid()
    def describe(mask):
        if not mask.any():return {'count':0}
        x=logits[mask];y=p[mask]
        return {'count':int(mask.sum()),'logit_quantiles':torch.quantile(x,torch.tensor([0.,.1,.5,.9,1.],device=x.device)).tolist(),
            'probability_quantiles':torch.quantile(y,torch.tensor([0.,.1,.5,.9,1.],device=x.device)).tolist(),
            'mean_probability':float(y.mean()),'greedy_replan_frequency':float((y>=.5).float().mean())}
    pm=packet['units'][...,12:17].argmax(-1)
    return {'all_active':describe(active),'free':describe(free),'forced':describe(active&~free),
        'previous_mode':{str(m):describe(free&(pm==m)) for m in range(5)},
        'sampling_note':'sigmoid probabilities audited; full-free stochastic frequency measured in actual short trajectory'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--reference-snapshot',type=Path,required=True);args=p.parse_args()
    started=time.perf_counter();setup(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=args.output_dir
    if out.exists() and any(out.iterdir()):raise ValueError('preserve previous smoke; use fresh output')
    out.mkdir(parents=True)
    source_check={}
    for folder in ('agents/goal_review','agents/strategic'):
        for source in sorted((args.reference_snapshot/folder).glob('*.py')):
            relative=source.relative_to(args.reference_snapshot)
            source_check[str(relative)]=digest(source)==digest(ROOT/relative)
    if not source_check or not all(source_check.values()):raise ValueError('delivered IL source differs from active source')
    il=ROOT/'experiments/goal_review/E6_candidate_seed42/best_model.pth'
    reference=args.reference_snapshot.parent/'E6_candidate_seed42/best_model.pth'
    assert digest(il)==digest(reference)=='4a356ca463c5beee4acb4949cf93e8b6b453634262167025cd54043d88b2013b'
    from ml_workspace.resource_guard import cuda_memory_guard,cuda_memory_stats
    from ml_workspace.tracking import tracked_run
    import mlflow
    with tracked_run('mappo-v5-interface-smoke',experiment_name='lux-s3-mappo-v5',
        params={'seed':42,'formal_training':False,'max_seconds':115,'no_complete_games':True}) as run:
        result={'provenance':provenance(),'IL_source_snapshot_checks':source_check,'IL_sha256':digest(il),
            'smoke_only':True,'pilot_games':0,'formal_training':False,'mlflow_run_id':run.info.run_id,
            'source_sha256':sources(),'check_script_sha256':digest(Path(__file__))}
        cfg=load_config(ROOT/'configs/mappo_v5/E0_frozen.yaml');actor,critic,init=initialize(cfg,42)
        # Causal replay INPUTS only for a broad R initial-output audit; no labels.
        path=sorted((ROOT/'data/dataset_v3_goal_review').glob('*.npz'))[0]
        with np.load(path,allow_pickle=False) as z:d=unpack_review(z)
        ids=np.linspace(0,len(d['steps'])-1,24).astype(int)
        from agents.strategic.data import history_at
        packet={'history':torch.from_numpy(np.stack([history_at(d['features'],d['steps'],d['match_steps'],int(t)) for t in ids])).cuda(),
            'positions':torch.from_numpy(d['positions'][ids].astype(np.int64)).cuda(),
            'units':torch.from_numpy(d['unit_vectors'][ids].astype(np.float32)).cuda(),
            'alive':torch.from_numpy(d['active'][ids]).cuda()}
        with cuda_memory_guard('mappo-v5-causal-input-audit'):
            result['original_R']=probe_r(actor,packet)
            actor_initial={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
            neutralize_r(actor);result['neutral_R']=probe_r(actor,packet)
            result['neutral_modified_tensors']=[k for k,v in actor.state_dict().items() if not torch.equal(v.cpu(),actor_initial[k])]
            assert set(result['neutral_modified_tensors'])=={'replan_head.weight','replan_head.bias'}
        result['R_audit_scope']={'replay':path.name,'frames':ids.tolist(),'only_existing_causal_inputs':True,
            'missing_previous_mode_support':'zero-count conditions are unobserved, not assigned performance scores'}
        env=TeamEnv(cfg,out/'worker.log');result['arms']={}
        try:
            for arm in ('E0_frozen','E1_GM_pilot','E2_RMG_pilot'):
                if time.perf_counter()-started>85:raise TimeoutError('do not expand short checks into long training')
                c=load_config(ROOT/f'configs/mappo_v5/{arm}.yaml');env.config=c
                setup(42);actor,critic,init=initialize(c,42);opt=optimizer(actor,critic,c)
                initial=state_hash(actor.state_dict());s=env.reset(15001,0,'v2');buffer=[]
                with cuda_memory_guard('mappo-v5-'+arm):
                    while not env.truncated:
                        actor.eval();critic.eval();b=device(s)
                        with torch.no_grad():
                            outputs=policy_outputs(actor,b,c['model'])
                            assert outputs[0].shape==(1,16) and outputs[1].shape==(1,16,5) and outputs[2].shape==(1,16,5,24,24)
                            a,_=sample(actor,b,c);value=float(critic(b)[0])
                        ns,r,terminal,truncated,info=env.step(a,max_steps=18)
                        with torch.no_grad():nv=0. if terminal else float(critic(device(ns))[0])
                        expected=sum(c['ppo']['gamma']**j*x for j,x in enumerate(info['primitive_rewards']))
                        assert abs(expected-r)<1e-12
                        buffer.append(Transition(s,ns,a,r,info['tau'],value,nv,terminal,truncated,0,
                            info['engine_terminal'],info['engine_truncated'],info['task_complete'],tuple(info['primitive_rewards'])));s=ns
                        if terminal:raise AssertionError('short test unexpectedly completed a game')
                    torch.save(buffer[:2],out/f'{arm}_rollout_sample.pth')
                    save_rollout(out/f'{arm}_behavior_journal.npz',buffer,c['ppo']['gamma'])
                    journal=load_rollout(out/f'{arm}_behavior_journal.npz')
                    np.testing.assert_array_equal(journal['action__old_logprob'],torch.cat([t.action['old_logprob'] for t in buffer]).numpy())
                    metrics=None
                    if arm!='E0_frozen':
                        test_cfg=copy.deepcopy(c);test_cfg['ppo']['epochs']=1
                        metrics=update(actor,critic,opt,buffer[-4:],test_cfg)
                    else:assert state_hash(actor.state_dict())==initial
                    # CPU game-boundary checkpoint / optimizer / RNG restoration.
                    progress={'next_game':0,'note':'software smoke checkpoint, no complete games'}
                    save_checkpoint(out/f'{arm}_smoke.pth',actor,critic,opt,c,progress,{'initialization':init,'source_sha256':{}},smoke=True)
                    saved=torch.load(out/f'{arm}_smoke.pth',weights_only=False,map_location='cpu')
                    copy_actor,copy_critic,_=initialize(c,42);copy_opt=optimizer(copy_actor,copy_critic,c)
                    copy_actor.load_state_dict(saved['actor_state_dict']);copy_critic.load_state_dict(saved['critic_state_dict'])
                    copy_opt.load_state_dict(saved['optimizer_state_dict']);restore_rng(saved['rng'])
                    assert state_hash(copy_actor.state_dict())==state_hash(actor.state_dict())
                    assert state_hash(copy_critic.state_dict())==state_hash(critic.state_dict())
                    lhs,rhs=opt.state_dict(),copy_opt.state_dict()
                    assert lhs['param_groups']==rhs['param_groups']
                    for key,values in lhs['state'].items():
                        for field,x in values.items():
                            y=rhs['state'][key][field]
                            assert torch.equal(x.cpu(),y.cpu()) if isinstance(x,torch.Tensor) else x==y
                    restore_rng(saved['rng']);draw=torch.rand(4,device='cuda');npdraw=np.random.rand(4)
                    restore_rng(saved['rng']);assert torch.equal(draw,torch.rand(4,device='cuda'))
                    np.testing.assert_array_equal(npdraw,np.random.rand(4))
                    result['arms'][arm]={'summary':env.summary(),'ppo':metrics,'initialization':init,
                        'checkpoint_roundtrip':True,'optimizer_roundtrip':True,'compressed_journal_roundtrip':True,
                        'duration_list':[t.tau for t in buffer]}
            # A labelled-as-synthetic gradient fixture, NOT a return experiment.
            # Exercise Continue and Replan in the SAME true IL Actor update.
            c=load_config(ROOT/'configs/mappo_v5/E2_RMG_pilot.yaml');actor,critic,_=initialize(c,42)
            opt=optimizer(actor,critic,c);b={k:v.repeat((4,)+(1,)*(v.ndim-1)) for k,v in s.items()}
            b['review']=b['alive'].clone();b=device(b)
            with torch.no_grad():
                action,_=sample(actor,b,c);values=critic(b)
            assert action['branch_mask'][...,0].any() and action['branch_mask'][...,1].any()
            fixture=[Transition({k:v[i:i+1] for k,v in b.items()},{k:v[i:i+1] for k,v in b.items()},
                {k:v[i:i+1] for k,v in action.items() if isinstance(v,torch.Tensor)},r,t,float(values[i]),0.,False,True,i)
                for i,(r,t) in enumerate(zip((1.,-1.,.5,0.),(1,3,8,2),strict=True))]
            c['ppo']['epochs']=1;gradient_check=update(actor,critic,opt,fixture,c)
            assert all(gradient_check[k]>0 for k in ('grad_R','grad_Mode','grad_Goal'))
            result['synthetic_joint_update']={'metrics':gradient_check,
                'scope':'software gradient fixture with hand assigned test rewards; no performance/learning claim'}
            # Actual round boundary with unchanged public belief object. No Actor
            # forward or optimizer here, only deterministic no-op primitive steps.
            cfg=load_config(ROOT/'configs/mappo_v5/E0_frozen.yaml');env.config=cfg;s=env.reset(15001,0,'v2')
            reconstructor=env.builder.reconstructor;seen_before=None;crossed=False;belief_before=None
            while env.frame['step']<103:
                if time.perf_counter()-started>105:raise TimeoutError('round-boundary test exceeded bounded budget')
                before=env.frame
                response=env.bridge.request(op='step',actions=np.zeros((16,3),int).tolist())
                env.obs=response['obs'];env.frame=env.builder.observe(env.obs);env.manager.observe(env.frame)
                if before['match_step']==100:
                    crossed=True
                    assert env.frame['match_step']==0 and env.builder.reconstructor is reconstructor
                    assert not env.manager.valid.any() and not env.frame['alive'].any()
                    assert (env.frame['history'][1:]==0).all()
                    assert not response['task_complete'] and not response['engine_terminal'] and not response['engine_truncated']
                    assert all(reconstructor.known_relics.get(k)==v for k,v in belief_before.items())
                if before['match_step']==99:
                    seen_before=len(reconstructor.known_relics)
                    belief_before=dict(reconstructor.known_relics)
            assert crossed
            result['actual_match_reset']={'passed':True,'primitive_steps':103,'task_terminal':False,
                'history_cleared':True,'same_reconstructor':True,'known_relics_before':seen_before,
                'known_relics_after':len(reconstructor.known_relics),'intent_cleared':True}
        finally:env.close()
        result['wall_seconds']=time.perf_counter()-started;result['cuda_memory']=asdict(cuda_memory_stats())
        assert result['wall_seconds']<115
        result['status']='PASSED';save_json(out/'summary.json',result)
        mlflow.log_artifacts(str(out));print(json.dumps({'status':'PASSED','seconds':result['wall_seconds'],
            'pilot_games':0,'cuda_memory':result['cuda_memory'],'mlflow':run.info.run_id}),flush=True)


if __name__=='__main__':main()
