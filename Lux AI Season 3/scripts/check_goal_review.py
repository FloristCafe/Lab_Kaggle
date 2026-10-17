"""Bounded <90s sanity checks; never execute complete training experiments."""
import argparse
import gc
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
from agents.goal_review.data import ReviewDataset
from agents.goal_review.loss import objective
from agents.goal_review.model import ReviewActor,bounded_joint_bias,review_statistics,select_review_actions
from agents.strategic.data import collate_goal
from agents.strategic.experiment import save_json,setup
from agents.strategic.model import CentralizedCritic,actor_input
from agents.strategic.semimdp import clipped_actor_loss,smdp_gae
from scripts.evaluate_goal_review import online_probe,replay_evaluation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/dataset_v3_goal_review_pilot2')
    p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/goal_review/interface_smoke')
    args=p.parse_args();setup();started=time.perf_counter()
    # Compare behavior logprobs across batch shapes in true FP32. TF32
    # convolution kernels may change numerical precision with the batch size.
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    config=load_config(ROOT/'configs/goal_review/E4_with_r.yaml')
    paths=sorted(args.data_dir.glob('*.npz'));data=ReviewDataset(paths,config,True)
    cpu=collate_goal([data[int(i)] for i in np.linspace(0,len(data)-1,4).astype(int)])
    batch={k:v.cuda() for k,v in cpu.items()}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if (args.output_dir/'summary.json').exists():raise ValueError('preserve prior smoke; choose a fresh output directory')
    from ml_workspace.resource_guard import cuda_memory_guard,cuda_memory_stats
    from ml_workspace.tracking import tracked_run
    import mlflow
    results={}
    with tracked_run('goal_review_interface_smoke',experiment_name='lux-s3-goal-review',params={'seed':42,'optimizer_steps_per_architecture':2,'not_full_training':True}) as run:
        for arch in ('current','res6','res8','dilated6'):
            if time.perf_counter()-started>90:raise TimeoutError('90-second smoke budget exceeded; do not extend into long training')
            cfg={**config['model'],'architecture':arch};actor=ReviewActor(cfg).cuda()
            optimizer=torch.optim.Adam(actor.parameters(),lr=1e-3)
            torch.cuda.synchronize();tick=time.perf_counter()
            with cuda_memory_guard('goal-review-'+arch):
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):outputs=actor(*actor_input(batch))
                    loss,parts=objective(outputs,batch,{**config,'model':cfg});loss.backward();optimizer.step()
                with torch.no_grad():
                    recorded=[select_review_actions(actor(*(x[i:i+1] for x in actor_input(batch))),
                        batch['units'][i:i+1],batch['alive'][i:i+1],cfg) for i in range(len(batch['history']))]
                    r,m,g,old=[torch.cat([row[j] for row in recorded]) for j in range(4)]
                    outputs=actor(*actor_input(batch))
                    ml,gl=bounded_joint_bias(outputs[1],outputs[2],batch['units'],cfg)
                    new,_=review_statistics(outputs[0],ml,gl,r,m,g,batch['units'],batch['alive'],True)
                    error=float((new-old).abs().max())
                    assert error<1e-4, f'{arch}: single-row versus batched FP32 logprob error={error}'
                value,clip=clipped_actor_loss(new,old,torch.zeros(len(new),device='cuda'),batch['alive'])
                assert float(clip)==0.
                assert torch.isfinite(value)
            torch.cuda.synchronize()
            results[arch]={'optimizer_updates':2,'loss':float(loss.detach()),'logprob_error':error,
                'seconds_two_steps':time.perf_counter()-tick,'cuda_memory':asdict(cuda_memory_stats()),
                'parameters':sum(p.numel() for p in actor.parameters() if p.requires_grad)}
            if arch=='current':
                torch.save({'actor_state_dict':actor.state_dict(),'config':config,'smoke_only':True},args.output_dir/'smoke_model.pth')
            del actor,optimizer,outputs,loss;gc.collect();torch.cuda.empty_cache()
        a,ret=smdp_gae([.1,.2],[3,7],[.4,.3],[.3,.8],[False,False],[True,False])
        assert np.isfinite(a).all() and np.isfinite(ret).all()
        checkpoint=torch.load(args.output_dir/'smoke_model.pth',weights_only=False,map_location='cuda')
        actor=ReviewActor(config['model']).cuda().eval();actor.load_state_dict(checkpoint['actor_state_dict'])
        critic=CentralizedCritic().cuda()
        assert critic(*actor_input(batch)).shape==(4,)
        conditions={c:replay_evaluation(actor,config,paths[:1],c,max_frames=32) for c in ('scripted','corrupted','policy_replay')}
        online=online_probe(actor,config,args.output_dir,steps=32)
        result={'architectures':results,'memory_conditions':conditions,'online_interface':online,
            'smoke_only':True,'not_a_validation_experiment':True,'formal_training_executed':False,
            'mappo_optimizer_updates':0,'wall_seconds':time.perf_counter()-started,'mlflow_run_id':run.info.run_id}
        if result['wall_seconds']>120:raise TimeoutError('smoke exceeded requested two-minute boundary')
        save_json(args.output_dir/'summary.json',result);mlflow.log_artifacts(str(args.output_dir))
        print(json.dumps({'wall_seconds':result['wall_seconds'],'smoke_only':True,'mappo_updates':0}),flush=True)


if __name__=='__main__':main()
