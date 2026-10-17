"""Controlled Goal IL training. Long runs require the user's --run-training."""
import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.goal_review.config import load_config
from agents.goal_review.data import ReviewDataset, UnitEpochSampler, class_weights, paths_for_split
from agents.goal_review.loss import objective
from agents.goal_review.metrics import Metrics, evaluate
from agents.goal_review.model import ReviewActor
from agents.strategic.data import collate_goal
from agents.strategic.experiment import digest, provenance, save_json, setup


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/dataset_v3_goal_review')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--run-training',action='store_true')
    p.add_argument('--smoke-steps',type=int,default=0,help='1..4 tiny pipeline updates; never a research result')
    args=p.parse_args()
    if not args.run_training and not args.smoke_steps:
        print('No training started. Full controlled runs exceed 2 minutes; execute with --run-training in VS Code.')
        return
    config=load_config(args.config);config['study']['seed']=args.seed
    smoke=bool(args.smoke_steps)
    if smoke:
        if not 1<=args.smoke_steps<=4 or args.run_training:raise ValueError('smoke is 1..4 steps and separate from full training')
        config['study'].update(epochs=1,updates_per_epoch=args.smoke_steps,batch_units=4,max_validation_units=16)
    setup(args.seed)
    manifest=json.loads((args.data_dir/'manifest.json').read_text())
    if not manifest.get('complete') or (not smoke and (manifest.get('pilot') or len(manifest['files'])<=400)):
        raise ValueError('require completed expanded clean corpus >400 games; pilot/partial data refused')
    for name,expected in manifest['pipeline_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError(f'dataset producer changed since build: {name}')
    for section in ('labeler','review','tactical'):
        if config[section]!=manifest['config'][section]:
            raise ValueError(f'config {section} does not match frozen dataset semantics; rebuild a separate dataset')
    train_paths=paths_for_split(args.data_dir,'train');val_paths=paths_for_split(args.data_dir,'validation')
    if smoke and not val_paths:val_paths=train_paths[:1]
    if not smoke:assert not set(train_paths)&set(val_paths)
    train_seeds={manifest['files'][p.name]['source_seed'] for p in train_paths}-{None}
    val_seeds={manifest['files'][p.name]['source_seed'] for p in val_paths}-{None}
    if not smoke and train_seeds&val_seeds:raise ValueError('source seed leakage')
    for path in train_paths+val_paths:
        if digest(path)!=manifest['files'][path.name]['npz_sha256']:raise ValueError('dataset hash mismatch')
    train=ReviewDataset(train_paths,config,training=True)
    validation=ReviewDataset(val_paths,config,training=False)
    if not len(train) or not len(validation):raise ValueError('empty split')
    study=config['study']
    budget=study['updates_per_epoch']*study['batch_units']
    sampler=UnitEpochSampler(train,args.seed,budget,balanced=study['loss']=='balanced_sampler')
    loader=DataLoader(train,batch_size=study['batch_units'],sampler=sampler,num_workers=0,collate_fn=collate_goal,pin_memory=True)
    vl=DataLoader(validation,batch_size=study['batch_units'],collate_fn=collate_goal,num_workers=0,pin_memory=True)
    out=args.output_dir or ROOT/'experiments/goal_review'/f'{"SMOKE_" if smoke else ""}{args.config.stem}_seed{args.seed}'
    if out.exists() and any(out.iterdir()):raise ValueError('immutable run directory already exists; choose a fresh path')
    out.mkdir(parents=True)
    (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    from ml_workspace.resource_guard import cuda_memory_guard,cuda_memory_stats
    from ml_workspace.tracking import tracked_run
    import mlflow
    started=time.perf_counter()
    with tracked_run(out.name,experiment_name='lux-s3-goal-review',params={'seed':args.seed,'architecture':config['model']['architecture'],
        'loss':study['loss'],'replan':config['model']['replan'],'context':study['context'],'unit_exposures_per_epoch':budget,
        'data_version':manifest['dataset_version']}) as run:
        model=ReviewActor(config['model']).cuda()
        optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=study['lr'],weight_decay=study['weight_decay'])
        weights=(torch.from_numpy(class_weights(train.mode_counts,study['class_weight_cap'])).cuda()
                 if study['loss']=='class_weighted' else None)
        metadata={'config':config,'provenance':provenance(),'dataset_manifest_sha256':digest(args.data_dir/'manifest.json'),
                  'smoke_only':smoke,'smoke_validation_may_overlap_train':smoke,
                  'train_files':[p.name for p in train_paths],'validation_files':[p.name for p in val_paths],
                  'training_unit_rows':len(train),'validation_unit_rows':len(validation),'training_mode_counts':train.mode_counts.tolist(),
                  'validation_mode_counts':validation.mode_counts.tolist(),'class_weights':None if weights is None else weights.tolist(),
                  'selection':'mode-head macro-F1, tie resolved joint correctness; patience preregistered',
                  'evaluation_target':'strict CLEAN labels for ALL arms, including noisy-trained control',
                  'sampling':'unit rows, identical budget, shard-grouped IO for normal and balanced',
                  'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'agents/goal_review').glob('*.py'))},
                  'training_script_sha256':digest(Path(__file__)),'mlflow_run_id':run.info.run_id}
        save_json(out/'manifest.json',metadata)
        best=(-float('inf'),-float('inf'));stale=0;rows=[]
        with cuda_memory_guard('controlled-goal-il-training'):
            for epoch in range(study['epochs']):
                train.epoch=sampler.epoch=epoch
                model.train();total_loss=0.;seen=np.zeros(5,np.int64);train_metrics=Metrics()
                for cpu in loader:
                    if smoke and time.perf_counter()-started>90:raise TimeoutError('90s smoke limit')
                    b={k:v.cuda(non_blocking=True) for k,v in cpu.items()}
                    probability=study['previous_context_dropout']
                    if probability:
                        if config['model']['replan']:raise ValueError('replan dropout must recompute memory-conditioned targets; use explicit corrupted condition')
                        drop=torch.rand(len(b['units']),device='cuda')<probability
                        b['units'][drop,:,8:17]=0
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):outputs=model(b['history'],b['positions'],b['units'],b['alive'])
                    loss,parts=objective(outputs,b,config,weights)
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite objective')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                    total_loss+=float(loss.detach())
                    mask=b['valid']&b['sampled']
                    seen+=np.bincount(b['mode'][mask].cpu().numpy(),minlength=5)
                    train_metrics.update(tuple(x.detach() for x in outputs),b,config['model'])
                metrics=evaluate(model,vl,config,max_units=study['max_validation_units'])
                training_stats=train_metrics.finish()
                metrics['training_epoch_mixed_policy']=training_stats
                metrics['training_sampled_mode_counts']=seen.tolist()
                save_json(out/f'validation_epoch_{epoch+1}.json',metrics)
                row={'epoch':epoch+1,'train_loss':total_loss/max(study['updates_per_epoch'],1),
                     'macro_f1':metrics['mode']['macro_f1'],'balanced_accuracy':metrics['mode']['balanced_accuracy'],
                     'joint_accuracy':metrics['joint_mode_goal_accuracy'],'explore_recall':metrics['per_mode']['EXPLORE']['recall'],
                     'replan_f1':metrics['replan']['f1'][1],'elapsed_seconds':time.perf_counter()-started}
                rows.append(row)
                with (out/'train_log.csv').open('w',newline='') as file:
                    writer=csv.DictWriter(file,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
                mlflow.log_metrics({k:v for k,v in row.items() if k!='epoch'},step=epoch+1)
                print(json.dumps(row),flush=True)
                score=(row['macro_f1'],row['joint_accuracy'])
                if score>best:
                    best,stale=score,0
                    torch.save({'actor_state_dict':model.state_dict(),'config':config,'epoch':epoch+1,
                                'metrics':metrics,'smoke_only':smoke,'dataset_manifest_sha256':metadata['dataset_manifest_sha256']},out/'best_model.pth')
                else:stale+=1
                if stale>=study['patience']:break
        checkpoint=torch.load(out/'best_model.pth',weights_only=False,map_location='cuda')
        model.load_state_dict(checkpoint['actor_state_dict'])
        metrics=evaluate(model,vl,config,max_units=study['max_validation_units'])
        save_json(out/'validation_log.json',metrics)
        summary={'best_epoch':checkpoint['epoch'],'stopped_epoch':rows[-1]['epoch'],'metrics':metrics,'config':config,
                 'train_games':len(train_paths),'validation_games':len(val_paths),'wall_seconds':time.perf_counter()-started,
                 'cuda_memory':asdict(cuda_memory_stats()),'status':'SMOKE_ONLY' if smoke else 'FINISHED',
                 'smoke_only':smoke,'ready_for_formal_mappo':'NOT READY',
                 'mlflow_run_id':run.info.run_id,'dataset_manifest_sha256':metadata['dataset_manifest_sha256']}
        save_json(out/'summary.json',summary);mlflow.log_artifacts(str(out))
        print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
