"""Authorized finite follow-up pipeline. No downloads, holdout or formal RL."""
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.goal_review.config import load_config
from agents.strategic.experiment import digest,save_json,setup

PYTHON='/home/issue/ml-workspace/.venv/bin/python'
BASE=ROOT/'experiments/goal_review'
PROGRESS=BASE/'autonomous_progress'
DATA=ROOT/'data/dataset_v3_goal_review'
PENDING=('E3_dilated','E4_without_r','E4_with_r','E5_corrupted')


def read(path):return json.loads(Path(path).read_text())


def finished(path):return path.exists() and read(path).get('status')=='FINISHED'


def status(stage,**values):
    value={'stage':stage,**values}
    save_json(PROGRESS/'pipeline_status.json',value);print(json.dumps(value),flush=True)


def subprocess_checked(args,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:subprocess.run([PYTHON,*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)


def verify(arm):
    subprocess_checked(['scripts/summarize_goal_review_run.py','--run-dir',BASE/f'{arm}_seed42',
        '--output-dir',BASE/f'{arm}_result_audit','--verify-only'],ROOT/'outputs/goal_review'/f'{arm}_result_audit.log')


def study_command(arm,output,etas):
    return [PYTHON,'-u','scripts/evaluate_goal_review_study.py','--checkpoint',str(BASE/f'{arm}_seed42/best_model.pth'),
        '--data-dir',str(DATA),'--output-dir',str(output),'--hysteresis-values',*map(str,etas),
        '--closed-loop-steps','128','--sides','0','1','--batch-games','16']


def run_memory_studies():
    specifications=[('without_r','E4_without_r',[0.]),('with_r','E4_with_r',[0.,.25]),('corrupted','E5_corrupted',[0.,.25])]
    active=[];pending=list(specifications)
    while pending or active:
        while pending and len(active)<2:
            key,arm,etas=pending.pop(0);out=BASE/f'E5_memory_{key}'
            if finished(out/'summary.json'):continue
            log=(ROOT/'outputs/goal_review'/f'E5_memory_{key}.log').open('a')
            proc=subprocess.Popen(study_command(arm,out,etas),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            active.append((key,proc,log));status('FROZEN_MEMORY_EVALUATION',active=[x[0] for x in active],pending=[x[0] for x in pending])
        for item in list(active):
            key,proc,log=item;code=proc.poll()
            if code is None:continue
            log.close();active.remove(item)
            if code:raise RuntimeError(f'frozen study failed: {key}, exit={code}; prior outputs preserved')
        if pending or active:time.sleep(15)


def memory_score(summary,eta):
    metrics=summary['conditions'][str(float(eta))]
    scores=[metrics[c]['all_frame_held_intent']['joint'] for c in ('scripted','corrupted','policy_replay')]
    counts=[metrics[c]['all_frame_held_intent']['units'] for c in ('scripted','corrupted','policy_replay')]
    if len(set(counts))!=1:raise ValueError('fixed-denominator condition comparison violated')
    return min(scores),float(np.mean(scores))


def select_candidate(criteria):
    architectures={'current':'E1_clean','res6':'E3_res6','res8':'E3_res8','dilated6':'E3_dilated'}
    loss_arms={'normal':'E1_clean','class_weighted':'E2_weighted','balanced_sampler':'E2_balanced'}
    architecture_results={key:read(BASE/f'{arm}_seed42/summary.json') for key,arm in architectures.items()}
    loss_results={key:read(BASE/f'{arm}_seed42/summary.json') for key,arm in loss_arms.items()}
    arch=max(architectures,key=lambda key:(architecture_results[key]['metrics']['joint_mode_goal_accuracy'],
                                         architecture_results[key]['metrics']['mode']['macro_f1']))
    loss=max(loss_arms,key=lambda key:(loss_results[key]['metrics']['mode']['macro_f1'],
                                     loss_results[key]['metrics']['joint_mode_goal_accuracy']))
    studies={key:read(BASE/f'E5_memory_{key}/summary.json') for key in ('with_r','corrupted')}
    for key,s in studies.items():
        if s.get('smoke_only') or len(s['plan']['validation_files'])!=145:raise ValueError(f'incomplete context study {key}')
    context=max(studies,key=lambda key:memory_score(studies[key],0.))
    eta=max((0.,.25),key=lambda value:(*memory_score(studies[context],value),-value))
    source='E4_with_r' if context=='with_r' else 'E5_corrupted'
    config=copy.deepcopy(load_config(ROOT/'configs/goal_review'/f'{source}.yaml'))
    config['model'].update(architecture=arch,hysteresis=eta,replan=True)
    config['study']['loss']=loss
    evidence={'criteria':criteria,'architecture':arch,'loss':loss,'context_source':source,'hysteresis':eta,
        'R_head':True,'seed':42,'selection_scope':'validation-selected exploratory composition; no untouched test claim',
        'architecture_candidates':{key:{'joint':s['metrics']['joint_mode_goal_accuracy'],'macro_f1':s['metrics']['mode']['macro_f1']} for key,s in architecture_results.items()},
        'loss_candidates':{key:{'joint':s['metrics']['joint_mode_goal_accuracy'],'macro_f1':s['metrics']['mode']['macro_f1']} for key,s in loss_results.items()},
        'context_scores_eta0':{key:memory_score(s,0.) for key,s in studies.items()},
        'eta_scores_selected_context':{str(e):memory_score(studies[context],e) for e in (0.,.25)},
        'dataset_manifest_sha256':digest(DATA/'manifest.json'),
        'memory_summary_sha256':{key:digest(BASE/f'E5_memory_{key}/summary.json') for key in studies}}
    return config,evidence


def main():
    setup();PROGRESS.mkdir(parents=True,exist_ok=True)
    authorization=read(PROGRESS/'plan.json')
    if authorization.get('execution_owner')!='agent':raise ValueError('explicit autonomous execution plan required')
    criteria={'label':'strict clean only; noisy labels remain quarantined control',
        'architecture':'max validation Joint, tie mode Macro-F1, among fixed E3 arms/current; each arm checkpoint already selected by preregistered Macro-F1',
        'loss_sampling':'max E2 mode Macro-F1, tie Joint, among normal/weighted/balanced',
        'R':'required True for final requested pi(r,m,g) interface',
        'context':'max worst-case all-frame held-intent Joint across scripted/corrupted/policy-memory at eta0; tie mean',
        'hysteresis':'same selected context checkpoint; max worst-case then mean all-frame held Joint over eta0/.25; exact tie chooses eta0',
        'budget':{'epochs':6,'updates_per_epoch':1000,'batch_units':16,'seed':42},
        'interpretation':'exploratory validation selection, no statistical winner or independent test claim',
        'no_formal_RL':True,'source_sha256':digest(Path(__file__))}
    prior=PROGRESS/'candidate_selection_criteria.json'
    if prior.exists() and read(prior)!=criteria:raise ValueError('selection criteria changed after registration')
    save_json(prior,criteria)
    from ml_workspace.tracking import tracked_run
    import mlflow
    with tracked_run('authorized_goal_review_followups',experiment_name='lux-s3-goal-review',params={
        'seed':42,'operation':'finite_controlled_IL_study','authorization':'explicit_user_2026-10-17',
        'training_budget_epochs':6,'formal_RL':False}) as parent:
        last=None
        while True:
            completed=[arm for arm in PENDING if finished(BASE/f'{arm}_seed42/summary.json')]
            if completed!=last:status('WAITING_FROZEN_TRAINING_ARMS',completed=completed,total=len(PENDING));last=completed
            if len(completed)==len(PENDING):break
            for arm in PENDING:
                log=ROOT/'outputs/goal_review'/f'{arm}_seed42.log'
                if log.exists() and 'Traceback (most recent call last)' in log.read_text():
                    raise RuntimeError(f'training error in {arm}; inspect log, no artifacts removed')
            time.sleep(30)
        for arm in PENDING:verify(arm)
        subprocess_checked(['scripts/compare_goal_review_capacity.py'],ROOT/'outputs/goal_review/E3_capacity_comparison.log')
        run_memory_studies()
        subprocess_checked(['scripts/compare_goal_review_replan.py'],ROOT/'outputs/goal_review/E45_comparison.log')
        config,selection=select_candidate(criteria)
        save_json(PROGRESS/'candidate_selection.json',selection)
        config_path=ROOT/'configs/goal_review/E6_candidate.yaml'
        if config_path.exists() and load_config(config_path)!=config:raise ValueError('existing candidate config differs')
        if not config_path.exists():config_path.write_text(yaml.safe_dump(config,sort_keys=False))
        candidate=BASE/'E6_candidate_seed42'
        if not finished(candidate/'summary.json'):
            status('TRAINING_E6_CANDIDATE',selection=selection)
            subprocess_checked(['scripts/train_goal_review.py','--config',config_path,'--data-dir',DATA,
                '--output-dir',candidate,'--seed','42','--run-training'],ROOT/'outputs/goal_review/E6_candidate_seed42.log')
        verify('E6_candidate')
        evaluation=BASE/'E6_memory_candidate'
        if not finished(evaluation/'summary.json'):
            status('EVALUATING_E6_CANDIDATE')
            subprocess_checked(study_command('E6_candidate',evaluation,[0.,.25])[1:],ROOT/'outputs/goal_review/E6_memory_candidate.log')
        subprocess_checked(['scripts/report_goal_review.py'],ROOT/'outputs/report_goal_review.log')
        result={'status':'FINISHED','training_arms':list(PENDING),'candidate_selection':selection,
                'candidate_summary_sha256':digest(candidate/'summary.json'),
                'candidate_memory_sha256':digest(evaluation/'summary.json'),
                'readiness':'NOT READY','readiness_note':'requires final scientific audit; never auto-approve from a single metric',
                'formal_RL_updates':0,'parent_mlflow_run_id':parent.info.run_id}
        save_json(PROGRESS/'completion.json',result);mlflow.log_artifacts(str(PROGRESS))
        status('FINISHED_AWAITING_SCIENTIFIC_SYNTHESIS',parent_mlflow_run_id=parent.info.run_id)


if __name__=='__main__':main()
