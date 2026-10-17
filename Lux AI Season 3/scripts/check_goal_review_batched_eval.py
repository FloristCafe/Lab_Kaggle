"""Compare sequential and independent-game batching on a bounded causal prefix."""
import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.goal_review.config import load_config
from agents.goal_review.data import paths_for_split
from agents.goal_review.model import ReviewActor
from agents.goal_review.memory import ReviewMemory,frame_at
from agents.goal_review.data import unpack_review
from agents.goal_review.runtime import RuntimeReviewMemory
from agents.strategic.experiment import digest,save_json,setup
from scripts.evaluate_goal_review import replay_evaluation
from scripts.goal_review_batched_eval import batched_replay_evaluation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/goal_review/batched_eval_sanity')
    args=p.parse_args()
    setup();started=time.perf_counter()
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    config=load_config(ROOT/'configs/goal_review/E4_with_r.yaml')
    actor=ReviewActor(config['model']).cuda().eval()
    # Force nontrivial Continue and invalid->Replan branches without fitting.
    with torch.no_grad():actor.replan_head.weight.zero_();actor.replan_head.bias.fill_(-1)
    paths=paths_for_split(ROOT/'data/dataset_v3_goal_review','validation')[:2]
    from ml_workspace.tracking import tracked_run
    import mlflow
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('preserve earlier sanity artifact; choose a fresh directory')
    checks={}
    with tracked_run('batched_eval_causal_sanity',experiment_name='lux-s3-goal-review',params={'games':2,'prefix_frames':120,'training_updates':0}) as run:
        with np.load(paths[0],allow_pickle=False) as z:d=unpack_review(z)
        public=json.loads(str(d['metadata']))['env_cfg']
        old_memory=ReviewMemory(config['review'],public);new_memory=RuntimeReviewMemory(config['review'],public)
        for t in range(110):
            f=frame_at(d,t);old_review=old_memory.observe(f);new_review=new_memory.observe(f)
            assert old_review==new_review
            np.testing.assert_array_equal(new_memory.vectors(f).astype(np.float16),d['unit_vectors'][t])
            if old_review:
                old_memory.scripted_decision(f,config['tactical']);new_memory.scripted_decision(f,config['tactical'])
            np.testing.assert_array_equal(old_memory.ages,new_memory.ages)
        assert new_memory.stats['identical_replan_renewals']==0
        for condition in ('scripted','corrupted','policy_replay'):
            with patch('scripts.evaluate_goal_review.ReviewMemory',RuntimeReviewMemory):
                old=replay_evaluation(actor,config,paths,condition,max_frames=120)
            new=batched_replay_evaluation(actor,config,paths,condition,batch_games=2,max_frames=120)
            serial=batched_replay_evaluation(actor,config,paths,condition,batch_games=1,max_frames=120)
            for key in ('environment_frames','macro_review_epochs','previous_mode_counts','valid_goal_units'):
                assert old[key]==new[key],(condition,key,old[key],new[key])
            for key in ('mode','executed_mode','replan','free_replan'):
                assert old[key]['confusion_matrix']==new[key]['confusion_matrix'],(condition,key)
            for key in ('joint_mode_goal_accuracy','goal_top1','goal_top5','goal_top10','goal_manhattan','previous_age_mean'):
                np.testing.assert_allclose(old[key],new[key],rtol=0,atol=1e-7,err_msg=f'{condition}: {key}')
            for key in ('goal_entropy','goal_peak'):
                np.testing.assert_allclose(old[key],new[key],rtol=0,atol=1e-5,err_msg=f'{condition}: FP32 reduction {key}')
            assert serial['all_frame_held_intent']==new['all_frame_held_intent'],condition
            expected=0
            for path in paths:
                with np.load(path,allow_pickle=False) as z:expected+=int((z['label_valid'][:120]&z['active'][:120]).sum())
            assert new['all_frame_held_intent']['units']==expected
            checks[condition]={'passed':True,'frames':new['environment_frames'],'reviews':new['macro_review_epochs'],
                               'batching':'independent games only, no cross-time batching'}
        result={'checks':checks,'wall_seconds':time.perf_counter()-started,'training_updates':0,'smoke_only':True,
                'runtime_sha256':digest(ROOT/'agents/goal_review/runtime.py'),'script_sha256':digest(Path(__file__)),
                'dataset_manifest_sha256':digest(ROOT/'data/dataset_v3_goal_review/manifest.json'),
                'mlflow_run_id':run.info.run_id}
        save_json(out/'summary.json',result);mlflow.log_artifacts(str(out));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
