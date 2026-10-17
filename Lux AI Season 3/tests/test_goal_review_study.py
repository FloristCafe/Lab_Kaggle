import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.goal_review.config import load_config
from agents.goal_review.loss import objective
from agents.goal_review.runtime import RuntimeReviewMemory
from scripts import run_goal_review_followups as pipeline


class FrozenInterventionTests(unittest.TestCase):
    def test_hysteresis_changes_no_training_gradients(self):
        torch.manual_seed(42)
        configs=[load_config(ROOT/'configs/goal_review'/name) for name in ('E4_with_r.yaml','E5_hysteresis.yaml')]
        outputs=(torch.randn(2,16,requires_grad=True),torch.randn(2,16,5,requires_grad=True),
                 torch.randn(2,16,5,24,24,requires_grad=True))
        alive=torch.ones(2,16,dtype=torch.bool);units=torch.zeros(2,16,22);units[:,:,11]=1
        regions=torch.zeros(2,16,24,24,dtype=torch.bool);regions[:,:,7,9]=True
        targets={'valid':alive,'alive':alive,'replan_valid':alive,'units':units,
                 'mode':torch.zeros(2,16,dtype=torch.long),'regions':regions,
                 'replan':(torch.arange(16)%2).expand(2,-1).float(),'confidence':torch.ones(2,16)}
        losses=[objective(outputs,targets,cfg)[0] for cfg in configs]
        gradients=[torch.autograd.grad(loss,outputs,retain_graph=True) for loss in losses]
        torch.testing.assert_close(losses[0],losses[1],rtol=0,atol=0)
        for a,b in zip(*gradients,strict=True):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_matched_inference_configs_only_change_eta(self):
        for a,b in (('E4_with_r.yaml','E5_hysteresis.yaml'),('E5_corrupted.yaml','E5_combination.yaml')):
            left=copy.deepcopy(load_config(ROOT/'configs/goal_review'/a))
            right=copy.deepcopy(load_config(ROOT/'configs/goal_review'/b))
            self.assertEqual(left['model'].pop('hysteresis'),0.)
            self.assertEqual(right['model'].pop('hysteresis'),.25)
            self.assertEqual(left,right)


class SelectionTests(unittest.TestCase):
    @staticmethod
    def summary(scores,counts=None):
        counts=counts or [100]*3
        return {'conditions':{'0.0':{name:{'all_frame_held_intent':{'joint':score,'units':count}}
            for name,score,count in zip(('scripted','corrupted','policy_replay'),scores,counts,strict=True)}}}

    def test_worst_case_has_priority_over_good_teacher_score(self):
        teacher_good=self.summary([.8,.1,.1]);robust=self.summary([.2,.2,.2])
        self.assertGreater(pipeline.memory_score(robust,0),pipeline.memory_score(teacher_good,0))

    def test_changing_label_denominator_cannot_win_context_selection(self):
        with self.assertRaisesRegex(ValueError,'fixed-denominator'):
            pipeline.memory_score(self.summary([.5,.5,.5],[100,100,50]),0)

    def test_candidate_rules_compose_components_and_keep_binary_R(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            values={'E1_clean':(.20,.45),'E3_res6':(.21,.44),'E3_res8':(.19,.60),
                    'E3_dilated':(.21,.46),'E2_weighted':(.20,.53),'E2_balanced':(.18,.54)}
            for arm,(joint,macro) in values.items():
                folder=root/f'{arm}_seed42';folder.mkdir()
                (folder/'summary.json').write_text(json.dumps({'metrics':{'joint_mode_goal_accuracy':joint,'mode':{'macro_f1':macro}}}))
            for key,scores in (('with_r',[.3,.1,.3]),('corrupted',[.2,.2,.2])):
                s=self.summary(scores);s['plan']={'validation_files':['fixture']*145};s['smoke_only']=False
                s['conditions']['0.25']=self.summary([.3,.2,.2])['conditions']['0.0']
                folder=root/f'E5_memory_{key}';folder.mkdir();(folder/'summary.json').write_text(json.dumps(s))
            with patch.object(pipeline,'BASE',root):config,evidence=pipeline.select_candidate({'software_test':True})
            self.assertEqual(config['model']['architecture'],'dilated6')
            self.assertEqual(config['study']['loss'],'balanced_sampler')
            self.assertEqual(config['study']['context'],'corrupted')
            self.assertTrue(config['model']['replan'])
            self.assertEqual(config['model']['hysteresis'],.25)
            self.assertEqual(evidence['context_source'],'E5_corrupted')


class IdentityAgeTests(unittest.TestCase):
    def test_identical_replan_does_not_refresh_hysteresis_lifetime(self):
        import numpy as np
        cfg=load_config(ROOT/'configs/goal_review/common.yaml')
        memory=RuntimeReviewMemory(cfg['review'],{'unit_move_cost':1})
        memory.valid[0]=True;memory.modes[0]=3;memory.goals[0]=[12,12];memory.ages[0]=81
        memory.blocked[0]=7
        alive=np.zeros(16,bool);alive[0]=True
        r=alive.copy();m=np.full(16,3);g=np.full(16,12*24+12)
        memory.assign_choices(r,m,g,alive)
        self.assertEqual(memory.ages[0],81)
        self.assertEqual(memory.blocked[0],7)
        self.assertEqual(memory.stats['identical_replan_renewals'],1)
        for t in range(19):
            frame={'alive':alive,'positions':np.zeros((16,2),int),'energy':np.full(16,100),
                'base':np.zeros((21,24,24)),'enemy':[],'step':t,'match_step':t,'points':np.zeros(2),'rank':0}
            memory.observe(frame)
        self.assertFalse(memory.valid[0])
        self.assertIn('goal_memory_expired',memory.reasons)

    def test_new_goal_identity_resets_age(self):
        import numpy as np
        cfg=load_config(ROOT/'configs/goal_review/common.yaml')
        memory=RuntimeReviewMemory(cfg['review'],{'unit_move_cost':1})
        memory.valid[0]=True;memory.modes[0]=3;memory.goals[0]=[12,12];memory.ages[0]=81
        alive=np.zeros(16,bool);alive[0]=True
        memory.assign_choices(alive,np.full(16,3),np.full(16,13*24+12),alive)
        self.assertEqual(memory.ages[0],0)

    def test_identical_replan_does_not_fake_another_first_arrival(self):
        import numpy as np
        cfg=load_config(ROOT/'configs/goal_review/common.yaml')
        memory=RuntimeReviewMemory(cfg['review'],{'unit_move_cost':1})
        memory.valid[0]=True;memory.modes[0]=3;memory.goals[0]=[0,0];memory.reached[0]=True
        alive=np.zeros(16,bool);alive[0]=True
        memory.assign_choices(alive,np.full(16,3),np.zeros(16,int),alive)
        self.assertTrue(memory.reached[0])
        frame={'alive':alive,'positions':np.zeros((16,2),int),'energy':np.full(16,100),
            'base':np.zeros((21,24,24)),'enemy':[],'step':0,'match_step':0,'points':np.zeros(2),'rank':0}
        memory.observe(frame)
        self.assertEqual(memory.stats['goal_completions'],0)


if __name__=='__main__':unittest.main()
