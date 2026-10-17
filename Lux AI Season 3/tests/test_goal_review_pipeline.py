import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.goal_review.config import load_config
from agents.goal_review.labels import clean_labels, legal_sap, replan_targets
from agents.goal_review.memory import ReviewMemory
from test_strategic_labels import trajectory


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT/'configs/goal_review/common.yaml')
        self.public = {'unit_move_cost':2,'unit_sap_cost':30,'unit_sap_range':4}

    def test_illegal_sap_is_removed_before_segmentation_and_mode_evidence(self):
        d = trajectory()
        d['env_cfg'] = self.public
        d['actions'][:,0] = (5,5,0)
        attempt, legal = legal_sap(d)
        self.assertEqual(int(attempt.sum()),20)
        self.assertEqual(int(legal.sum()),0)
        original = d['actions'].copy()
        labels,_ = clean_labels(d,self.cfg)
        self.assertEqual(float(labels['semantic_events'][:,:,21].sum()),0.)
        np.testing.assert_array_equal(original,d['actions'])
        d['actions'][:,0] = (5,4,4)
        self.assertEqual(int(legal_sap(d)[1].sum()),20)
        d['energies'][:,0] = 29
        self.assertEqual(int(legal_sap(d)[1].sum()),0)

    def test_right_censored_terminal_segment_is_not_a_goal(self):
        d = trajectory(30);d['env_cfg'] = self.public
        d['features'][:,20,8,8] = 1
        d['active'][-1,0] = False
        labels,segments = clean_labels(d,self.cfg)
        self.assertTrue(any(s['right_censored'] for s in segments))
        self.assertFalse(labels['label_valid'].any())

    def test_unknown_filtered_is_not_automatically_continue(self):
        units = np.zeros((16,22),np.float32)
        regions = np.zeros((16,24,24),bool)
        target,mask,_ = replan_targets(np.zeros(16,int),regions,np.zeros(16,bool),
            np.zeros(16,bool),np.zeros(16,bool),units,np.zeros(16,bool))
        self.assertFalse(mask.any())
        self.assertTrue((target==1).all())  # deterministic inference fallback, not training labels

    def test_continue_requires_matching_causal_previous_goal(self):
        units = np.zeros((16,22),np.float32)
        units[0,8:10] = (8/23,9/23);units[0,11]=1;units[0,12]=1
        regions = np.zeros((16,24,24),bool);regions[0,8,9]=True
        valid = np.zeros(16,bool);valid[0]=True
        r,mask,_ = replan_targets(np.zeros(16,int),regions,valid,np.zeros(16,bool),
                                  np.zeros(16,bool),units,valid)
        self.assertTrue(mask[0]);self.assertEqual(int(r[0]),0)
        units[0,8] = 7/23
        r,_,_ = replan_targets(np.zeros(16,int),regions,valid,np.zeros(16,bool),np.zeros(16,bool),units,valid)
        self.assertEqual(int(r[0]),1)

    def f(self,t):
        f = {'base':np.zeros((21,24,24),np.float32),'positions':np.zeros((16,2),np.int16),
             'energy':np.full(16,100,np.int16),'alive':np.zeros(16,bool),'enemy':[],
             'step':t,'match_step':t+1,'points':np.zeros(2,int),'rank':0}
        f['alive'][0]=True;f['positions'][0]=(8,8)
        return f

    def test_micro_stall_and_single_dodge_do_not_open_review(self):
        memory = ReviewMemory(self.cfg['review'],self.public)
        self.assertTrue(memory.observe(self.f(0)))
        r = np.ones(16,bool);goals = np.full(16,15*24+8)
        memory.assign_choices(r,np.zeros(16,int),goals,self.f(0)['alive'])
        for t in range(1,8):
            self.assertFalse(memory.observe(self.f(t),{'severe_threat':t==3}))
        self.assertTrue(memory.observe(self.f(8)))
        self.assertIn('persistent_blockage',memory.reasons)

    def test_continue_preserves_goal_age_and_invalid_v2_forces_review(self):
        memory = ReviewMemory(self.cfg['review'],self.public)
        memory.observe(self.f(0))
        memory.assign_choices(np.ones(16,bool),np.zeros(16,int),np.full(16,15*24+8),self.f(0)['alive'])
        memory.observe(self.f(1))
        memory.assign_choices(np.zeros(16,bool),np.ones(16,int),np.zeros(16,int),self.f(1)['alive'])
        self.assertEqual(int(memory.ages[0]),1)
        self.assertEqual(memory.goals[0].tolist(),[15,8])
        f = self.f(2);f['base'][19,15,8]=1
        self.assertTrue(memory.observe(f))
        self.assertFalse(memory.valid[0])


if __name__=='__main__':
    unittest.main()
