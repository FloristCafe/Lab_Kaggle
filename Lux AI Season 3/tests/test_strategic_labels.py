import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.strategic.labeler import StrategicGoalLabeler, semantic_events


def trajectory(T=20):
    active = np.zeros((T,16),bool)
    active[:,0] = True
    positions = np.full((T,16,2),-1,np.int16)
    positions[:,0] = (8,8)
    features = np.zeros((T,21,24,24),np.float32)
    features[:,11] = .5
    return dict(positions=positions,active=active,energies=np.ones((T,16),np.int16)*100,
                features=features,steps=np.arange(T),match_steps=np.arange(T)+1,
                enemy_active=np.zeros((T,16),bool),enemy_positions=np.full((T,16,2),-1,np.int16),
                enemy_energies=np.zeros((T,16),np.int16),actions=np.zeros((T,16,3),np.int16),
                rank=np.zeros(T,np.int16),team_points=np.zeros((T,2),np.int16),team_id=0)


class StrategicLabelTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((ROOT/'configs/strategic_baseline.yaml').read_text())['labeler']
        self.labeler = StrategicGoalLabeler(self.cfg)

    def test_true_scoring_stable_anchor_and_no_12_step_mode(self):
        d = trajectory(40)
        d['positions'][:16,0,0] = np.arange(16)
        d['positions'][16:,0,0] = 15
        d['features'][:,20,15,8] = 1
        d['features'][:,10,15,8] = 1
        d['team_points'][:,0] = np.arange(40)
        labels,segments = self.labeler.label(d)
        good = labels['label_valid'][:,0]
        self.assertTrue(good.any())
        self.assertTrue((labels['goal'][good,0]==[15,8]).all())
        self.assertTrue(any(s['valid'] and s['end']-s['start']+1>12 for s in segments))

    def test_death_endpoint_without_anchor_is_invalid(self):
        d = trajectory()
        d['positions'][:,0,0] = np.arange(20)
        d['active'][-1,0] = False
        labels,segments = self.labeler.label(d)
        self.assertFalse(labels['label_valid'].any())
        self.assertTrue(any(s['right_censored'] for s in segments))

    def test_recharge_anchor_needs_low_energy_gain_and_positive_field(self):
        d = trajectory(20)
        d['energies'][:,0] = np.arange(20)*4+10
        d['features'][:,9,8,8] = 1
        labels,segments = self.labeler.label(d)
        self.assertTrue(any(s['valid'] and s['mode']=='RECHARGE' for s in segments))
        d['features'][:,9] = 0
        labels,_ = self.labeler.label(d)
        self.assertFalse(labels['label_valid'].any())

    def test_life_slot_reuse_and_round_reset_do_not_share_segments(self):
        d = trajectory(30)
        d['positions'][10:,0] = (0,0)
        d['match_steps'][20:] = np.arange(10)
        lives = list(self.labeler._lives(d,0))
        self.assertEqual([(a,b) for a,b,_ in lives],[(0,9),(10,19),(21,29)])

    def test_prefix_features_are_not_modified_by_labeler(self):
        d = trajectory()
        original = d['features'].copy()
        self.labeler.label(d)
        np.testing.assert_array_equal(d['features'],original)

    def test_enemy_advancing_does_not_attribute_approach_to_stationary_unit(self):
        d = trajectory(15)
        d['enemy_active'][:,0] = True
        d['enemy_positions'][:,0,0] = np.arange(22,7,-1)
        d['enemy_positions'][:,0,1] = 8
        events = semantic_events(d,self.cfg)
        self.assertEqual(float(events[:,0,24].sum()),0.)
        d['enemy_positions'][:,0] = (20,8)
        d['positions'][:,0,0] = np.arange(15)
        events = semantic_events(d,self.cfg)
        self.assertGreater(float(events[:,0,24].sum()),3.)


if __name__=='__main__':
    unittest.main()
