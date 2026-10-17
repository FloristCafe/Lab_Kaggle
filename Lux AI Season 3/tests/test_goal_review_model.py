import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.goal_review.config import load_config
from agents.goal_review.data import ReviewDataset, UnitEpochSampler, class_weights, corrupt_context
from agents.goal_review.metrics import classification
from agents.strategic.data import MASKS, VALUES
from agents.goal_review.loss import objective
from agents.goal_review.model import ReviewActor,bounded_joint_bias,review_statistics,select_review_actions
from agents.strategic.model import StrategicActor


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);torch.set_num_threads(2)
        self.cfg=load_config(ROOT/'configs/goal_review/common.yaml')
        self.inputs=(torch.randn(2,3,18,24,24),torch.zeros(2,16,2,dtype=torch.long),
                     torch.zeros(2,16,22),torch.ones(2,16,dtype=torch.bool))

    def test_current_architecture_is_bitwise_legacy_without_r(self):
        old=StrategicActor().eval();new=ReviewActor(self.cfg['model']).eval()
        new.load_state_dict(old.state_dict(),strict=True)
        with torch.no_grad():
            ml,gl=old(*self.inputs);_,nm,ng=new(*self.inputs)
        torch.testing.assert_close(ml,nm,rtol=0,atol=0)
        torch.testing.assert_close(gl,ng,rtol=0,atol=0)

    def test_residual_capacity_variants_keep_resolution_and_all_heads(self):
        for arch in ('res6','res8','dilated6'):
            cfg={**self.cfg['model'],'architecture':arch,'replan':True}
            actor=ReviewActor(cfg)
            r,ml,gl=actor(*self.inputs)
            self.assertEqual(r.shape,(2,16));self.assertEqual(gl.shape,(2,16,5,24,24))
            self.assertEqual(ml.shape,(2,16,5))
            (r.mean()+ml.mean()+gl.mean()).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None))

    def test_continue_ignores_new_mode_goal_and_forced_replan_likelihood(self):
        r=torch.full((1,16),-50.);ml=torch.randn(1,16,5);gl=torch.randn(1,16,5,24,24)
        units=torch.zeros(1,16,22);units[:,:,11]=1;units[:,:,8:10]=torch.tensor([7/23,8/23]);units[:,:,14]=1
        alive=torch.ones(1,16,dtype=torch.bool)
        cfg={**self.cfg['model'],'replan':True}
        rr,m,g,lp,_=select_review_actions((r,ml,gl),units,alive,cfg,greedy=True)
        self.assertFalse(rr.any());self.assertTrue((m==2).all());self.assertTrue((g==7*24+8).all())
        torch.testing.assert_close(lp,torch.nn.functional.logsigmoid(-r))
        units[:,:,11]=0
        rr,m,g,lp,_=select_review_actions((r,ml,gl),units,alive,cfg,greedy=True)
        self.assertTrue(rr.all())
        new,_=review_statistics(r,ml,gl,rr,m,g,units,alive,True)
        torch.testing.assert_close(new,lp)
        torch.testing.assert_close((new-lp).exp(),torch.ones_like(lp))

    def test_hysteresis_is_bounded_on_only_one_joint_pair_and_disabled_when_invalid(self):
        ml=torch.zeros(1,1,5);gl=torch.zeros(1,1,5,24,24);units=torch.zeros(1,1,22)
        units[:,:,11]=1;units[:,:,15]=1;units[:,:,8:10]=torch.tensor([3/23,4/23])
        cfg={**self.cfg['model'],'hysteresis':99.,'hysteresis_cap':.5}
        nm,ng=bounded_joint_bias(ml,gl,units,cfg)
        joint=nm.log_softmax(-1)[...,None]+ng.flatten(-2).log_softmax(-1)
        delta=joint[0,0,3,3*24+4]-joint[0,0,2,3*24+4]
        self.assertAlmostEqual(float(delta),.5,places=5)
        units[:,:,11]=0
        nm,ng=bounded_joint_bias(ml,gl,units,cfg)
        torch.testing.assert_close(nm,ml);torch.testing.assert_close(ng,gl)

    def test_continue_supervision_has_zero_new_goal_gradient(self):
        cfg=load_config(ROOT/'configs/goal_review/common.yaml');cfg['model']['replan']=True
        r=torch.zeros(1,16,requires_grad=True);ml=torch.zeros(1,16,5,requires_grad=True)
        gl=torch.zeros(1,16,5,24,24,requires_grad=True)
        valid=torch.ones(1,16,dtype=torch.bool);units=torch.zeros(1,16,22);units[:,:,11]=1
        batch={'valid':valid,'alive':valid,'replan_valid':valid,'replan':torch.zeros(1,16),
               'mode':torch.zeros(1,16,dtype=torch.long),'regions':torch.ones(1,16,24,24,dtype=torch.bool),
               'confidence':torch.ones(1,16),'units':units}
        loss,_=objective((r,ml,gl),batch,cfg);loss.backward()
        self.assertGreater(float(r.grad.abs().sum()),0)
        self.assertEqual(float(ml.grad.abs().sum()),0)
        self.assertEqual(float(gl.grad.abs().sum()),0)

    def test_balanced_sampler_balances_unit_gradients_and_capped_weights(self):
        class D:
            rows=np.column_stack((np.zeros(1000,int),np.arange(1000),np.zeros(1000,int)))
            modes=np.r_[np.zeros(990,int),np.ones(10,int)]
            mode_counts=np.array([990,10,0,0,0])
            def __len__(self):return 1000
        d=D();sampler=UnitEpochSampler(d,42,10000,True)
        counts=np.bincount(d.modes[list(sampler)],minlength=2)
        self.assertLess(abs(counts[0]-counts[1]),400)
        self.assertLessEqual(float(class_weights(d.mode_counts,5).max()),5.)
        units=np.zeros((16,22),np.float32);units[:,3]=1;units[:,11]=1
        np.testing.assert_array_equal(corrupt_context(units,42),corrupt_context(units,42))

    def test_invalid_previous_cannot_receive_continue_behavior_probability(self):
        with self.assertRaisesRegex(ValueError,'forced Replan'):
            review_statistics(torch.zeros(1,16),torch.zeros(1,16,5),torch.zeros(1,16,5,24,24),
                torch.zeros(1,16,dtype=torch.bool),torch.zeros(1,16,dtype=torch.long),
                torch.zeros(1,16,dtype=torch.long),torch.zeros(1,16,22),torch.ones(1,16,dtype=torch.bool),True)

    def test_absent_explore_predictions_are_visible_despite_high_accuracy(self):
        confusion=np.zeros((5,5),int);confusion[3,3]=99;confusion[1,3]=1
        result=classification(confusion)
        self.assertEqual(result['accuracy'],.99)
        self.assertEqual(result['recall'][1],0.)
        self.assertEqual(result['predicted_count'][1],0)
        self.assertLess(result['balanced_accuracy'],.3)

    def test_reader_samples_exactly_one_unit_and_clean_validation_ignores_dirty_view(self):
        n=1;valid=np.zeros((n,16),bool);valid[0,:2]=True
        units=np.zeros((n,16,22),np.float16);units[:,:,3]=1
        arrays=dict(schema_version=np.asarray(5),steps=np.array([0]),match_steps=np.array([0]),
            feature_masks=np.zeros((n,len(MASKS),24,24),np.uint8),
            feature_values=np.zeros((n,len(VALUES),24,24),np.float16),
            label_valid=valid,mode=np.zeros((n,16),np.int8),positions=np.zeros((n,16,2),np.int16),
            active=np.ones((n,16),bool),unit_vectors=units,semantic_valid=valid,
            segment_id=np.zeros((n,16),np.int32),goal_regions=np.ones((1,24,24),np.uint8),
            boundary_replan=valid,bridge_continue=np.zeros_like(valid),label_confidence=np.ones((n,16),np.float16))
        config=load_config(ROOT/'configs/goal_review/common.yaml');config['study']['label_view']='dirty'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fixture.npz';np.savez_compressed(path,**arrays)
            dataset=ReviewDataset([path],config,training=False)
            self.assertEqual(len(dataset),2)
            a,b=dataset[0],dataset[1]
            self.assertEqual(a['history'].shape,(3,18,24,24))
            self.assertEqual(int(a['valid'].sum()),1)
            self.assertTrue(a['sampled'][0]);self.assertTrue(b['sampled'][1])
            self.assertEqual(int((a['sampled']&b['sampled']).sum()),0)


if __name__=='__main__':unittest.main()
