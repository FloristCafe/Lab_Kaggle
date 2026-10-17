import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.goal_review.config import load_config
from agents.mappo_v5.environment import TeamMemory,end_flags,official_reward,reachable,discounted_macro_reward
from agents.mappo_v5.learning import TeamCritic,Transition,returns,update,save_rollout,load_rollout
from agents.mappo_v5.policy import sample,statistics,optimizer
from agents.strategic.features import CausalObservationBuilder


def batch(B=1):
    units=torch.zeros(B,16,22);units[...,11]=1;units[...,13]=1;units[...,9]=torch.arange(16)/23
    return {'history':torch.zeros(B,3,18,24,24),'positions':torch.zeros(B,16,2,dtype=torch.long),
        'units':units,'alive':torch.ones(B,16,dtype=torch.bool),'review':torch.ones(B,16,dtype=torch.bool),
        'gate_replan':torch.ones(B,16,dtype=torch.bool),'legal_goals':torch.ones(B,576,dtype=torch.bool),
        'critic_maps':torch.zeros(B,12,24,24),'critic_units':torch.zeros(B,16,30),'team':torch.zeros(B,12)}


class ToyActor(nn.Module):
    def __init__(self,r=0.):
        super().__init__();self.replan_head=nn.Linear(1,1)
        with torch.no_grad():self.replan_head.weight.zero_();self.replan_head.bias.fill_(r)
        self.mode=nn.Parameter(torch.zeros(5));self.goal=nn.Parameter(torch.zeros(5,24,24))

    def forward(self,history,positions,units,alive):
        B,K=alive.shape
        return self.replan_head(units[...,:1]).squeeze(-1),self.mode.expand(B,K,-1),self.goal.expand(B,K,-1,-1,-1)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        self.cfg=load_config(ROOT/'configs/mappo_v5/E2_RMG_pilot.yaml')

    def test_continue_never_draws_mode_goal(self):
        a,_=sample(ToyActor(-100),batch(),self.cfg)
        self.assertEqual(a['draw_counts'],{'R':16,'Mode':0,'Goal':0})
        self.assertFalse(a['r'].any());self.assertFalse(a['branch_mask'][...,1:].any())
        torch.testing.assert_close(a['goal'],torch.arange(16)[None])

    def test_forced_actions_have_no_fake_policy_terms(self):
        b=batch();b['units'][...,11]=0
        actor=ToyActor();a,_=sample(actor,b,self.cfg)
        self.assertTrue(a['forced_action_mask'].all());self.assertFalse(a['policy_mask'].any())
        self.assertEqual(a['draw_counts'],{'R':0,'Mode':0,'Goal':0})
        self.assertEqual(float(a['old_logprob'].abs().max()),0.)
        self.assertEqual(a['goal'].unique().numel(),16)

    def test_saved_masks_reconstruct_actual_ar_probability(self):
        actor=ToyActor(100);b=batch();a,_=sample(actor,b,self.cfg)
        new,_,_=statistics(actor(b['history'],b['positions'],b['units'],b['alive']),b['units'],a)
        torch.testing.assert_close(new,a['old_logprob'])
        torch.testing.assert_close(a['old_logprob_factors'].sum(-1),a['old_logprob'])
        self.assertEqual(a['draw_counts'],{'R':16,'Mode':16,'Goal':16})
        self.assertEqual(a['goal'].unique().numel(),16)
        for u in range(16):
            m,g=int(a['mode'][0,u]),int(a['goal'][0,u]);support=a['goal_mask'][0,u,m]
            self.assertTrue(support[g]);self.assertAlmostEqual(float(new[0,u].detach()),-np.log(5)-np.log(int(support.sum())),places=5)
            for v in range(u):self.assertFalse(support[a['goal'][0,v]])

    def test_nonreview_units_keep_intent_without_policy_loss(self):
        b=batch();b['review'][:,1:]=False;a,_=sample(ToyActor(100),b,self.cfg)
        self.assertFalse(a['policy_mask'][:,1:].any());torch.testing.assert_close(a['goal'][:,1:],torch.arange(1,16)[None])

    def test_continue_has_zero_mode_goal_gradient(self):
        actor=ToyActor(-100);b=batch();a,_=sample(actor,b,self.cfg)
        actor.replan_head.bias.data.zero_()
        lp,h,_=statistics(actor(b['history'],b['positions'],b['units'],b['alive']),b['units'],a)
        (-lp.mean()-.01*h.mean()).backward()
        self.assertGreater(float(actor.replan_head.bias.grad.abs().sum()),0)
        self.assertEqual(float(actor.mode.grad.abs().sum()),0.)
        self.assertEqual(float(actor.goal.grad.abs().sum()),0.)

    def test_gate_has_only_actual_mode_goal_branches(self):
        cfg=copy.deepcopy(self.cfg);cfg['policy']['gate']='deterministic'
        b=batch();b['gate_replan'][:,::2]=False;a,_=sample(ToyActor(),b,cfg)
        self.assertFalse(a['branch_mask'][...,0].any());self.assertEqual(a['draw_counts']['R'],0)
        self.assertEqual(a['draw_counts']['Mode'],8);self.assertFalse(a['branch_mask'][:,::2].any())

    def test_action_support_is_hard_and_persisted(self):
        b=batch();b['legal_goals'][:,:100]=False;a,_=sample(ToyActor(100),b,self.cfg)
        self.assertTrue((a['goal']>=100).all());self.assertFalse(a['goal_mask'][...,:100].any())

    def test_config_independence_and_parameter_groups(self):
        a=load_config(ROOT/'configs/mappo_v5/E1_GM_pilot.yaml');b=load_config(ROOT/'configs/mappo_v5/E2_RMG_pilot.yaml')
        a['policy']['gate']='changed';self.assertEqual(b['policy']['gate'],'learned')
        actor=ToyActor();critic=TeamCritic();opt=optimizer(actor,critic,b)
        ids=[id(p) for g in opt.param_groups for p in g['params']];self.assertEqual(len(ids),len(set(ids)))


class TimeTests(unittest.TestCase):
    def test_nonzero_macro_rewards_use_primitive_offsets(self):
        self.assertAlmostEqual(discounted_macro_reward([1,2,3],.5),2.75)
        self.assertAlmostEqual(discounted_macro_reward([0,0,-3],.99),-3*.99**2)

    def test_terminal_truncation_and_official_reward(self):
        r={'engine_terminal':False,'engine_truncated':True,'task_complete':True,'official_reward':[3,2]}
        self.assertEqual(end_flags(r),(True,False));self.assertEqual(official_reward(r,'terminal_official_win_margin'),1.)
        r['task_complete']=False
        self.assertEqual(end_flags(r),(False,True));self.assertEqual(official_reward(r,'terminal_official_win_margin'),0.)
        r['engine_truncated']=False;self.assertEqual(end_flags(r,True),(False,True))

    def test_duration_gae_and_bootstrap(self):
        b=batch();a,_=sample(ToyActor(),b,load_config(ROOT/'configs/mappo_v5/E2_RMG_pilot.yaml'))
        ts=[Transition(b,b,a,.1,3,.4,.3,False,False,0),Transition(b,b,a,.2,7,.3,.8,False,True,0),
            Transition(b,b,a,1,1,.2,0,True,False,1)]
        adv,ret=returns(ts,.99,.95)
        d1=.2+.99**7*.8-.3;d0=.1+.99**3*.3-.4
        np.testing.assert_allclose(adv,[d0+(.99*.95)**3*d1,d1,.8])
        np.testing.assert_allclose(ret,adv+[.4,.3,.2])

    def test_buffer_owns_behavior_snapshot(self):
        b=batch();a,_=sample(ToyActor(),b,load_config(ROOT/'configs/mappo_v5/E2_RMG_pilot.yaml'))
        t=Transition(b,b,a,0,2,0,0,False,False,0);old=t.action['old_logprob'].clone()
        a['old_logprob'].fill_(100);b['units'].fill_(5)
        torch.testing.assert_close(old,t.action['old_logprob']);self.assertNotEqual(float(t.state['units'].mean()),5.)

    def test_compressed_rollout_is_lossless_and_no_pickle(self):
        b=batch();a,_=sample(ToyActor(),b,load_config(ROOT/'configs/mappo_v5/E2_RMG_pilot.yaml'))
        t=Transition(b,b,a,.5,2,0,0,False,True,0,primitive_rewards=(0.,.5/.99))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'rollout.npz';save_rollout(path,[t],.99);d=load_rollout(path)
            np.testing.assert_array_equal(d['state__history'],t.state['history'].numpy())
            np.testing.assert_array_equal(d['action__goal_mask'],t.action['goal_mask'].numpy())
            np.testing.assert_array_equal(d['action__old_logprob'],t.action['old_logprob'].numpy())
            with self.assertRaises(ValueError):save_rollout(path,[t],.99)

    def test_greedy_density_is_rejected_by_ppo(self):
        cfg=load_config(ROOT/'configs/mappo_v5/E2_RMG_pilot.yaml');actor=ToyActor();critic=TeamCritic()
        b=batch();a,_=sample(actor,b,cfg,greedy=True)
        with self.assertRaisesRegex(ValueError,'greedy'):
            update(actor,critic,optimizer(actor,critic,cfg),[Transition(b,b,a,0,1,0,0,False,True,0)],cfg)

    def test_public_reachability_is_conservative(self):
        obstacles=np.zeros((24,24),bool);self.assertTrue(reachable(obstacles,(0,0),(23,23)))
        obstacles[12,:]=True;self.assertFalse(reachable(obstacles,(0,0),(23,23)))

    def test_stale_asteroid_belief_is_not_a_hard_unreachability_proof(self):
        cfg=load_config(ROOT/'configs/mappo_v5/E2_RMG_pilot.yaml');m=TeamMemory(cfg['review'],{'unit_move_cost':1})
        alive=np.zeros(16,bool);alive[0]=True;m.valid[0]=True;m.goals[0]=23;m.modes[0]=3
        base=np.zeros((21,24,24));base[5,23,23]=1
        f={'alive':alive,'positions':np.zeros((16,2),int),'energy':np.full(16,100),
            'base':base,'sensor':np.zeros((24,24),bool),'enemy':[],'step':5,'match_step':5,'points':np.zeros(2),'rank':0}
        m.observe(f);self.assertTrue(m.valid[0])
        f=copy.deepcopy(f);f.update(step=6,match_step=6);f['sensor'][23,23]=True
        m.observe(f);self.assertFalse(m.valid[0])

    def test_other_unit_reviews_do_not_starve_periodic_review(self):
        cfg=load_config(ROOT/'configs/mappo_v5/E2_RMG_pilot.yaml');m=TeamMemory(cfg['review'],{'unit_move_cost':1})
        alive=np.zeros(16,bool);alive[:2]=True
        f={'alive':alive,'positions':np.zeros((16,2),int),'energy':np.full(16,100),
            'base':np.zeros((21,24,24)),'enemy':[],'step':5,'match_step':5,'points':np.zeros(2),'rank':0}
        m.last=copy.deepcopy(f);m.valid[:2]=True;m.goals[:2]=23;m.modes[:2]=3;m.review_age[:2]=[7,0]
        selected=np.zeros(16,bool);selected[1]=True
        m.assign_review(np.zeros(16,bool),np.zeros(16,int),np.zeros(16,int),selected)
        f.update(step=6,match_step=6);f['energy'][:]=150
        m.observe(f)
        self.assertTrue(m.valid[0]);self.assertTrue(m.review_mask[0]);self.assertEqual(m.review_age[0],8)

    def test_goal_age_continue_and_match_reset(self):
        cfg=load_config(ROOT/'configs/mappo_v5/E2_RMG_pilot.yaml')
        memory=TeamMemory(cfg['review'],{'unit_move_cost':1})
        alive=np.zeros(16,bool);alive[0]=True
        f={'alive':alive,'positions':np.zeros((16,2),int),'energy':np.full(16,100),
            'base':np.zeros((21,24,24)),'enemy':[],'step':99,'match_step':99,'points':np.zeros(2),'rank':0}
        memory.observe(f);memory.assign_review(alive,np.full(16,3),np.full(16,575),alive)
        memory.ages[0]=20;memory.blocked[0]=3
        memory.assign_review(np.zeros(16,bool),np.full(16,1),np.zeros(16),alive)
        self.assertEqual(memory.ages[0],20);self.assertEqual(memory.blocked[0],3);self.assertEqual(memory.modes[0],3)
        f=copy.deepcopy(f);f['step']=100;f['match_step']=0;f['alive'][:]=False;memory.observe(f)
        self.assertFalse(memory.valid.any());self.assertFalse(memory.ages.any())

    def test_nonempty_public_map_memory_survives_round_reset(self):
        cfg={'map_width':24,'map_height':24,'max_units':16,'max_steps_in_match':100,
             'unit_sensor_range':2,'unit_move_cost':1,'unit_sap_cost':30,'unit_sap_range':3}
        builder=CausalObservationBuilder(0,cfg)
        obs={'steps':100,'match_steps':100,'team_points':[0,0],'team_wins':[0,0],
            'units':{'position':np.full((2,16,2),-1).tolist(),'energy':np.full((2,16),-1).tolist()},
            'units_mask':np.zeros((2,16),bool).tolist(),'sensor_mask':np.ones((24,24),bool).tolist(),
            'map_features':{'tile_type':np.zeros((24,24),int).tolist(),'energy':np.ones((24,24),int).tolist()},
            'relic_nodes':[[5,5]],'relic_nodes_mask':[True]}
        builder.observe(obs);memory=builder.reconstructor;solver=memory.solver;drift=memory.tile_drift
        obs=copy.deepcopy(obs);obs.update(steps=101,match_steps=0,relic_nodes=[[-1,-1]],relic_nodes_mask=[False])
        obs['sensor_mask']=np.zeros((24,24),bool).tolist()
        f=builder.observe(obs)
        self.assertIs(memory,builder.reconstructor);self.assertIs(solver,memory.solver);self.assertIs(drift,memory.tile_drift)
        self.assertEqual(memory.known_relics[0],(5,5));self.assertEqual(memory.energy_seen[5,5],100)
        self.assertTrue((f['history'][1:]==0).all())

    def test_critic_empty_team_finite_and_update(self):
        torch.set_num_threads(2);cfg=load_config(ROOT/'configs/mappo_v5/E2_RMG_pilot.yaml')
        actor=ToyActor();critic=TeamCritic();b=batch(2);b['alive'][:]=False
        self.assertTrue(torch.isfinite(critic(b)).all());self.assertEqual(critic(b).shape,(2,))
        b=batch();a,_=sample(actor,b,cfg);opt=optimizer(actor,critic,cfg)
        buf=[Transition(b,b,a,1,3,0,.2,False,True,0)]
        cfg['ppo']['epochs']=1;metrics=update(actor,critic,opt,buf,cfg)
        self.assertLess(metrics['first_logprob_error'],1e-4);self.assertTrue(np.isfinite(metrics['value_loss']))

    def test_continue_does_not_step_mode_goal_adam_momentum(self):
        torch.set_num_threads(2);cfg=load_config(ROOT/'configs/mappo_v5/E2_RMG_pilot.yaml');cfg['ppo']['epochs']=1
        actor=ToyActor(100);critic=TeamCritic();opt=optimizer(actor,critic,cfg);b=batch()
        a,_=sample(actor,b,cfg)
        update(actor,critic,opt,[Transition(b,b,a,1,3,0,0,False,True,0)],cfg)
        actor.replan_head.bias.data.fill_(-100);a,_=sample(actor,b,cfg)
        mode,goal=actor.mode.detach().clone(),actor.goal.detach().clone()
        update(actor,critic,opt,[Transition(b,b,a,-1,3,0,0,False,True,0)],cfg)
        torch.testing.assert_close(actor.mode,mode,rtol=0,atol=0)
        torch.testing.assert_close(actor.goal,goal,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
