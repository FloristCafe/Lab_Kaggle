"""Public-observation team SMDP. Finite five-match game is a task terminal."""
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from agents.goal_review.runtime import RuntimeReviewMemory
from agents.strategic.control import TacticalController
from agents.strategic.experiment import ROOT
from agents.strategic.features import CausalObservationBuilder,entropy_map


class TeamMemory(RuntimeReviewMemory):
    """Per-unit review set; conservative hard invalidation from public evidence.

    Energy recharge alone is NOT evidence of unit respawn. Unknown terrain is
    passable for reachability, so a failed path only certifies known enclosure.
    """
    def __init__(self,cfg,public):
        super().__init__(cfg,public)
        self.review_age=np.zeros(16,np.int16)
        self.threat_age=np.zeros(16,np.int16)

    def observe(self,f,tactical=None):
        old=self.last;reset=old is not None and (f['match_step']<old['match_step'] or f['step']!=old['step']+1)
        reused=np.zeros(16,bool)
        if old is not None:reused=old['alive']&f['alive']&(np.abs(old['positions']-f['positions']).sum(1)>1)
        clear=~f['alive']|reused
        if reset:clear[:]=True
        self.valid[clear]=False;self.ages[clear]=0;self.reached[clear]=False;self.blocked[clear]=0
        self.review_age[clear]=0;self.threat_age[clear]=0
        self.review_age[f['alive']]+=1
        self.ages[self.valid]+=1;self.since_review+=1
        self.review_mask=np.zeros(16,bool);self.gate_replan=np.zeros(16,bool);reasons=[]
        confirmed_blocked=(f['base'][5]>.5)&np.asarray(f.get('sensor',np.zeros((24,24),bool)))
        for u in np.flatnonzero(f['alive']):
            if self.valid[u]:
                goal=tuple(self.goals[u]);bad=confirmed_blocked[goal]
                bad|=self.modes[u]==0 and f['base'][19,*goal]>.5
                bad|=self.ages[u]>=self.cfg['max_goal_age']
                # Recompute reachability on the 24x24 public map only.
                if not bad and self.since_review>=self.cfg['min_review_steps']:
                    bad=not reachable(confirmed_blocked,tuple(f['positions'][u]),goal)
                if bad:self.valid[u]=False;reasons.append('public_goal_invalid')
            at=self.valid[u] and np.array_equal(self.goals[u],f['positions'][u])
            first=at and not self.reached[u]
            if first:self.stats['goal_first_reached']+=1
            self.reached[u]=at
            stopped=old is not None and np.array_equal(f['positions'][u],old['positions'][u])
            blocked=self.valid[u] and not at and stopped and self.modes[u]!=4 and f['energy'][u]>=self.public['unit_move_cost']
            self.blocked[u]=self.blocked[u]+1 if blocked else 0
            stalled=self.blocked[u]>=self.cfg['blockage_steps']
            severe=any(np.abs(np.asarray(p)-f['positions'][u]).sum()<=2 and e>2*max(1,f['energy'][u]) for _,p,e in f['enemy'])
            self.threat_age[u]=self.threat_age[u]+1 if severe else 0
            soft=first or stalled or self.threat_age[u]>=self.cfg['severe_threat_steps']
            hard=not self.valid[u]
            due=self.review_age[u]>=self.cfg['max_review_horizon']
            self.review_mask[u]=hard or due or (soft and self.review_age[u]>=self.cfg['min_review_steps'])
            self.gate_replan[u]=hard or at or stalled or self.ages[u]>=self.cfg['gate_replan_age']
        global_review=self.since_review>=self.cfg['max_review_horizon']
        if old is not None and not reset:
            global_review|=np.abs(np.asarray(f['points'])-np.asarray(old['points'])).max()>=self.cfg['major_score_delta']
            global_review|=f['rank']-old['rank']>=self.cfg['major_rank_gain']
            global_review|=float(entropy_map(old['base'][11]).sum()-entropy_map(f['base'][11]).sum())>=self.cfg['major_information']
        if global_review:self.review_mask|=f['alive'];reasons.append('global_review')
        if reset:reasons.append('match_reset')
        review=bool(self.review_mask.any())
        if review:reasons.append('unit_review')
        self.reasons=reasons;self.stats['environment_frames']+=1;self.stats['macro_reviews']+=review
        self.stats.update(reasons);self.last=f
        return review

    def assign_review(self,replan,modes,goals,review):
        # All other live intents remain byte-for-byte untouched.
        self.assign_choices(replan,modes,goals,np.asarray(review,bool))
        self.review_age[np.asarray(review,bool)]=0


def reachable(blocked,start,goal):
    if blocked[goal]:return False
    visited={start};stack=[start]
    while stack:
        x,y=stack.pop()
        if (x,y)==goal:return True
        for nx,ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
            if 0<=nx<24 and 0<=ny<24 and not blocked[nx,ny] and (nx,ny) not in visited:
                visited.add((nx,ny));stack.append((nx,ny))
    return False


class Bridge:
    def __init__(self,path):
        self.log=Path(path).open('a')
        self.process=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/mappo_v5_worker.py')],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,cwd=ROOT,
            env=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES=''))

    def request(self,**kw):
        self.process.stdin.write(json.dumps(kw)+'\n');self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('official worker exited; inspect worker.log')
        return json.loads(line)

    def close(self):
        if self.process.poll() is None:
            try:self.request_close()
            except (BrokenPipeError,TimeoutError):self.process.terminate()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.log.close()

    def request_close(self):
        self.process.stdin.write('{"op":"close"}\n');self.process.stdin.flush();self.process.stdin.close()


def end_flags(response,external_limit=False):
    # Official is_terminal always False; its 505-step truncation IS the finite
    # game's terminal. Artificial shorter limits bootstrap, and stop GAE links.
    terminal=bool(response['engine_terminal'] or response['task_complete'])
    truncated=bool((response['engine_truncated'] or external_limit) and not terminal)
    return terminal,truncated


def official_reward(response,mode):
    if mode!='terminal_official_win_margin':raise ValueError('unregistered reward transform')
    return float(response['official_reward'][0]-response['official_reward'][1]) if response['task_complete'] or response['engine_terminal'] else 0.


def discounted_macro_reward(rewards,gamma):
    return sum(gamma**j*float(reward) for j,reward in enumerate(rewards))


class TeamEnv:
    def __init__(self,config,log_path,bridge=None):
        self.config=config;self.bridge=bridge or Bridge(log_path)

    def reset(self,seed,side=0,opponent='v2'):
        response=self.bridge.request(op='reset',seed=seed,side=side,opponent=opponent)
        self.seed,self.side,self.opponent=seed,side,opponent
        self.env_cfg=response['env_cfg'];self.obs=response['obs']
        self.builder=CausalObservationBuilder(side,self.env_cfg)
        self.frame=self.builder.observe(self.obs)
        self.manager=TeamMemory(self.config['review'],self.env_cfg)
        self.tactical=TacticalController(self.env_cfg,self.config['tactical'])
        self.manager.observe(self.frame)
        self.terminal=self.truncated=False;self.steps=0;self.taus=[];self.metrics=Counter()
        self.modes=np.zeros(5,int);self.completion_seen=np.ones(16,bool);self.completion=0;self.issued=0
        self.previous_goals=np.zeros((16,2),int);self.previous_modes=np.zeros(16,int)
        return self.packet()

    def packet(self):
        f=self.frame;m=self.manager;units=m.vectors(f);maps=np.zeros((12,24,24),np.float32)
        cu=np.zeros((16,30),np.float32)
        for u in np.flatnonzero(f['alive']):
            x,y=f['positions'][u];state=int(self.tactical.states[u])
            maps[0,x,y]+=1/16;maps[1,x,y]+=f['energy'][u]/6400
            maps[4,x,y]+=not m.valid[u];maps[6+state,x,y]+=1/16
            maps[11,x,y]+=self.completion_seen[u]/16
            if m.valid[u]:
                gx,gy=m.goals[u];maps[2,gx,gy]+=1/16;maps[3,gx,gy]+=m.ages[u]/1600
            cu[u,:22]=units[u];cu[u,22:24]=f['positions'][u]/23;cu[u,24+state]=1
            cu[u,29]=np.abs(f['positions'][u]-m.goals[u]).sum()/46 if m.valid[u] else 0
        maps[5]=np.maximum(0,maps[2]*16-1)/16
        points=np.asarray(self.obs['team_points']);wins=np.asarray(self.obs['team_wins'])
        horizon=(self.env_cfg['max_steps_in_match']+1)*self.env_cfg['match_count_per_episode']
        match_length=self.env_cfg['max_steps_in_match'];matches=self.env_cfg['match_count_per_episode']
        team=np.array([points[self.side]/1000,points[1-self.side]/1000,wins[self.side]/matches,wins[1-self.side]/matches,
            max(0,horizon-f['step'])/horizon,min(f['step']//(match_length+1),matches-1)/max(1,matches-1),
            (match_length-f['match_step'])/match_length,
            f['rank']/576,entropy_map(f['base'][11]).mean(),self.completion/100,f['alive'].sum()/16,m.valid.sum()/16],np.float32)
        values={'history':f['history'],'positions':f['positions'].astype(np.int64),'units':units,'alive':f['alive'],
            'primitive_step':np.asarray(f['step'],np.int64),'match_step':np.asarray(f['match_step'],np.int64),
            'previous_intent':np.column_stack((m.modes,m.goals,m.ages,m.valid,m.blocked,m.reached)).astype(np.int64),
            'review':m.review_mask & f['alive'],'gate_replan':m.gate_replan,
            'legal_goals':~((f['base'][5]>.5)&f['sensor']).flatten(),
            'critic_maps':maps,'critic_units':cu,'team':team}
        return {k:torch.from_numpy(np.asarray(v)[None].copy()) for k,v in values.items()}

    def step(self,action,max_steps=None):
        a={k:(v[0].detach().cpu().numpy() if isinstance(v,torch.Tensor) else v) for k,v in action.items()}
        review=self.manager.review_mask & self.frame['alive']
        if not np.array_equal(a['decision_mask'],review):raise ValueError('stale decision mask')
        old_valid=self.manager.valid.copy();old_goal=self.manager.goals.copy();old_mode=self.manager.modes.copy()
        self.manager.assign_review(a['r'],a['mode'],a['goal'],review)
        comparable=review & old_valid
        self.metrics['intent_comparisons']+=int(comparable.sum())
        same_goal=(old_goal==self.manager.goals).all(1)
        self.metrics['goals_kept']+=int((comparable & same_goal).sum())
        self.metrics['intents_kept']+=int((comparable & same_goal & (old_mode==self.manager.modes)).sum())
        assigned=[tuple(self.manager.goals[u]) for u in np.flatnonzero(self.manager.valid & self.frame['alive'])]
        self.metrics['goal_conflicts']+=len(assigned)-len(set(assigned))
        for u in np.flatnonzero(review):
            same=old_valid[u] and old_mode[u]==self.manager.modes[u] and np.array_equal(old_goal[u],self.manager.goals[u])
            if not same:
                at=np.array_equal(self.frame['positions'][u],self.manager.goals[u]);self.completion_seen[u]=at
                self.issued+=not at
            self.metrics['assignment_distance_sum']+=int(np.abs(self.frame['positions'][u]-self.manager.goals[u]).sum())
            self.metrics['anchor_assignments']+=int(np.array_equal(self.frame['positions'][u],self.manager.goals[u]))
            self.modes[self.manager.modes[u]]+=1
        self.metrics['reviewed_units']+=int(review.sum());self.metrics['forced_actions']+=int(a['forced_action_mask'].sum())
        self.metrics['free_R']+=int(a['branch_mask'][:,0].sum())
        self.metrics['free_continue']+=int((a['branch_mask'][:,0]&~a['r']).sum())
        self.metrics['replans']+=int((review&a['r']).sum())
        raw_rewards=[];match_resets=0
        for k in range(self.config['review']['max_review_horizon']):
            actions,_=self.tactical.act(self.frame,self.manager)
            before=self.frame;response=self.bridge.request(op='step',actions=actions.tolist())
            self.obs=response['obs'];self.frame=self.builder.observe(self.obs);self.steps+=1
            reset=self.frame['match_step']<before['match_step']
            lost=~self.frame['alive']
            if reset:
                lost[:]=True;match_resets+=1;self.completion_seen[:]=True
            self.tactical.states[lost]=0;self.tactical.state_age[lost]=0
            reward=official_reward(response,self.config['reward']);raw_rewards.append(reward)
            opened=self.manager.observe(self.frame)
            comparable=before['alive'] & self.frame['alive']
            if reset:comparable[:]=False
            stopped=(before['positions']==self.frame['positions']).all(1) & comparable
            at_goal=(self.frame['positions']==self.manager.goals).all(1) & self.manager.valid & self.frame['alive']
            self.metrics['alive_unit_steps']+=int(self.frame['alive'].sum())
            self.metrics['stationary_unit_steps']+=int(stopped.sum())
            self.metrics['navigational_stall_steps']+=int((stopped & ~at_goal & (self.frame['energy']>=self.env_cfg['unit_move_cost'])).sum())
            self.metrics['goal_age_sum']+=int(self.manager.ages[self.manager.valid].sum())
            self.metrics['valid_intent_unit_steps']+=int(self.manager.valid.sum())
            for u in np.flatnonzero(self.frame['alive'] & self.manager.valid & ~self.completion_seen):
                if np.array_equal(self.frame['positions'][u],self.manager.goals[u]):
                    self.completion+=1;self.completion_seen[u]=True
            self.terminal,self.truncated=end_flags(response,max_steps is not None and self.steps>=max_steps)
            if opened or self.terminal or self.truncated:break
        tau=k+1;self.taus.append(tau);self.metrics['match_resets']+=match_resets
        info={'tau':tau,'terminal':self.terminal,'truncated':self.truncated,
            'engine_terminal':response['engine_terminal'],'engine_truncated':response['engine_truncated'],
            'task_complete':response['task_complete'],'primitive_rewards':raw_rewards,'match_resets':match_resets}
        return self.packet(),discounted_macro_reward(raw_rewards,self.config['ppo']['gamma']),self.terminal,self.truncated,info

    def summary(self):
        wins=np.asarray(self.obs['team_wins'])
        return {'seed':self.seed,'side':self.side,'opponent':self.opponent,'complete':self.terminal,'truncated':self.truncated,
            'steps':self.steps,'options':len(self.taus),'mean_tau':float(np.mean(self.taus)) if self.taus else 0.,
            'review_frequency':len(self.taus)/max(1,self.steps),'mode_counts':self.modes.tolist(),
            'official_wins':wins.tolist(),'win':bool(wins[self.side]>wins[1-self.side]) if self.terminal else None,
            'game_margin':float(wins[self.side]-wins[1-self.side]),'goal_completion':self.completion,
            'nontrivial_intents':self.issued,'completion_rate':self.completion/max(1,self.issued),
            'free_continue_rate':self.metrics['free_continue']/max(1,self.metrics['free_R']),
            'goal_persistence':self.metrics['goals_kept']/max(1,self.metrics['intent_comparisons']),
            'intent_persistence':self.metrics['intents_kept']/max(1,self.metrics['intent_comparisons']),
            'goal_conflicts':self.metrics['goal_conflicts'],
            'mean_goal_age':self.metrics['goal_age_sum']/max(1,self.metrics['valid_intent_unit_steps']),
            'mean_assignment_distance':self.metrics['assignment_distance_sum']/max(1,self.metrics['reviewed_units']),
            'anchor_assignment_rate':self.metrics['anchor_assignments']/max(1,self.metrics['reviewed_units']),
            'stationary_rate':self.metrics['stationary_unit_steps']/max(1,self.metrics['alive_unit_steps']),
            'tau_quantiles':np.quantile(self.taus,[0,.5,.9,1]).tolist() if self.taus else [],
            'memory':dict(self.manager.stats),'metrics':dict(self.metrics),
            'tactical':{k:(v.item() if isinstance(v,np.generic) else v) for k,v in self.tactical.stats.items()}}

    def close(self):self.bridge.close()
