"""Team-synchronized review epochs; tactical rerouting never replaces intent."""
from collections import Counter

import numpy as np

from agents.strategic.environment import GoalLevelEnv, lifecycle_events
from agents.strategic.features import entropy_map
from .memory import ReviewMemory


class ReviewGoalEnv(GoalLevelEnv):
    def reset(self,seed,side=0):
        super().reset(seed,side)
        self.manager=ReviewMemory(self.config['review'],self.env_cfg)
        self.review_pending=self.manager.observe(self.frame)
        self.review_trace=[]
        self.completion_seen=np.ones(16,bool)
        self.navigational_intents=self.intent_completions=self.assigned_at_goal=0
        self.previous_comparisons=self.goal_reuse=self.intent_reuse=0
        return self.packet()

    def step(self,replan,modes,goals):
        if self.frame['alive'].any() and not self.review_pending:
            raise ValueError('Actor may act only at an opened macro review epoch')
        start_alive=self.frame['alive'].copy()
        old_valid=self.manager.valid.copy()
        replan=np.asarray(replan,bool)
        if (start_alive & ~old_valid & ~replan).any():
            raise ValueError('invalid previous goal requires deterministic Replan; trace/logprob must match')
        if start_alive.any():
            comparable=start_alive & old_valid
            previous_goals=self.manager.goals.copy();previous_modes=self.manager.modes.copy()
            self.manager.assign_choices(replan,modes,goals,start_alive)
            same_goal=(previous_goals==self.manager.goals).all(1)
            self.previous_comparisons+=int(comparable.sum())
            self.goal_reuse+=int((comparable & same_goal).sum())
            self.intent_reuse+=int((comparable & same_goal & (previous_modes==self.manager.modes)).sum())
            for u in np.flatnonzero(start_alive & replan):
                at_goal=np.array_equal(self.frame['positions'][u],self.manager.goals[u])
                self.completion_seen[u]=at_goal
                self.assigned_at_goal+=at_goal
                self.navigational_intents+=not at_goal
        self.mode_counts+=np.bincount(self.manager.modes[start_alive],minlength=5)
        total=0.;components=Counter();start=self.frame['step']
        for k in range(self.config['review']['max_review_horizon']):
            actions,_=self.tactical.act(self.frame,self.manager)
            before=self.frame
            old_points=np.asarray(self.obs['team_points']);old_wins=np.asarray(self.obs['team_wins'])
            response=self.bridge.request(op='step',actions=actions.tolist())
            self.obs=response['obs'];self.frame=self.builder.observe(self.obs)
            delta=np.asarray(self.obs['team_wins'])-old_wins
            win=float(delta[self.side]-delta[1-self.side])
            points=0.
            if self.frame['match_step']>before['match_step']:
                diff=np.asarray(self.obs['team_points'])-old_points
                points=float(diff[self.side]-diff[1-self.side])
            reset,reused,lost,deaths=lifecycle_events(before,self.frame)
            for u in np.flatnonzero(deaths):self.tactical.stats['retreat_failure']+=self.tactical.states[u]==3
            cleared=lost|reused
            if reset:cleared[:]=True
            self.tactical.states[cleared]=0;self.tactical.state_age[cleared]=0
            raw={'win':win,'points':points,'death':-int(deaths.sum()),
                 'information':max(0.,float(entropy_map(before['base'][11]).sum()-entropy_map(self.frame['base'][11]).sum())),
                 'energy_waste':-int((actions[:,0]==5).sum())*self.env_cfg['unit_sap_cost']/400}
            rl=self.config['rl']
            reward=(rl['win_reward']*win+rl['point_shaping']*points/(self.env_cfg['max_units']*self.env_cfg['max_steps_in_match'])
                    +rl['information_shaping']*raw['information']+rl['death_shaping']*raw['death']
                    +rl['energy_waste_shaping']*raw['energy_waste'])
            total+=rl['gamma']**k*reward;components.update(raw)
            self.metrics['undiscounted_reward']+=reward
            for name,value in response['metrics'].items():
                if name=='engine_score_gain':self.metrics['score']+=value[0];self.metrics['opponent_score']+=value[1]
                elif name=='alive_units':self.metrics['alive_unit_steps']+=value
                else:self.metrics[name]+=value
            self.primitive_steps+=1
            self.review_pending=self.manager.observe(self.frame)
            for u in np.flatnonzero(self.manager.valid & ~self.completion_seen):
                if np.array_equal(self.frame['positions'][u],self.manager.goals[u]):
                    self.intent_completions+=1;self.completion_seen[u]=True
            self.goal_age_sum+=int(self.manager.ages[self.manager.valid].sum())
            self.goal_age_count+=int(self.manager.valid.sum())
            self.done=bool(response['done'])
            if self.done or self.review_pending:break
        tau=k+1;self.taus.append(tau)
        trace={'start':start,'end':self.frame['step'],'tau':tau,'reward':total,'terminal':self.done,
               'replan':np.asarray(replan,bool).tolist(),'mode':np.asarray(modes).tolist(),'goal':np.asarray(goals).tolist(),
               'start_alive':start_alive.tolist(),'previous_valid':old_valid.tolist(),
               'review_events':self.manager.reasons,'components':dict(components)}
        self.review_trace.append(trace)
        return self.packet(),total,self.done,{'tau':tau,'components':dict(components),'events':self.manager.reasons}

    def summary(self):
        m=self.manager.stats;n=max(1,m['decisions']);taus=np.asarray(self.taus)
        wins=np.asarray(self.obs['team_wins'])
        return {'completed':self.done,'primitive_steps':self.primitive_steps,'macro_decisions':len(self.taus),
                'review_frequency':m['macro_reviews']/max(1,self.primitive_steps),
                'replan_rate':m['replans']/n,'continue_rate':m['continues']/n,
                'goal_switch_rate':m['goal_switches']/max(1,self.previous_comparisons),
                'mode_switch_rate':m['mode_switches']/max(1,self.previous_comparisons),
                'goal_persistence':self.goal_reuse/max(1,self.previous_comparisons),
                'intent_persistence':self.intent_reuse/max(1,self.previous_comparisons),
                'previous_intent_comparisons':self.previous_comparisons,
                'switch_persistence_scope':'identity comparison among alive units with a valid previous intent, independent of r',
                'invalid_goal_persistence':0.,
                'invalid_persistence_note':'Continue is forbidden by protocol when previous_valid=0; this is not learned competence',
                'goal_completion_events':self.intent_completions,'goal_completion_rate':self.intent_completions/max(1,self.navigational_intents),
                'assigned_at_goal':int(self.assigned_at_goal),'navigational_intents':int(self.navigational_intents),
                'completion_definition':'first arrival per issued nontrivial intent, excludes goals already occupied at assignment',
                'mean_goal_age':self.goal_age_sum/max(1,self.goal_age_count),
                'option_duration':{'mean':float(taus.mean()) if len(taus) else 0.,
                    **{str(q):float(np.quantile(taus,q)) if len(taus) else 0. for q in (.5,.9,.95)}},
                'mode_distribution':self.mode_counts.tolist(),'memory_events':dict(m),
                'metrics':dict(self.metrics),'full_game_win':bool(wins[self.side]>wins[1-self.side]) if self.done else None,
                'replan_precision':None,'replan_recall':None,'replan_f1':None,
                'online_label_note':'no online expert intent ground truth; no invented replan accuracy',
                'tactical':{k:(v.item() if isinstance(v,np.generic) else v) for k,v in self.tactical.stats.items()}}
