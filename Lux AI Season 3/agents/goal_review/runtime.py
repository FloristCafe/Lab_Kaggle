"""Deployment identity-age guard, compatible with scripted dataset contexts.

Scripted memory replans only when invalid or (mode,goal) changes, so the override
below is never active in the frozen dataset producer. Actor-driven Replan may
choose the SAME identity; preserving its age prevents recurrent renewals from
keeping bounded hysteresis fresh forever. Original producer files stay frozen.
"""
import numpy as np

from .environment import ReviewGoalEnv
from .memory import ReviewMemory


class RuntimeReviewMemory(ReviewMemory):
    def assign_choices(self,replan,modes,goals,alive):
        old_goals,old_modes=self.goals.copy(),self.modes.copy()
        old_valid,old_age=self.valid.copy(),self.ages.copy()
        old_reached,old_blocked=self.reached.copy(),self.blocked.copy()
        super().assign_choices(replan,modes,goals,alive)
        same=(np.asarray(alive,bool)&old_valid&(old_modes==self.modes)&(old_goals==self.goals).all(1))
        renew=same&np.asarray(replan,bool)
        self.ages[same]=old_age[same]
        self.reached[same]=old_reached[same]
        self.blocked[same]=old_blocked[same]
        self.stats['identical_replan_renewals']+=int(renew.sum())


class RuntimeGoalEnv(ReviewGoalEnv):
    def reset(self,seed,side=0):
        super().reset(seed,side)
        # Reuse the already processed initial legal observation; never consume
        # it twice or reset the official environment a second time.
        self.manager=RuntimeReviewMemory(self.config['review'],self.env_cfg)
        self.review_pending=self.manager.observe(self.frame)
        return self.packet()

    def summary(self):
        result=super().summary()
        result['goal_age_definition']='age of unchanged (mode,goal) identity; same-pair Replan does not reset it'
        return result
