"""Runtime intent retention extracted from the frozen review stack."""
from collections import Counter

import numpy as np

from .features import entropy_map, unit_vectors


class IntentMemory:
    def __init__(self, cfg, public):
        self.cfg, self.public = cfg, public
        self.goals = np.zeros((16, 2), np.int16)
        self.modes = np.zeros(16, np.int8)
        self.ages = np.zeros(16, np.int16)
        self.valid = np.zeros(16, bool)
        self.reached = np.zeros(16, bool)
        self.blocked = np.zeros(16, np.int16)
        self.threat_steps = 0
        self.since_review = 0
        self.last = None
        self.reasons = []
        self.stats = Counter()
    def vectors(self, f):
        return unit_vectors(f, self.goals, self.modes, self.ages, self.valid, self.public)

    def _assign_choices(self, replan, modes, goals, alive):
        replan = np.asarray(replan, bool) | ~self.valid
        for u in np.flatnonzero(alive):
            self.stats['decisions'] += 1
            if replan[u]:
                xy = np.array([int(goals[u]) // 24, int(goals[u]) % 24])
                self.stats['goal_switches'] += bool(self.valid[u] and not np.array_equal(xy, self.goals[u]))
                self.stats['mode_switches'] += bool(self.valid[u] and self.modes[u] != modes[u])
                self.stats['replans'] += 1
                self.goals[u], self.modes[u], self.ages[u], self.valid[u] = xy, modes[u], 0, True
                self.reached[u] = False
                self.blocked[u] = 0
            else:
                self.stats['continues'] += 1
        self.since_review = 0

    def assign_choices(self,replan,modes,goals,alive):
        old_goals,old_modes=self.goals.copy(),self.modes.copy()
        old_valid,old_age=self.valid.copy(),self.ages.copy()
        old_reached,old_blocked=self.reached.copy(),self.blocked.copy()
        self._assign_choices(replan,modes,goals,alive)
        same=(np.asarray(alive,bool)&old_valid&(old_modes==self.modes)&(old_goals==self.goals).all(1))
        renew=same&np.asarray(replan,bool)
        self.ages[same]=old_age[same]
        self.reached[same]=old_reached[same]
        self.blocked[same]=old_blocked[same]
        self.stats['identical_replan_renewals']+=int(renew.sum())


class TeamMemory(IntentMemory):
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
        self.invalidation = [None] * 16
        for u in np.flatnonzero(reused):
            self.invalidation[u] = "uid_reused"
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
                goal=tuple(self.goals[u])
                bad = []
                if confirmed_blocked[goal]: bad.append("blocked")
                if self.modes[u] == 0 and f["base"][19,*goal] > .5: bad.append("zero_score")
                if self.ages[u] >= self.cfg["max_goal_age"]: bad.append("age_limit")
                # Recompute reachability on the 24x24 public map only.
                if not bad and self.since_review>=self.cfg['min_review_steps']:
                    if not reachable(confirmed_blocked,tuple(f['positions'][u]),goal):
                        bad.append('unreachable')
                if bad:
                    self.invalidation[u] = '+'.join(bad)
                    self.valid[u] = False
                    reasons.append('public_goal_invalid')
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


