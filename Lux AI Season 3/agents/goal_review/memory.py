"""One causal memory/review protocol for replay contexts and deployment."""
from collections import Counter

import numpy as np

from agents.strategic.features import causal_context_update, entropy_map, unit_vectors


class ReviewMemory:
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

    def observe(self, f, tactical=None):
        """Only <=T evidence; goal assignments always precede the next packet."""
        before = self.last
        reset = before is not None and (f['match_step'] < before['match_step'] or f['step'] != before['step'] + 1)
        reused = np.zeros(16, bool)
        if before is not None:
            reused = before['alive'] & f['alive'] & (
                (np.abs(f['positions'] - before['positions']).sum(1) > 1)
                | (f['energy'] - before['energy'] > 20))
        clear = ~f['alive'] | reused
        if reset:
            clear[:] = True
        self.valid[clear] = False
        self.ages[clear] = 0
        self.reached[clear] = False
        self.blocked[clear] = 0
        self.ages[self.valid] += 1
        self.since_review += 1
        reasons = []
        if reset:
            reasons.append('match_reset')
        if before is None or reused.any() or (before is not None and not np.array_equal(before['alive'], f['alive'])):
            reasons.append('life_change')
        for u in np.flatnonzero(f['alive']):
            if self.valid[u] and self.modes[u] == 0 and f['base'][19, *self.goals[u]] > .5:
                self.valid[u] = False
                reasons.append('v2_goal_invalid')
            if self.valid[u] and self.ages[u] >= self.cfg['max_goal_age']:
                self.valid[u] = False
                reasons.append('goal_memory_expired')
            at_goal = self.valid[u] and np.array_equal(f['positions'][u], self.goals[u])
            if at_goal and not self.reached[u]:
                reasons.append('goal_first_reached')
                self.stats['goal_completions'] += 1
            self.reached[u] = at_goal
            stationary = (before is not None and np.array_equal(f['positions'][u], before['positions'][u]))
            stalled = (self.valid[u] and not at_goal and stationary and self.modes[u] != 4
                       and f['energy'][u] >= self.public['unit_move_cost'])
            self.blocked[u] = self.blocked[u] + 1 if stalled else 0
            if self.blocked[u] == self.cfg['blockage_steps']:
                reasons.append('persistent_blockage')
        severe = bool(tactical and tactical.get('severe_threat'))
        # Replay contexts have the same public-energy threat proxy available.
        if tactical is None:
            for u in np.flatnonzero(f['alive']):
                near = [e for _, p, e in f['enemy'] if np.abs(np.asarray(p) - f['positions'][u]).sum() <= 2]
                severe |= bool(near and sum(near) > self.cfg['threat_ratio'] * max(1, f['energy'][u]))
        self.threat_steps = self.threat_steps + 1 if severe else 0
        if self.threat_steps == self.cfg['severe_threat_steps']:
            reasons.append('sustained_severe_threat')
        if before is not None and not reset:
            delta = np.asarray(f['points']) - np.asarray(before['points'])
            if np.abs(delta).max() >= self.cfg['major_score_delta']:
                reasons.append('major_score')
            information = float(entropy_map(before['base'][11]).sum() - entropy_map(f['base'][11]).sum())
            if information >= self.cfg['major_information'] or f['rank'] - before['rank'] >= self.cfg['major_rank_gain']:
                reasons.append('major_information')
        if self.since_review >= self.cfg['max_review_horizon']:
            reasons.append('max_review_horizon')
        hard = any(r in reasons for r in ('match_reset', 'life_change', 'v2_goal_invalid', 'goal_memory_expired'))
        review = bool(f['alive'].any() and (hard or (reasons and self.since_review >= self.cfg['min_review_steps'])))
        self.reasons = sorted(set(reasons)) if review else []
        self.stats['environment_frames'] += 1
        self.stats['macro_reviews'] += review
        self.stats.update(self.reasons)
        self.last = f
        return review

    def vectors(self, f):
        return unit_vectors(f, self.goals, self.modes, self.ages, self.valid, self.public)

    def assign_choices(self, replan, modes, goals, alive):
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

    def scripted_decision(self, f, tactical_cfg):
        goals, modes = self.goals.copy(), self.modes.copy()
        causal_context_update(f, goals, modes, np.zeros(16, np.int16), np.zeros(16, bool), tactical_cfg)
        replan = ~self.valid | (modes != self.modes) | (goals != self.goals).any(1)
        self.assign_choices(replan, modes, goals[:, 0] * 24 + goals[:, 1], f['alive'])


def frame_at(d, t):
    ids = np.flatnonzero(d['enemy_active'][t])
    return {'base': d['features'][t], 'positions': d['positions'][t], 'energy': d['energies'][t],
            'alive': d['active'][t], 'enemy': [(int(u), tuple(d['enemy_positions'][t, u]), int(d['enemy_energies'][t, u])) for u in ids],
            'step': int(d['steps'][t]), 'match_step': int(d['match_steps'][t]),
            'points': d['team_points'][t], 'rank': int(d['rank'][t])}


def causal_contexts(d, cfg):
    memory = ReviewMemory(cfg['review'], d['env_cfg'])
    vectors, reviews = [], []
    for t in range(len(d['steps'])):
        f = frame_at(d, t)
        review = memory.observe(f)
        vectors.append(memory.vectors(f).astype(np.float16))
        reviews.append(review)
        if review:
            memory.scripted_decision(f, cfg['tactical'])
    return np.asarray(vectors), np.asarray(reviews), dict(memory.stats)
