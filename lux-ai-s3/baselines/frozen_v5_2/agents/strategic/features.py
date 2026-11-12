"""Exactly the same causal observation construction in offline and online use."""
from collections import deque

import numpy as np

from agents.v3.dataset import CAUSAL_CURRENT_CHANNELS
from scripts.replay_parser import ReplayReconstructor, _units

MODES = ('SCORE', 'EXPLORE', 'DEFEND', 'INTERCEPT', 'RECHARGE')
UNIT_DIM = 22


def entropy_map(probability):
    p = np.clip(probability, 1e-6, 1-1e-6)
    return -(p*np.log(p)+(1-p)*np.log(1-p))


class CausalObservationBuilder:
    def __init__(self, team_id, env_cfg):
        self.team_id = team_id
        self.env_cfg = env_cfg
        self.reconstructor = ReplayReconstructor(team_id)
        self.history = deque(maxlen=3)
        self.last_step = self.last_match_step = None

    def observe(self, obs):
        step, ms = int(obs['steps']), int(obs['match_steps'])
        if self.last_step is not None and step == self.last_step:
            raise ValueError('observation must be processed exactly once')
        if (self.last_step is None or step != self.last_step+1
                or ms != self.last_match_step+1):
            self.history.clear()
        f, own = self.reconstructor._feature(step, obs)
        r = self.reconstructor
        # Undo legacy's incorrect center-symmetry entity expansion. Keep only
        # actual observations and causal equation conclusions in the new data.
        f[9] = np.where(r.energy_seen >= 0, r.energy_field/20, 0)
        f[10] = 0
        for x,y in r.known_relics.values():
            f[10,max(0,x-2):min(24,x+3),max(0,y-2):min(24,y+3)] = 1
        f[11] = r.solver.feature_map()
        f[12] = 0
        for x,y in r.solver.values:
            f[12,x,y] = 1
        f[19], f[20] = f[11] == 0, f[11] == 1
        # Match disk precision exactly online; masks remain exact binary values.
        from .data import VALUES
        f[list(VALUES)] = f[list(VALUES)].astype(np.float16).astype(np.float32)
        self.history.append(f[list(CAUSAL_CURRENT_CHANNELS)].copy())
        history = np.zeros((3,18,24,24), dtype=np.float32)
        for lag, frame in enumerate(reversed(self.history)):
            history[lag] = frame
        self.last_step, self.last_match_step = step, ms
        positions = np.full((16,2), -1, dtype=np.int16)
        energy = np.zeros(16, dtype=np.int16)
        alive = np.zeros(16, dtype=bool)
        for uid, xy, e in own:
            positions[uid], energy[uid], alive[uid] = xy, e, True
        if ms==0 and step>0:
            alive[:] = False  # official engine clears old-round units next tick
        return dict(history=history, base=f, positions=positions, energy=energy, alive=alive,
                    enemy=_units(obs, 1-self.team_id, 24,24), step=step, match_step=ms,
                    rank=r.solver.rank, points=np.asarray(obs['team_points']),
                    sensor=np.asarray(obs['sensor_mask'], dtype=bool))


def unit_vectors(frame, goals, modes, ages, goal_valid, cfg):
    """22 physical/context scalars; no unavailable cooldown is invented."""
    out = np.zeros((16,UNIT_DIM), dtype=np.float32)
    for uid in np.flatnonzero(frame['alive']):
        x,y = frame['positions'][uid]
        out[uid,:8] = (x/23,y/23,frame['energy'][uid]/400,1,0,0,
                      frame['match_step']/100,frame['step']/505)
        # cooldown value=0 and cooldown-available flag=0: not in official obs.
        out[uid,8:10] = goals[uid]/23 if goal_valid[uid] else 0
        out[uid,10] = min(ages[uid]/100,1)
        out[uid,11] = goal_valid[uid]
        if goal_valid[uid]:
            out[uid,12+int(modes[uid])] = 1
        nearby = [e for e in frame['enemy'] if abs(e[1][0]-x)+abs(e[1][1]-y)<=6]
        out[uid,17] = len(nearby)/16
        out[uid,18] = frame['base'][9,x,y]
        out[uid,19:22] = (cfg['unit_move_cost']/5,cfg['unit_sap_cost']/50,cfg['unit_sap_range']/7)
    return out.astype(np.float16).astype(np.float32)


def causal_context_update(frame, goals, modes, ages, valid, cfg):
    """Offline context policy, computed only from past/current public evidence.

    Expert segment goals are NEVER teacher-forced into previous-goal inputs.
    This context-policy/on-policy mismatch is logged, with IL context dropout.
    """
    gx,gy = np.indices((24,24))
    reserved = set()
    for uid in range(16):
        if not frame['alive'][uid]:
            valid[uid] = False
            continue
        pos = frame['positions'][uid]
        ages[uid] += 1
        if valid[uid] and ages[uid]<cfg['macro_horizon'] and not np.array_equal(pos,goals[uid]):
            reserved.add(tuple(goals[uid]))
            continue
        distance = abs(gx-pos[0])+abs(gy-pos[1])
        if frame['energy'][uid]<cfg['recharge_below'] and (frame['base'][9]>0).any():
            value = frame['base'][9]*5-distance/24
            mode = 4
        elif frame['base'][20].any():
            value = frame['base'][20]*10-distance/24
            mode = 0
        else:
            value = frame['base'][1]*2-distance/24
            mode = 1
        value = value.copy()
        value[frame['base'][5]>.5] -= 100
        for cell in reserved:
            value[cell] -= 2
        goals[uid] = np.unravel_index(value.argmax(), (24,24))
        modes[uid], ages[uid], valid[uid] = mode, 0, True
        reserved.add(tuple(goals[uid]))
