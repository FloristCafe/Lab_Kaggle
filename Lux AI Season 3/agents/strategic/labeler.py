"""Event-based, evidence-scored long-horizon pseudo-labels from legal trajectories."""
import numpy as np

from .features import MODES, entropy_map

EVENT_NAMES = ('speed','heading_x','heading_y','heading_change','life_displacement','stationary',
               'energy','energy_delta','depletion','recharge','score_delta','on_true','support',
               'belief_entropy','entropy_reduction','new_visible','enemies','friends','enemy_distance',
               'energy_advantage','contact','sap','death','rank_gain','enemy_approach')


def semantic_events(data, config):
    positions, alive, energies = data['positions'], data['active'], data['energies']
    features = data['features']
    events = np.zeros((*alive.shape,len(EVENT_NAMES)), dtype=np.float32)
    events[:,:,18] = 48
    T = len(alive)
    for uid in range(16):
        last_heading = np.zeros(2)
        start = None
        stationary = 0
        for t in range(T):
            if not alive[t,uid]:
                start = None
                continue
            xy = positions[t,uid]
            continuous = (t>0 and alive[t-1,uid] and data['steps'][t]-data['steps'][t-1]==1
                          and data['match_steps'][t]-data['match_steps'][t-1]==1
                          and np.abs(xy-positions[t-1,uid]).sum()<=1
                          and energies[t,uid]-energies[t-1,uid]<=20)
            if not continuous:
                start, stationary, last_heading = xy.copy(), 0, np.zeros(2)
            velocity = xy-positions[t-1,uid] if continuous else np.zeros(2)
            speed = np.linalg.norm(velocity)
            heading = velocity/max(speed,1)
            angle = np.arccos(np.clip(heading@last_heading,-1,1)) if speed and np.linalg.norm(last_heading) else 0
            if speed:
                last_heading = heading
            stationary = stationary+1 if continuous and speed==0 else 1
            edelta = int(energies[t,uid]-energies[t-1,uid]) if continuous else 0
            radius = config['neighborhood_radius']
            x,y = map(int,xy)
            region = (slice(max(0,x-radius),min(24,x+radius+1)),slice(max(0,y-radius),min(24,y+radius+1)))
            current_H = entropy_map(features[t,11][region]).sum()
            reduction = max(0,entropy_map(features[t-1,11][region]).sum()-current_H) if continuous else 0
            new_visible = np.sum((features[t-1,1][region]>.5)&(features[t,1][region]<.5)) if continuous else 0
            own_dist = np.abs(positions[t]-xy).sum(1)
            friends = int(((own_dist<=radius)&alive[t]).sum())-1
            visible_enemy = data['enemy_active'][t]
            enemies_xy = data['enemy_positions'][t,visible_enemy]
            distances = np.abs(enemies_xy-xy).sum(1)
            nearest = int(distances.min()) if len(distances) else 48
            enemy_n = int((distances<=radius).sum())
            enemy_e = data['enemy_energies'][t,visible_enemy][distances<=radius]
            advantage = energies[t,uid]-float(enemy_e.mean()) if len(enemy_e) else 0
            score = max(0,int(data['team_points'][t,data['team_id']]-data['team_points'][t-1,data['team_id']])) if continuous else 0
            # Attribute approach to OUR movement toward a previously observed
            # enemy, not to an enemy advancing toward a stationary defender.
            approach = 0
            if continuous and speed>0 and data['enemy_active'][t-1].any():
                old_enemies = data['enemy_positions'][t-1,data['enemy_active'][t-1]]
                old_distances = np.abs(old_enemies-positions[t-1,uid]).sum(1)
                previous_enemy = old_enemies[old_distances.argmin()]
                approach = max(0,float(old_distances.min()-np.abs(previous_enemy-xy).sum()))
            death = (t+1<T and data['match_steps'][t+1]>0 and not alive[t+1,uid])
            rank_gain = max(0,int(data['rank'][t]-data['rank'][t-1])) if continuous else 0
            events[t,uid] = (speed,*heading,angle,np.abs(xy-start).sum(),stationary,
                             energies[t,uid],edelta,max(0,-edelta),max(0,edelta),score,
                             features[t,20,x,y],features[t,10,x,y],current_H,reduction,new_visible,
                             enemy_n,friends,nearest,advantage,nearest<=config['engagement_radius'],
                             data['actions'][t,uid,0]==5,death,rank_gain,approach)
    return events


class StrategicGoalLabeler:
    def __init__(self, config):
        self.cfg = config
        self.weights = np.asarray([config['weights'][name] for name in MODES], dtype=np.float32)

    def _lives(self, data, uid):
        alive, pos, energy = data['active'], data['positions'], data['energies']
        start = None
        for t in range(len(alive)+1):
            exists = t<len(alive) and alive[t,uid] and not (data['match_steps'][t]==0 and data['steps'][t]>0)
            continuous = (exists and start is not None and data['steps'][t]==data['steps'][t-1]+1
                          and data['match_steps'][t]==data['match_steps'][t-1]+1
                          and np.abs(pos[t,uid]-pos[t-1,uid]).sum()<=1
                          and energy[t,uid]-energy[t-1,uid]<=20)
            if start is not None and not continuous:
                end = t-1
                death = (t<len(alive) and data['match_steps'][t]>0
                         and data['match_steps'][t]>data['match_steps'][end])
                yield start,end,bool(death)
                start = None
            if exists and start is None:
                start = t

    def _segment(self, events, start, end):
        cfg = self.cfg
        previous = start
        for t in range(start+1,end+1):
            e = events[t]
            p = events[t-1]
            signs = dict(stable=e[5]==cfg['stable_occupancy_steps'],
                         heading=e[3]>cfg['heading_change_threshold'],
                         belief=e[14]>=cfg['belief_entropy_threshold'] and p[14]<cfg['belief_entropy_threshold'],
                         visibility=e[15]>=cfg['new_visible_threshold'] and p[15]<cfg['new_visible_threshold'],
                         contact=e[20]>p[20],sap=e[21]>p[21],
                         score=e[10]-p[10]>=cfg['score_onset_threshold'],
                         energy=(e[6]<cfg['low_energy']) != (p[6]<cfg['low_energy']))
            event = sum(cfg['event_weights'][name]*flag for name,flag in signs.items())>=cfg['event_significance']
            if ((event and t-previous>=max(cfg['min_segment_length'],cfg['hysteresis_steps']))
                    or t-previous+1>=cfg['max_segment_length']):
                yield previous,t
                previous = t+1
        if previous<=end:
            yield previous,end

    def label(self, data):
        cfg = self.cfg
        events = semantic_events(data,cfg)
        shape = data['active'].shape
        modes = np.full(shape,-1,dtype=np.int8)
        goals = np.full((*shape,2),-1,dtype=np.int8)
        confidence = np.zeros(shape,dtype=np.float16)
        valid = np.zeros(shape,dtype=bool)
        segment_ids = np.full(shape,-1,dtype=np.int16)
        regions, segments = [], []
        for uid in range(16):
            for life_start,life_end,death in self._lives(data,uid):
                for start,end in self._segment(events[:,uid],life_start,life_end):
                    e = events[start:end+1,uid]
                    pos = data['positions'][start:end+1,uid]
                    duration = end-start+1
                    tail = min(cfg['anchor_tail_steps'],duration)
                    stability = min(1,e[-1,5]/cfg['stable_occupancy_steps'])
                    stable = e[-1,5]>=cfg['stable_occupancy_steps']
                    displacement = np.abs(pos[-1]-pos[0]).sum()
                    evidence = np.array([
                        e[:,11].max(), stability, min(1,e[:,10].sum()/duration),
                        min(1,e[:,14].sum()/max(cfg['belief_entropy_threshold'],1e-6)),
                        min(1,e[:,15].sum()/max(cfg['new_visible_threshold'],1)),
                        min(1,e[:,24].sum()/3),e[:,20].max(),e[:,21].max(),
                        float(e[0,6]<cfg['low_energy']),
                        min(1,max(0,e[-1,6]-e[:,6].min())/cfg['recharge_gain']),
                        float(data['features'][end,9,*pos[-1]]>0),
                        min(1,displacement/cfg['min_goal_distance']),
                        min(1,e[:,16].max()/2),min(1,e[:,23].sum()),
                    ])
                    scores = self.weights@evidence
                    probs = np.exp((scores-scores.max())/cfg['evidence_temperature'])
                    probs /= probs.sum()
                    mode = int(probs.argmax())
                    region = np.zeros((24,24),dtype=bool)
                    anchor_frame = end
                    quality = 0.0
                    if mode==0:  # true scoring anchor, not team-score attribution alone
                        if stable and data['features'][end,20,*pos[-1]]:
                            for xy in pos[-tail:]:
                                if data['features'][end,20,*xy]:
                                    region[tuple(xy)] = True
                            quality = 1.0
                    elif mode==1:
                        information = e[:,14]+e[:,15]/cfg['new_visible_threshold']
                        idx = int(np.argmax(information))
                        if information[idx]>=1 and idx>=duration-tail and displacement>=cfg['min_goal_distance']:
                            anchor_frame = start+idx
                            region[tuple(pos[idx])] = True
                            quality = .8  # team information gain is not uniquely attributable
                    elif mode==2:
                        if stable and e[:,16].max()>0 and data['features'][end,10,*pos[-1]]:
                            region[tuple(pos[-1])] = True
                            quality = .9
                    elif mode==3:
                        sap = np.flatnonzero(e[:,21]>0)
                        contact = np.flatnonzero(e[:,20]>0)
                        indices = sap if len(sap) else contact
                        if death and end==life_end:
                            indices = indices[indices<=duration-1-cfg['death_anchor_margin']]
                        if len(indices) and e[:,24].sum()>=cfg['intercept_min_approach']:
                            idx = int(indices[-1])
                            anchor_frame = start+idx
                            xy = pos[idx].copy()
                            if len(sap):
                                xy += data['actions'][anchor_frame,uid,1:]
                            if (xy>=0).all() and (xy<24).all():
                                region[tuple(xy)] = True
                                quality = .9 if len(sap) else .75
                    elif mode==4:
                        if stable and evidence[8]>0 and evidence[9]>=1 and evidence[10]>0:
                            region[tuple(pos[-1])] = True
                            quality = .9
                    conf = float(probs[mode]*quality)
                    margin = float(np.sort(probs)[-1]-np.sort(probs)[-2])
                    good = (duration>=cfg['min_segment_length'] and region.any()
                            and conf>=cfg['confidence_threshold'] and margin>=cfg['evidence_margin'])
                    if death and end==life_end and not stable and anchor_frame>end-cfg['death_anchor_margin']:
                        good = False  # a death endpoint alone is never a strategic anchor
                    sid = len(regions)
                    regions.append(region)
                    chosen = np.argwhere(region)[0] if region.any() else np.array([-1,-1])
                    # Label only frames at/before the established event/anchor.
                    stop = anchor_frame+1
                    modes[start:stop,uid] = mode
                    goals[start:stop,uid] = chosen
                    confidence[start:stop,uid] = conf
                    valid[start:stop,uid] = good
                    segment_ids[start:stop,uid] = sid
                    segments.append(dict(unit_id=uid,start=start,end=end,anchor_frame=anchor_frame,
                                         mode=MODES[mode],confidence=conf,valid=bool(good),
                                         right_censored=bool(death and end==life_end),
                                         evidence=evidence.tolist(),mode_scores=scores.tolist()))
        return dict(mode=modes,goal=goals,label_confidence=confidence,label_valid=valid,
                    segment_id=segment_ids,goal_regions=np.asarray(regions,dtype=np.uint8),
                    semantic_events=events.astype(np.float16)), segments
