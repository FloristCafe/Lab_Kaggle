"""Strict legal SAP evidence, explicit censoring and auditable label-only bridges."""
import numpy as np

from agents.strategic.labeler import StrategicGoalLabeler

REASONS = ('valid_goal', 'right_censored', 'episode_boundary', 'invalid_sap_evidence',
           'ambiguous_intent', 'stationary_waiting', 'no_stable_anchor', 'insufficient_duration',
           'after_anchor', 'other')


def legal_sap(d):
    a = d['actions']
    targets = d['positions'] + a[:, :, 1:]
    cfg = d['env_cfg']
    attempt = d['active'] & (a[:, :, 0] == 5)
    valid = (attempt & (d['energies'] >= cfg['unit_sap_cost'])
             & (np.abs(a[:, :, 1:]).max(-1) <= cfg['unit_sap_range'])
             & (targets >= 0).all(-1) & (targets < 24).all(-1))
    return attempt, valid


def clean_labels(d, cfg, include_dirty=True):
    labeler = StrategicGoalLabeler(cfg['labeler'])
    attempt, legal = legal_sap(d)
    cleaned = {**d, 'actions': d['actions'].copy()}
    cleaned['actions'][attempt & ~legal] = 0
    labels, segments = labeler.label(cleaned)
    old, _ = labeler.label(d) if include_dirty else (None, None)
    shape = d['active'].shape
    reasons = np.full(shape, -1, np.int8)
    semantic = labels['label_valid'].copy()
    boundary = np.zeros(shape, bool)
    # Future evidence is used only for SUPERVISION. It never updates contexts.
    for s in segments:
        u, start, end = s['unit_id'], s['start'], s['end']
        sl = (slice(start, end + 1), u)
        why = 'other'
        if s['right_censored']:
            s['valid'] = False
            labels['label_valid'][sl] = False
            semantic[sl] = False
            why = 'right_censored'
        elif end == len(d['steps']) - 1 and d['match_steps'][end] != 0:
            s['valid'] = False
            labels['label_valid'][sl] = False
            semantic[sl] = False
            why = 'episode_boundary'
        elif end - start + 1 < cfg['labeler']['min_segment_length']:
            why = 'insufficient_duration'
        elif not labels['goal_regions'][labels['segment_id'][start, u]].any():
            why = 'invalid_sap_evidence' if (attempt & ~legal)[sl].any() else 'no_stable_anchor'
        elif s['confidence'] < cfg['labeler']['confidence_threshold']:
            why = 'ambiguous_intent'
        elif np.all(d['positions'][start:end + 1, u] == d['positions'][start, u]):
            why = 'stationary_waiting'
        else:
            why = 'after_anchor'
        reasons[sl] = REASONS.index(why)
        s['filter_reason'] = why
    reasons[labels['label_valid']] = 0
    # A short gap between SAME reliable intents is a weak continuation label,
    # never a new goal label. Require one continuous life and no invalid SAP.
    bridge = np.zeros(shape, bool)
    for u in range(16):
        good = [s for s in segments if s['unit_id'] == u and s['valid']]
        for left, right in zip(good, good[1:]):
            a, b = left['end'] + 1, right['start']
            if not (0 < b - a <= cfg['bridge_max_steps']) or left['mode'] != right['mode']:
                continue
            ls, rs = labels['segment_id'][left['start'], u], labels['segment_id'][right['start'], u]
            overlap = labels['goal_regions'][ls].astype(bool) & labels['goal_regions'][rs].astype(bool)
            continuous = (d['active'][a-1:b+1, u].all() and (np.diff(d['match_steps'][a-1:b+1]) == 1).all()
                          and (np.abs(np.diff(d['positions'][a-1:b+1, u], axis=0)).sum(1) <= 1).all()
                          and not (attempt & ~legal)[a:b, u].any())
            if overlap.any() and continuous:
                semantic[a:b, u] = True
                bridge[a:b, u] = True
                labels['mode'][a:b, u] = labels['mode'][left['start'], u]
                labels['goal'][a:b, u] = np.argwhere(overlap)[0]
                bridge_sid = len(labels['goal_regions'])
                labels['goal_regions'] = np.concatenate((labels['goal_regions'], overlap[None].astype(np.uint8)))
                labels['segment_id'][a:b, u] = bridge_sid
                labels['label_confidence'][a:b, u] = min(left['confidence'], right['confidence']) * .5
    for u in range(16):
        previous = None
        for t in range(len(d['steps'])):
            if not semantic[t, u]:
                previous = None
                continue
            sid = int(labels['segment_id'][t, u])
            region = labels['goal_regions'][sid].astype(bool)
            mode = int(labels['mode'][t, u])
            boundary[t, u] = (previous is None or previous[0] != mode or not (previous[1] & region).any())
            previous = mode, region
    labels.update(filter_reason=reasons, semantic_valid=semantic, boundary_replan=boundary,
                  bridge_continue=bridge, sap_attempt=attempt, sap_legal=legal)
    if old is not None:
        for key in ('mode', 'goal', 'label_confidence', 'label_valid', 'segment_id', 'goal_regions'):
            labels['dirty_' + key] = old[key]
    return labels, segments


def replan_targets(mode, regions, semantic, boundary, bridge, units, goal_valid):
    """Targets depend on causal candidate memory, never teacher-forced goals.

    Event boundaries and corrective replans are separately counted. Bridge
    supervision is permitted only for a compatible, genuinely existing goal.
    """
    previous = np.rint(units[:, 8:10] * 23).astype(int).clip(0, 23)
    pm = units[:, 12:17].argmax(1)
    valid_previous = units[:, 11] > .5
    compatible = valid_previous & (pm == mode) & regions[np.arange(16), previous[:, 0], previous[:, 1]]
    target = (boundary | ~compatible).astype(np.int8)
    mask = semantic & (goal_valid | (bridge & compatible))
    return target, mask, compatible
