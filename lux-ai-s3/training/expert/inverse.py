"""Event-based, set-valued controller inversion on public recorded observations."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agents.mappo.shared.features import CausalObservationBuilder
from agents.mappo.shared.manager import TeamMemory
from agents.mappo.shared.physics import PhysicsState
from agents.mappo.shared.policy import pack_policy
from agents.mappo.shared.tactics import TacticalController
from training.contracts import config,json_write
from .replays import observation,actions,identity,sha


def segment(replay,start,side,uid):
    """Use a real boundary, retaining 0–100 steps rather than a fixed waypoint."""
    first=observation(replay,start,side)
    initial_alive=np.asarray(first["units_mask"])[side]
    dwell=0
    last=np.asarray(first["units"]["position"])[side,uid]
    for t in range(start+1,min(506,start+101)):
        obs=observation(replay,t,side)
        alive=np.asarray(obs["units_mask"])[side]
        if obs["match_steps"]==0:return t,"match_reset"
        if not alive[uid]:return t,"death_or_absence"
        if not np.array_equal(alive,initial_alive):return t,"team_membership_changed"
        xy=np.asarray(obs["units"]["position"])[side,uid]
        dwell=dwell+1 if np.array_equal(last,xy) else 0
        if dwell>=3:return t,"persistent_dwell"
        last=xy
    return min(505,start+100),"censored_at_max_option_age"


def diagnose(path,side,output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    replay=json.loads(path.read_text());info=identity(path,replay)
    cfg=config("a");env_cfg=info["environment"]
    builder=CausalObservationBuilder(side,env_cfg)
    physics=PhysicsState(side,env_cfg)
    manager=TeamMemory(cfg["review"],env_cfg)
    tactical=TacticalController(env_cfg,cfg["tactical"])
    t=0
    while t<505:
        obs=observation(replay,t,side)
        frame=builder.observe(obs)
        physics.observe(obs,frame,builder.reconstructor)
        manager.observe(frame)
        if frame["alive"].any():break
        t+=1
    uid=int(np.flatnonzero(frame["alive"])[0])
    # This initial context has no previous student intent and no unrecorded team assignments.
    if manager.valid.any():raise RuntimeError("inverse cold-context condition failed")
    passive=SimpleNamespace(side=side,env_cfg=env_cfg,obs=obs,steps=t,frame=frame,
                            builder=builder,physics=physics,manager=manager)
    state=pack_policy(passive,True)
    state_hash=hashlib.sha256(b"".join(v.tobytes() for v in state.values())).hexdigest()
    end,reason=segment(replay,t,side,uid)
    frames=[frame];teacher=[actions(replay,t,side)]
    for k in range(t+1,end):
        o=observation(replay,k,side)
        f=builder.observe(o)
        frames.append(f);teacher.append(actions(replay,k,side))
    # Every comparison is conditional on the recorded public trajectory.
    # No alternate environment state or counterfactual next observation is invented.
    candidate=np.flatnonzero(state["legal"])
    modes=[]
    counts=np.zeros(5,np.int64)
    begin=time.monotonic()
    original_rng=np.random.get_state()
    for mode in range(5):
        for canonical in candidate:
            if mode==0 and state["false_score"][canonical]:continue
            world=int(canonical if side==0 else 575-canonical)
            memory=copy.deepcopy(manager);control=copy.deepcopy(tactical)
            memory.valid[uid]=True;memory.goals[uid]=divmod(world,24);memory.modes[uid]=mode
            matched=True
            for f,expected in zip(frames,teacher,strict=True):
                selected,_=control.act(f,memory)
                if not np.array_equal(selected,expected):
                    matched=False;break
            if matched:
                modes.append([mode,int(canonical)]);counts[mode]+=1
    after=np.random.get_state()
    if original_rng[0]!=after[0] or not np.array_equal(original_rng[1],after[1]) or original_rng[2:]!=after[2:]:
        raise RuntimeError("inverse controller consumed RNG")
    if state_hash!=hashlib.sha256(b"".join(v.tobytes() for v in state.values())).hexdigest():
        raise RuntimeError("inverse controller modified feature snapshot")
    np.savez_compressed(output/"causal_state.npz",**state)
    support=sum(int((state["legal"] & (~state["false_score"] if m==0 else True)).sum()) for m in range(5))
    report={"schema":"primitive_inverse_diagnostic_v1","replay":info,"side":side,
            "teacher":info["teachers"][side],"uid":uid,"start":t,"end":end,"boundary":reason,
            "input_max_step":t,"label_future_steps":len(frames),"R":"unavailable",
            "Mode_Goal_set":modes,"count_by_mode":counts.tolist(),"legal_joint_support":support,
            "fraction_explained":len(modes)/max(1,support),
            "matching_rule":"exact full-team primitive actions on recorded public frames",
            "context":"known initial student context; teacher internal intent unavailable",
            "strict_whole_trajectory_counterfactual":False,
            "formal_bc_admitted":False,
            "limitations":["finite action agreement does not identify the teacher's latent Goal",
                           "cold-context result does not establish later fleet or Continue labels"],
            "seconds":time.monotonic()-begin,"state_sha256":sha(output/"causal_state.npz")}
    json_write(output/"inverse_report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("Mode_Goal_set","replay")},indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay",type=Path,required=True)
    p.add_argument("--side",type=int,choices=(0,1),required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();diagnose(a.replay,a.side,a.output)


if __name__=="__main__":main()
