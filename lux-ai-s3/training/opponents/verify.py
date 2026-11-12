"""Complete public engineering games for admission, no network or RL updates."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from agents.mappo.shared.environment import Bridge
from training.contracts import json_write

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"agents/v2"))
from agent import Agent as V2Agent


def run(name,side,steps,output):
    output.mkdir(parents=True,exist_ok=False)
    os.environ["LUX_OPPONENT_ENGINEERING"]="1"
    bridge=Bridge(output/"worker.log")
    begin=time.monotonic()
    rows=[]
    rng=15001+103*side
    np.random.seed(rng)
    try:
        result=bridge.request(op="reset",seed=15001,side=side,opponent=name)
        own=V2Agent(f"player_{side}",result["env_cfg"])
        for t in range(steps):
            obs=result["obs"]
            action=own.act(t,obs)
            result=bridge.request(op="step",actions=action.tolist())
            theirs=result["opponent_action"]
            a=np.asarray(theirs)
            if a.shape!=(16,3) or a.dtype.kind not in "iu" or not np.isin(a[:,0],np.arange(6)).all():
                raise RuntimeError("malformed external action")
            if (np.abs(a[a[:,0]==5,1:])>7).any():
                raise RuntimeError("SAP offset outside official action box")
            rows.append({"step":t+1,"ours":action.tolist(),"opponent":theirs,
                         "public_obs_sha256":hashlib.sha256(json.dumps(result["obs"],sort_keys=True).encode()).hexdigest(),
                         "wins":result["obs"]["team_wins"],"terminal":result["task_complete"]})
        if steps==505 and (not result["task_complete"] or sum(rows[-1]["wins"])!=5):
            raise RuntimeError("incomplete five-match game")
        wins=rows[-1]["wins"]
        summary={"status":"PASS","agent":name,"side":side,"root":15001,"steps":steps,
                 "complete":bool(result["task_complete"]),"wins":wins,
                 "our_margin":wins[side]-wins[1-side],"seconds":time.monotonic()-begin}
        json_write(output/"trajectory.json",rows)
        json_write(output/"summary.json",summary)
        print(summary,flush=True)
        return summary
    except BaseException as exc:
        json_write(output/"failure.json",{"type":type(exc).__name__,"error":str(exc)})
        raise
    finally:bridge.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=505,choices=(4,16,505))
    p.add_argument("--side",type=int,choices=(0,1),required=True)
    a=p.parse_args()
    run(a.agent,a.side,a.steps,a.output)


if __name__=="__main__":main()
