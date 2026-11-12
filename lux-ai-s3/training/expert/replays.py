"""Strict public-observation view of Kaggle replays; private info is never a feature."""
import hashlib
import json
from pathlib import Path

import numpy as np

PUBLIC_KEYS={"steps","match_steps","units","units_mask","sensor_mask","map_features",
             "relic_nodes","relic_nodes_mask","team_points","team_wins"}


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def observation(replay,t,side):
    raw=replay["steps"][t][side]["observation"]["obs"]
    raw=json.loads(raw) if isinstance(raw,str) else raw
    if not isinstance(raw,dict) or not PUBLIC_KEYS<=raw.keys():
        raise ValueError("incomplete public observation")
    # Only this allowlist may enter causal reconstruction.
    return {k:raw[k] for k in sorted(PUBLIC_KEYS)}


def actions(replay,t,side):
    # Action submitted in record t+1 is generated from record t's observation.
    value=replay["steps"][t+1][side]["action"]
    if isinstance(value,dict):value=value.get("action")
    value=np.asarray(value)
    if value.shape!=(16,3) or value.dtype.kind not in "iu":
        raise ValueError("missing/invalid submitted primitive action")
    return value.astype(np.int32)


def identity(path,replay):
    config=replay["configuration"]
    root=config.get("seed")
    if type(root) is not int or root<0 or root>=2**32:
        raise ValueError("map seed missing; cannot certify map-group separation")
    steps=replay["steps"]
    if len(steps)!=506 or replay.get("statuses")!=["DONE","DONE"]:
        raise ValueError("not a complete 505-step game")
    teachers=replay.get("info",{}).get("TeamNames")
    if not isinstance(teachers,list) or len(teachers)!=2:
        raise ValueError("teacher identity missing")
    for side in range(2):
        first,last=observation(replay,0,side),observation(replay,505,side)
        if first["steps"]!=0 or last["steps"]!=505 or sum(last["team_wins"])!=5:
            raise ValueError("public boundary or official win counters invalid")
    wins=observation(replay,505,0)["team_wins"]
    if replay.get("rewards")!=wins:
        raise ValueError("raw official rewards disagree with final public wins")
    episode=str(replay.get("info",{}).get("EpisodeId",path.stem))
    return {"replay_id":episode,"replay_uuid":replay.get("id"),"root":root,"teachers":teachers,"wins":wins,
            "environment":config["env_cfg"],"raw_sha256":sha(path),"path":str(path.absolute())}


def split(root):
    x=int(hashlib.sha256(f"expert-map-v1:{root}".encode()).hexdigest()[:8],16)%10
    return "train" if x<8 else "calibration" if x==8 else "exploration"


def audit(root,output):
    from collections import Counter
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    rows=[]; failures=[]; seen={}; teacher=Counter(); bytes_read=0
    files=sorted(p for p in root.glob("*.json") if not p.is_symlink())
    for i,path in enumerate(files):
        try:
            replay=json.loads(path.read_text())
            row=identity(path,replay)
            row["split"]=split(row["root"])
            duplicate=seen.get(row["replay_id"])
            if duplicate:
                if duplicate!=row["raw_sha256"]:raise ValueError("same replay ID has differing content")
                failures.append({"path":str(path),"reason":"exact duplicate replay"});continue
            seen[row["replay_id"]]=row["raw_sha256"]
            rows.append(row);teacher.update(row["teachers"])
            bytes_read+=path.stat().st_size
        except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
            failures.append({"path":str(path),"reason":str(exc)})
        if (i+1)%100==0:print("AUDIT",i+1,len(files),flush=True)
    map_split={r["root"]:r["split"] for r in rows}
    report={"schema":"expert-raw-public-v1","sources":rows,"rejected":failures,
            "games":len(rows),"maps":len(map_split),"teacher_sides":dict(teacher),
            "split_maps":dict(Counter(map_split.values())),
            "raw_bytes":bytes_read,"private_info_used_as_feature":False,
            "old_waypoint_npz_directly_reusable":False,
            "teacher_version":"team names available; per-game submission version requires separate source index"}
    with (output/"manifest.json").open("x") as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ("sources","rejected")},indent=2))
    return report


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit",action="store_true",required=True)
    p.add_argument("--replay-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();audit(a.replay_root,a.output)


if __name__=="__main__":main()
