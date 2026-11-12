"""Independent dataset receipts, canonical node identities and grouped partition audit."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.contracts import json_write,sha


def verify(artifacts,output):
    if output.exists():raise FileExistsError(output)
    macro=artifacts/"macro_dataset01"
    primitive=artifacts/"primitive_flat02"
    mm=json.loads((macro/"manifest.json").read_text())
    pm=json.loads((primitive/"manifest.json").read_text())
    counts=Counter();modes=Counter();identities=set();maps={};replan=continuing=0
    for shard in mm["shards"]:
        path=macro/shard["file"]
        if sha(path)!=shard["sha256"] or sha(path.with_suffix(".json"))!=shard["meta_sha256"]:
            raise RuntimeError("macro receipt differs")
        with np.load(path,allow_pickle=False) as z:
            t=z["timestep"]
            if len(t)!=shard["rows"] or (np.diff(t)<=0).any():raise RuntimeError("macro node time duplicated")
            for row,step in enumerate(t):
                key=(shard["map"],shard["side"],int(step))
                if key in identities:raise RuntimeError("duplicate canonical team decision")
                identities.add(key)
                review=z["review"][row];cmd=z["commands"][row]
                claims=np.where(z["valid"][row],z["previous"][row],-1)
                for uid in range(16):
                    if not review[uid]:continue
                    claims[uid]=-1
                    if cmd[uid,0]:
                        mode,goal=cmd[uid,1:]
                        if not z["legal"][row,goal] or (mode==0 and z["false_score"][row,goal]) or goal in claims:
                            raise RuntimeError("teacher violates real sequential candidate support")
                        claims[uid]=goal;replan+=1;modes[int(mode)]+=1
                    else:
                        claims[uid]=z["previous"][row,uid];continuing+=1
        counts[shard["split"]]+=shard["rows"]
        if shard["map"] in maps and maps[shard["map"]]!=shard["split"]:raise RuntimeError("macro map split leakage")
        maps[shard["map"]]=shard["split"]
    primitive_counts=Counter();primitive_maps={};seen=set()
    for shard in pm["shards"]:
        path=primitive/shard["file"]
        if sha(path)!=shard["sha256"]:raise RuntimeError("primitive receipt differs")
        if shard["submission_id"]!=pm["submission_id"]:raise RuntimeError("teacher versions mixed")
        key=shard["replay"],shard["side"]
        if key in seen:raise RuntimeError("duplicate primitive game")
        seen.add(key)
        with np.load(path,allow_pickle=False) as z:
            if z["expert_action"].shape!=(shard["rows"],16) or not np.isin(z["expert_action"],range(6)).all():
                raise RuntimeError("expert action classes invalid")
            if z["grid"].shape!=(shard["rows"],19,24,24):raise RuntimeError("primitive feature schema differs")
        primitive_counts[shard["split"]]+=shard["rows"]
        if shard["root"] in primitive_maps and primitive_maps[shard["root"]]!=shard["split"]:
            raise RuntimeError("primitive map split leakage")
        primitive_maps[shard["root"]]=shard["split"]
    joint={}
    for roots in (maps,primitive_maps):
        for root,split in roots.items():
            for derived in (root,root+103):
                if derived in joint and joint[derived]!=split:
                    raise RuntimeError("cross-dataset root/derived split leakage")
                joint[derived]=split
    pool=json.loads((ROOT/"training/configs/maps.json").read_text())
    train={v for r in pool["training"] for v in (r,r+103)}
    report={"status":"PASS","macro_manifest_sha256":sha(macro/"manifest.json"),
            "primitive_manifest_sha256":sha(primitive/"manifest.json"),
            "macro_team_decisions":len(identities),"macro_counts":dict(counts),"macro_maps":len(maps),
            "macro_uid_replan":replan,"macro_uid_continue":continuing,"macro_modes":dict(modes),
            "primitive_teacher":pm["teacher"],"primitive_submission_id":pm["submission_id"],
            "primitive_counts":dict(primitive_counts),"primitive_games":len(seen),"primitive_maps":len(primitive_maps),
            "cross_dataset_root_intersection":len(set(maps)&set(primitive_maps)),
            "BC_train_root_and_derived_vs_RL_training_intersection":len({v for r,s in {**maps,**primitive_maps}.items() if s=="train" for v in (r,r+103)}&train),
            "BC_maps_vs_evaluation_map_intersection":len((set(maps)|set(primitive_maps))&set(pool["development"])),
            "labels":"actual G06 R/Mode/Manager Goal, plus separately identified elite primitive auxiliary",
            "elite_long_goal_admission":False,"old400_provenance":"unresolved"}
    if report["BC_train_root_and_derived_vs_RL_training_intersection"]:
        raise RuntimeError("documented dataset partition isolation differs")
    json_write(output,report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();verify(a.artifacts,a.output)
