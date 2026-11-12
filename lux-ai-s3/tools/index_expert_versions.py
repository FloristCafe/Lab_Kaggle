"""Join audited canonical EpisodeId/side to existing public submission metadata."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.contracts import json_write,sha


def index(manifest,metadata,output):
    if output.exists():raise FileExistsError(output)
    raw=json.loads(manifest.read_text())
    teams={};episodes={};sources={}
    for path in sorted(metadata.glob("submission_*.json")):
        if path.is_symlink():raise RuntimeError("unregistered metadata alias")
        data=json.loads(path.read_text());sources[str(path.absolute())]=sha(path)
        for team in data["teams"]:
            identity=team["teamName"]
            if team["id"] in teams and teams[team["id"]]!=identity:
                raise RuntimeError("team identity collision")
            teams[team["id"]]=identity
        for episode in data["episodes"]:
            rows={}
            for agent in episode["agents"]:
                side=agent.get("index",0)
                rows[str(side)]={"submission_id":agent["submissionId"],"team_id":agent["teamId"]}
            if set(rows)!={"0","1"}:raise RuntimeError("incomplete episode side identity")
            key=str(episode["id"])
            if key in episodes and rows!=episodes[key]:raise RuntimeError("submission identity collision")
            episodes[key]=rows
    results=[]
    for row in raw["sources"]:
        key=row["replay_id"]
        if key not in episodes:raise RuntimeError("missing canonical episode metadata: "+key)
        versions=episodes[key]
        for side,name in enumerate(row["teachers"]):
            if teams.get(versions[str(side)]["team_id"])!=name:
                raise RuntimeError("teacher name/side does not match metadata")
        results.append(dict(row,versions=versions))
    value={"schema":"expert_submission_join_v1","raw_manifest_sha256":sha(manifest),
           "sources":results,"indexed_games":len(results),"index_sources_sha256":sources,
           "selection":"No outcome re-filtering; original download selection may be win-biased."}
    json_write(output,value)
    print(json.dumps({"status":"PASS","games":len(results),"sha256":sha(output)}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--metadata",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();index(a.manifest,a.metadata,a.output)
