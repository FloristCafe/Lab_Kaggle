"""Admit only explicitly named two-side, complete-game engineering evidence."""
import argparse
import json
import subprocess
from pathlib import Path

from training.contracts import json_write,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--evidence-index",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    registry=json.loads(a.candidate.read_text());index=json.loads(a.evidence_index.read_text())
    root=Path(__file__).resolve().parents[2]
    for name,spec in registry["agents"].items():
        evidence=index.get(name)
        if evidence is None:continue
        outcomes=[]
        for side,directory in enumerate(evidence):
            directory=Path(directory)
            summary=json.loads((directory/"summary.json").read_text())
            rows=json.loads((directory/"trajectory.json").read_text())
            if (summary["status"]!="PASS" or summary["agent"]!=name or summary["side"]!=side or
                summary["steps"]!=505 or not summary["complete"] or
                [x["step"] for x in rows]!=list(range(1,506)) or not rows[-1]["terminal"]):
                raise RuntimeError("opponent evidence is incomplete")
            prev=[0,0]
            for t,row in enumerate(rows,1):
                diff=sorted(x-y for x,y in zip(row["wins"],prev,strict=True))
                if diff!=([0,1] if t%101==0 else [0,0]):raise RuntimeError("official win boundary mismatch")
                prev=row["wins"]
            if summary["wins"]!=rows[-1]["wins"]:raise RuntimeError("official terminal mismatch")
            outcomes.append({"summary":str(directory/"summary.json"),"summary_sha256":sha(directory/"summary.json"),
                             "trajectory_sha256":sha(directory/"trajectory.json"),"root":summary["root"],"side":side})
        if len(outcomes)!=2:raise RuntimeError("both sides must complete")
        for path,h in spec["sha256"].items():
            if sha(path)!=h:raise RuntimeError("tested native source changed")
        query="import sys,torch,numpy,scipy,json;print(json.dumps(dict(python=sys.version.split()[0],torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__)))"
        spec["runtime"]=json.loads(subprocess.check_output([spec["python"],"-c",query],text=True))
        spec["license_sha256"]=sha(spec["license_path"])
        spec["enabled"]=True
        spec["full_game_acceptance"]=outcomes
        spec["wrapper_sha256"]={str(root/p):sha(root/p) for p in
                                ("scripts/expert_worker.py","scripts/official_worker.py","training/opponents/client.py")}
    if sum(s["enabled"] for s in registry["agents"].values())<1:
        raise RuntimeError("no accepted external agents")
    json_write(a.output,registry)
    print(json.dumps({"status":"PASS","enabled":[n for n,v in registry["agents"].items() if v["enabled"]],
                      "registry_sha256":sha(a.output)},indent=2))


if __name__=="__main__":main()
