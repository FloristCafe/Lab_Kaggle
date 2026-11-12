"""Read-only reconstruction of committed research tables; no model or environment imports."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def read(path):return json.loads(Path(path).read_text())
def mean(values):return sum(values)/len(values)
def close(a,b):
    if abs(a-b)>1e-10:raise RuntimeError("saved statistic disagrees with retained records")
def digest(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def main():
    reports=ROOT/"reports";evidence=reports/"evidence";verified=0
    for source in read(evidence/"source_index.json"):
        if digest(ROOT/source["file"])!=source["sha256"]:raise RuntimeError("evidence snapshot changed")
        verified+=1
    result={"evidence_hashes_verified":verified}
    g05=read(evidence/"G05.json");model_deltas={}
    assert digest(evidence/"G05_protocol_R05.json")==g05["protocol_sha256"]
    for model,data in g05["models"].items():
        paired=data["paired"];rows=paired["map_rows"]
        delta=mean([mean(r["Bprime_by_side"])-mean(r["C_by_side"]) for r in rows])
        close(delta,paired["paired_Bprime_minus_C"]);model_deltas[model]=delta
    close(mean(list(model_deltas.values())),g05["aggregate"]["three_model_equal_mean_improvement"])
    result["G05"]={"models":model_deltas,"equal_model_delta":mean(list(model_deltas.values())),
                    "registered_map_interval":g05["aggregate"]["map_cluster_bootstrap95_fixed3_models"]}
    g06=read(evidence/"G06.json");result["G06"]={}
    assert digest(evidence/"G06_protocol.json")==g06["protocol_sha256"]
    for name,data in g06["official"]["aggregate"].items():
        delta=mean(data["map_differences"]);close(delta,data["three_model_equal_mean"])
        result["G06"][name]={"delta":delta,"registered_map_interval":data["map_bootstrap95_fixed_models"]}
    for study,left,right,arm_key,score_key,saved_key in [
        ("A01","JOINT","UID_GREEDY","allocator","official_margin","mean_JOINT_minus_UID_GREEDY"),
        ("A02","ABLATION","FULL","allocator","official_margin","mean_ABLATION_minus_FULL"),
        ("D12","PUBLIC","FULL","scoring","margin","point_PUBLIC_minus_FULL")]:
        data=read(evidence/(study+".json"));grid={}
        for row in data["games"]:
            case=row["case"];key=case["root"],case["side"],case[arm_key]
            if key in grid:raise RuntimeError("duplicate map/side/arm")
            value=row[score_key]
            if "official_wins" in row:
                wins=row["official_wins"];close(value,wins[case["side"]]-wins[1-case["side"]])
            grid[key]=value
        maps=sorted({k[0] for k in grid});assert len(maps)==24 and len(grid)==96
        differences=[mean([grid[(root,side,left)]-grid[(root,side,right)] for side in (0,1)]) for root in maps]
        delta=mean(differences);close(delta,data[saved_key])
        result[study]={"maps":24,"games":96,"delta":delta,"registered_map_interval":data["map_bootstrap95"],"decision":data["conclusion"]}
    pgc=read(evidence/"PGC01.json");same=reverse=zeros=0
    for row in pgc["rows"]:
        if row["missing"]:continue
        a,b=row["deltas"];product=(1 if a>0 else -1 if a<0 else 0)*(1 if b>0 else -1 if b<0 else 0)
        same+=product>0;reverse+=product<0;zeros+=product==0
    assert same==pgc["same_sign_nonzero"] and reverse==pgc["reverse_sign"]
    result["PGC01"]={"scheduled":pgc["scheduled_contexts"],"same_nonzero":same,"reversed":reverse,"zero_products":zeros,"decision":pgc["status"]}
    rows=[json.loads(line) for line in (reports/"results/training_a_games.jsonl").read_text().splitlines()];by_opp=defaultdict(list)
    assert sorted(r["game_index"] for r in rows)==list(range(1200))
    for row in rows:
        wins=row["official_wins"];side=row["side"];margin=wins[side]-wins[1-side]
        assert sum(wins)==5 and margin==row["margin"] and row["primitive_steps"]==505
        by_opp[row["opponent"]].append(margin)
    summary=read(reports/"results/training_a_summary.json");close(mean([r["margin"] for r in rows]),summary["training_margin"])
    for opponent,values in by_opp.items():
        expected=summary["by_opponent"][opponent];assert len(values)==expected["games"] and sum(v>0 for v in values)==expected["wins"];close(mean(values),expected["mean_margin"])
    result["A_training"]={"games":len(rows),"steps":sum(r["primitive_steps"] for r in rows),"mean_margin":summary["training_margin"],"scope":"changing training policies/opponent pool; not final checkpoint evaluation"}
    result["BC_selected_epochs"]={}
    for model,data in read(reports/"results/bc_training.json").items():
        best=min(data["epochs"],key=lambda row:(row["calibration_nll"],row["epoch"]))
        assert best["epoch"]==data["selection"]["selected_epoch"]
        result["BC_selected_epochs"][model]=best["epoch"]
    result["interval_scope"]="Intervals are preserved pre-registered outputs, not refitted here. This tool independently recomputes point estimates and retained outcome counts; full raw journals remain external."
    result["status"]="PASS"
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
