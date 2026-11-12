"""Recompute retained context statistics and the frozen map bootstrap, no rollouts."""
import argparse
import json

from research.evidence import new_output, read_evidence, save_svg
from .statistics import summarize_rows

def reconstruct():
    saved=read_evidence("PGC01.json")
    protocol=read_evidence("PGC01_protocol.json")
    contexts=read_evidence("PGC01_manifest.json")["contexts"]
    rebuilt=summarize_rows(saved["rows"],contexts,protocol["statistic"])
    for key,value in rebuilt.items():
        if key in saved and value!=saved[key]:
            raise RuntimeError("saved paired statistic mismatch: "+key)
    rebuilt["contexts_with_any_action_divergence"]=sum(any(r["action_divergence"]) for r in saved["rows"] if not r["missing"])
    rebuilt["duplicate_goal_contexts"]=rebuilt["observed_contexts"]-rebuilt["distinct_goal_contexts"]
    rebuilt["end_reasons"]=saved["end_reasons"]
    rebuilt["archived_manager_rejections"]=saved["manager_rejections"]
    rebuilt["frozen_bootstrap_seed"]=protocol["statistic"]["seed"]
    rebuilt["protocol_note"]="Executed frozen protocol specifies 20,000 draws; earlier conversation requested 10,000. Preserved version, not silently rewritten."
    rebuilt["offline_receipt_reconstruction_only"]=True
    return rebuilt

def plot(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"svg.hashsalt":"lux-research-sections","font.family":"DejaVu Sans"}):
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
        counts=[data["distinct_goal_contexts"],data["contexts_with_any_action_divergence"],data["same_sign_nonzero"]]
        axes[0].bar(["Different A/B","Any action\n divergence","Repeatable\nnonzero"],counts,color=["#4378a8","#dd9a3a","#b24c54"])
        axes[0].set(ylabel="Contexts / 48 (24 map clusters)",ylim=(0,50),title="Goal changes and reward changes differ")
        for i,n in enumerate(counts):
            axes[0].text(i,n+1,str(n),ha="center")
        saved=read_evidence("PGC01.json")
        deltas=[v for r in saved["rows"] if not r["missing"] for v in r["deltas"]]
        axes[1].hist(deltas,bins=[-.5,.5],color="#4378a8")
        axes[1].set(xlabel="Official reward difference A - B",ylabel="Repeated differences (not maps)",
                    title="PGC01: 96/96 repeated differences are zero",xticks=[0],xlim=(-2,2))
        save_svg(fig,output/"paired_goal_credit.svg")
        plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    args=parser.parse_args()
    data=reconstruct()
    if args.output_dir:
        out=new_output(args.output_dir)
        (out/"summary.json").write_text(json.dumps(data,indent=2)+"\n")
        plot(data,out)
    print(json.dumps(data,indent=2))

if __name__=="__main__":
    main()
