"""Rebuild historical diagnostic tables; never fit a model."""
import argparse
import json
from collections import defaultdict

from research.evidence import new_output, read_evidence, save_svg

def reconstruct():
    d02 = read_evidence("D02_results.json")
    grouped = defaultdict(list)
    for row in d02["model_fits"]:
        grouped[(row["decoder"], row["sample_nodes"])].append(row["final"]["top1"])
    fits = [{"method": method, "nodes": n, "fit_top1": sum(v)/len(v),
             "seeds": len(v)} for (method,n),v in sorted(grouped.items())]
    for row in d02["same_original_function_class_convex_fits"]:
        fits.append({"method":"same_class_effective_LBFGS", "nodes":row["sample_nodes"],
                     "fit_top1":row["top1"], "seeds":1})
    choice = read_evidence("D10_selection.json")
    for row in choice["choices"]:
        selected = min(row["NLL"], key=lambda e:(row["NLL"][e],int(e)))
        if int(selected) != row["epoch"]:
            raise RuntimeError("Calibration selection not reproducible")
    if choice["exploration_opened_for_selection"]:
        raise RuntimeError("Exploration used for checkpoint selection")
    return {"D02_train_subset":fits, "D02_oracle":d02["linear_oracle_summary"],
            "D02_exact_collisions":d02["collisions"],
            "D08_exploration":read_evidence("D08_fit_decision.json")["per_fit_seed"],
            "D10_calibration_selection":choice["choices"],
            "D10_verdict":read_evidence("D10.json")["paired_verdict"],
            "scope":"Historical training-subset fit and reused-map diagnostics, not official reward or independent generalization",
            "new_fits":0, "environment_starts":0}

def plot(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"svg.hashsalt":"lux-research-sections","font.family":"DejaVu Sans"}):
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
        for method,label in [("original6831","Original head / Adam"),("nonlinear32688","Nonlinear head"),
                             ("same_class_effective_LBFGS","Same class / effective LBFGS")]:
            rows=sorted((r for r in data["D02_train_subset"] if r["method"]==method),key=lambda r:r["nodes"])
            axes[0].plot([r["nodes"] for r in rows],[100*r["fit_top1"] for r in rows],"o-",label=label)
        axes[0].set(xlabel="Fixed training subset nodes",ylabel="Train Goal Top-1 (%)",
                    title="D02: fitting diagnostic, not generalization",ylim=(0,105),xticks=[64,512])
        axes[0].legend(fontsize=8)
        for row in data["D10_calibration_selection"]:
            epochs=[8,16,24,32]
            axes[1].plot(epochs,[row["NLL"][str(e)] for e in epochs],"o-",label=str(row["seed"]))
        axes[1].set(xlabel="Epoch",ylabel="Equal-map Calibration pair NLL",
                    title="D10: fixed scorer, longer optimization",xticks=[8,16,24,32])
        axes[1].legend(title="Fit seed",fontsize=8)
        save_svg(fig,output/"representation_diagnostics.svg")
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
