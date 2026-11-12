"""Map-cluster reconstruction from retained paired outcomes, not event-level causal attribution."""
from statistics import median

import numpy as np

def summarize_rows(rows,contexts,statistic):
    maps=int(statistic["maps"]);lookup={r["context"]:r for r in rows}
    if len(lookup)!=len(rows) or len(contexts)!=2*maps or len(lookup)!=len(contexts):
        raise RuntimeError("duplicate or omitted registered context")
    expected={c["context"]:c for c in contexts}
    if set(expected)!=set(lookup) or {(c["rank"],c["side"]) for c in contexts}!={(m,s) for m in range(maps) for s in (0,1)}:
        raise RuntimeError("registered map/side grid incomplete")
    products=np.zeros((maps,2));observed=missing=same=reverse=zeros=zero_deltas=action_divergences=distinct=0;durations=[]
    for key,context in expected.items():
        row=lookup[key]
        if any(row.get(field)!=context[field] for field in ("rank","side","root","window")):
            raise RuntimeError("context identity differs")
        if row["missing"]:
            if row.get("deltas") is not None:
                raise RuntimeError("missing node must not have invented rewards")
            missing+=1;continue
        deltas=row["deltas"]
        if len(deltas)!=2 or any(not np.isfinite(x) or abs(x)>10 or int(x)!=x for x in deltas):
            raise RuntimeError("two finite official reward differences required")
        product=int(np.sign(deltas).prod())
        if "product" in row and row["product"]!=product:raise RuntimeError("saved sign product differs")
        products[context["rank"],context["side"]]=product
        observed+=1;same+=product>0;reverse+=product<0;zeros+=product==0
        zero_deltas+=sum(x==0 for x in deltas)
        action_divergences+=sum(row.get("action_divergence",[]))
        distinct+=row["node"]["pair"][0]!=row["node"]["pair"][1]
        durations.extend(row.get("durations",[]))
    by_map=products.mean(1);draws=int(statistic["bootstrap"])
    if draws<1:raise RuntimeError("bootstrap budget must be positive")
    rng=np.random.default_rng(int(statistic["seed"]));indices=rng.integers(0,maps,size=(draws,maps))
    interval=np.quantile(by_map[indices].mean(1),[.025,.975]).tolist()
    fraction=same/len(contexts)
    supported=fraction>=statistic["agreement_fraction_min"] and interval[0]>statistic["interval_lower_must_exceed"]
    return {"status":"SIGNAL_SUPPORTED" if supported else "STOP_SIGNAL_NOT_ESTABLISHED",
            "maps":maps,"scheduled_contexts":len(contexts),"observed_contexts":observed,"missing_contexts":missing,
            "same_sign_nonzero":same,"same_sign_fraction_all":fraction,"reverse_sign":reverse,
            "zero_products_observed":zeros,"zero_deltas":zero_deltas,"repeated_pairs":2*observed,
            "action_divergences":action_divergences,"distinct_goal_contexts":distinct,
            "mean_sign_product":float(by_map.mean()),"95pct_map_interval":interval,
            "median_duration":median(durations) if durations else None,
            "per_map_sign_product":by_map.tolist(),"bootstrap_draws":draws,
            "scope":"fixed-prefix Actor sampling repeats; maps are clusters; no equivalence or general Goal irrelevance claim"}
