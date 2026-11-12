"""Per-node LP feasibility diagnostic, not a deployable model or a shared-head proof."""
import numpy as np
from scipy.optimize import linprog

def solve(spatial,pos,mask,target):
    axis=np.arange(24);gx,gy=np.meshgrid(axis,axis,indexing="ij")
    distance=(abs(gx-pos[0])+abs(gy-pos[1])).reshape(-1)/46
    x=np.column_stack((spatial.reshape(32,576).T.astype(np.float64),distance))
    ids=np.flatnonzero(mask);other=ids[ids!=target]
    if not len(other):return {"status":"single_support","margin":1.0,"positive_margin":True}
    scale=np.maximum(np.std(x[ids],axis=0),1e-6);x=x/scale
    delta=x[other]-x[target]
    matrix=np.column_stack((delta,np.ones(len(other))))
    obj=np.zeros(34);obj[-1]=-1
    fit=linprog(obj,A_ub=matrix,b_ub=np.zeros(len(other)),bounds=[(-1,1)]*33+[(None,None)],method="highs")
    if not fit.success:raise RuntimeError("linear oracle solver "+str(fit.message))
    margin=float(fit.x[-1]);true_margin=float(np.min((x[target]-x[other])@fit.x[:33]))
    residual=float(max(0,(matrix@fit.x).max()))
    if residual>1e-6 or abs(margin-true_margin)>1e-5:raise RuntimeError("oracle residual")
    return {"status":"PASS","margin":margin,"positive_margin":margin>1e-5,
        "near_zero_margin":margin<1e-8,"numerically_ambiguous":1e-8<=margin<=1e-5,
        "constraint_residual":residual,"true_margin":true_margin}
