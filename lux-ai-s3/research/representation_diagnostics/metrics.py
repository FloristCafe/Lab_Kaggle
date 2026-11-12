"""Original exhaustive within-node ranking definitions."""
import numpy as np

def ranking(target,score,support):
 ids=np.flatnonzero(support);v=target[ids];q=score[ids]
 if not len(ids):return {"n":0,"strict_pairs":0,"exact_ties":0,"pair_acc":None,"best_hit":None,"regret":None}
 a,b=np.triu_indices(len(ids),1);dv=v[a]-v[b];dq=q[a]-q[b];strict=dv!=0
 concord=(dv*dq)>0;predtie=dq==0
 accuracy=float((concord[strict]+.5*predtie[strict]).mean()) if strict.any() else None
 tol=1e-9*max(1.,float(abs(v).max()));near=abs(dv)>tol
 acceps=float((concord[near]+.5*predtie[near]).mean()) if near.any() else None
 choice=int(np.argmax(q));rng=float(v.max()-v.min());reg=float((v.max()-v[choice])/rng) if rng>0 else 0.
 return {"n":len(ids),"strict_pairs":int(strict.sum()),"exact_ties":int((~strict).sum()),
   "pair_acc":accuracy,"pair_acc_epsilon":acceps,"best_hit":bool(v[choice]==v.max()),"regret":reg,"all_tied":rng==0,
   "best_ties":int((v==v.max()).sum()),"teacher_top5_hit":bool((v>v[choice]).sum()<5)}
