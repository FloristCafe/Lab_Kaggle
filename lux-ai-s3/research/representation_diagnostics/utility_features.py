"""Same logical spatial views, with no teacher data in model features."""
import numpy as np
def window_bounds(x,y,r):
 return max(0,x-r),min(24,x+r+1),max(0,y-r),min(24,y+r+1)
def pooled_square(a,r):
 check=a.shape[-2:]==(24,24)
 if not check:raise ValueError("24x24 required")
 out=np.empty_like(a,dtype=np.float64)
 for x,y in np.ndindex(24,24):
  x0,x1,y0,y1=window_bounds(x,y,r)
  out[...,x,y]=a[...,x0:x1,y0:y1].mean(axis=(-2,-1),dtype=np.float64)
 return out
def raw_passable(history):
 # Native V2 only blocks asteroid observations from previous2 ticks, and current visible enemies.
 return ~((history[:,5]>.5).any(axis=0)|(history[0,10]>.0))
def public_row(history,xy,match_step,maximum,sensor,info_weight=1.,safe_bonus=.01):
 probability=history[0,8]
 support=history[0,7]>.5;unseen=history[0,1]>.5
 p=np.where(probability==.5,np.where(support,.25,.04),probability)
 information=np.zeros((24,24),np.float32)
 for x,y in np.ndindex(24,24):
  x0,x1,y0,y1=window_bounds(x,y,sensor);information[x,y]=unseen[x0:x1,y0:y1].sum()
 base=min(max(0,maximum-match_step),25)*.9*p+1.2*np.sqrt(information)
 base+=info_weight*4*p*(1-p)
 base+=safe_bonus*(p==0)
 x,y=np.indices((24,24));return (base/(1+.25*(abs(x-xy[0])+abs(y-xy[1])))).ravel()
def feature_views(history,spatial,position,unit,mode,step,match_step,sensor):
 """165 dims, equal nonpadded32D views; all current18 + past2x7 raw causal channels."""
 history=np.asarray(history,dtype=np.float64)
 raw=np.concatenate((history[0],history[1,[0,3,4,5,6,10,11]],history[2,[0,3,4,5,6,10,11]]),axis=0)
 enc=np.asarray(spatial,dtype=np.float64)
 if raw.shape!=(32,24,24) or enc.shape!=(32,24,24):raise ValueError("spatial shapes")
 x,y=map(int,position)
 raw_views=np.concatenate([raw.reshape(32,576).T,pooled_square(raw,sensor).reshape(32,576).T,
   np.repeat(raw[:,x,y][None],576,axis=0),np.repeat(raw.mean((1,2))[None],576,axis=0)],axis=1)
 enc_views=np.concatenate([enc.reshape(32,576).T,pooled_square(enc,sensor).reshape(32,576).T,
   np.repeat(enc[:,x,y][None],576,axis=0),np.repeat(enc.mean((1,2))[None],576,axis=0)],axis=1)
 gx,gy=np.indices((24,24));gx=gx.ravel();gy=gy.ravel()
 areas=np.array([(window_bounds(int(a),int(b),sensor)[1]-window_bounds(int(a),int(b),sensor)[0])*
  (window_bounds(int(a),int(b),sensor)[3]-window_bounds(int(a),int(b),sensor)[2]) for a,b in zip(gx,gy)])
 geo=np.stack((gx/23,gy/23,(gx-x)/46,(gy-y)/46,(abs(gx-x)+abs(gy-y))/46,
    areas/((2*sensor+1)**2),((gx==x)&(gy==y)).astype(float)),axis=1)
 common=np.concatenate([np.repeat(np.asarray(unit,dtype=float)[None],576,axis=0),
  np.repeat(np.eye(5)[int(mode)][None],576,axis=0),geo,
  np.repeat(np.array([[step/505,match_step/100,sensor/24]]),576,axis=0)],axis=1)
 if common.shape!=(576,37):raise ValueError("common37 shape")
 return np.concatenate((raw_views,common),axis=1).astype(np.float32),np.concatenate((enc_views,common),axis=1).astype(np.float32)
def analytic_from_RAW(x):
 """No teacher data; exact public formula recovered from declared RAW descriptors."""
 x=np.asarray(x);p=x[:,8].astype(np.float64);support=x[:,7]>.5
 p=np.where(p==.5,np.where(support,.25,.04),p)
 sensor=np.rint(x[:,-1]*24).astype(int);ms=np.rint(x[:,-2]*100).astype(int)
 # mean unseen is offset32+channel1; clipped box-area is shared geometry index5.
 area=np.rint(x[:,128+22+5+5].astype(float)*(2*sensor+1)**2)
 information=np.rint(x[:,32+1].astype(float)*area)
 distance=np.rint(x[:,128+22+5+4].astype(float)*46)
 base=np.minimum(np.maximum(0,100-ms),25)*.9*p+1.2*np.sqrt(information).astype(np.float32)
 # U_PUBLIC native scalar ops: replicate native sequence precision in independent verifier.
 base+=4*p*(1-p);base+=.01*(p==0)
 return base/(1+.25*distance)
def pairs_for_node(row,support,key,seed=202611086):
 """32 slots per node, equal row weight; strict public-label preferences only."""
 ids=np.flatnonzero(support)
 a,b=np.triu_indices(len(ids),1);a,b=ids[a],ids[b]
 scale=max(1.,float(np.max(abs(row[ids]))) if len(ids) else 1.)
 good=abs(row[a]-row[b])>1e-9*scale;a,b=a[good],b[good]
 gen=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,*map(int,key)])))
 out=np.zeros((32,2),np.int16);valid=np.zeros(32,bool)
 if not len(a):return out,valid
 # Uniform16 and near-best-vs-other16; fixed slots, not more supervision for larger masks.
 for start,kind in [(0,"uniform"),(16,"top")]:
  pool=np.arange(len(a))
  if kind=="top":
   top=ids[np.argsort(-row[ids],kind="stable")[:min(5,len(ids))]]
   take=np.isin(a,top)|np.isin(b,top)
   if take.any():pool=pool[take]
  chosen=gen.choice(pool,16,replace=len(pool)<16)
  out[start:start+16]=np.stack([a[chosen],b[chosen]],1);valid[start:start+16]=True
 return out,valid
