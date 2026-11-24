import numpy as np
from .io import require
DTYPE=np.dtype([('timestamp','<i8'),('raw','<f8',(10,))])

def prefix_selection(c,p,norm):
    from .features import causal_features,raw_role_bounds
    T,H,K=c['T'],c['H'],c['K'];bounds=raw_role_bounds(H,K)
    require(p.stat().st_size==(T+1)*DTYPE.itemsize,'PREFIX_MIXED_OR_TRUNCATED')
    rows=np.frombuffer(p.read_bytes(),dtype=DTYPE);t=rows['timestamp'].copy();raw=rows['raw'].copy();f=causal_features(t,raw,norm)
    so=np.arange(95,95+K,dtype=np.int64);lo,hi=bounds['validation'];vo=np.arange(lo+95,hi-H,dtype=np.int64)
    def labels(o):
        ix=o[:,None]+np.arange(1,H+1);require(ix.max()<=T,'LABEL_OUTSIDE_PREFIX')
        cpu=raw[ix,3];return (cpu/100).astype(np.float32),np.all(np.isfinite(cpu)&(cpu>=0),axis=1)
    sy,sm=labels(so);vy,vm=labels(vo);require(sm.all(),'SUPPORT_INVALID');vo=vo[vm];vy=vy[vm]
    require(len(vo)>=c['B'] and vo[-1]+H==T,'MATURE_ENDPOINT_OR_COUNT_INVALID')
    def inputs(o):return np.stack([f[i-95:i+1] for i in o]).astype(np.float32)
    require(not(set((so[:,None]+np.arange(1,H+1)).ravel())&set((vo[:,None]+np.arange(1,H+1)).ravel())),'SUPPORT_VAL_OVERLAP')
    return dict(Xs=inputs(so),ys=sy,Xv=inputs(vo),yv=vy,origins=vo,support_origins=so,features=f,timestamps=t,raw=raw)

