"""Paired ancestor-VM aggregation used by every reported comparison."""
import math
import numpy as np
from .io import require
METRICS=("MSE","MAE","Worst10","CVaR90_case","CVaR90_VM_supplement")

def task_metrics(pred,target,mask):
    p=np.asarray(pred,dtype=np.float64);y=np.asarray(target,dtype=np.float64);mask=np.asarray(mask,dtype=bool)
    require(p.shape==y.shape and p.ndim==2 and mask.shape==(p.shape[0],),'METRIC_SHAPE')
    require(np.isfinite(p).all(),'NONFINITE_PREDICTION_MUST_NOT_MASK')
    require(mask.any() and np.isfinite(y[mask]).all(),'INVALID_VALID_TARGET')
    residual=p[mask]-y[mask];window=np.square(residual).mean(axis=1);k=max(1,round(.1*len(window)))
    return dict(MSE=float(window.mean()),MAE=float(np.abs(residual).mean()),Worst10=float(np.sort(window)[-k:].mean()),N_valid=int(mask.sum()),worst_window_count=k)


def tailmean(values,fraction=.1,axis=-1):
    a=np.asarray(values,dtype=np.float64);k=max(1,math.ceil(fraction*a.shape[axis]));return np.sort(a,axis=axis).take(indices=range(a.shape[axis]-k,a.shape[axis]),axis=axis).mean(axis=axis)


def matrix_for(rows,vms,metric):
    lookup={(r['VM'],r['H'],r['K']):r[metric] for r in rows}
    require(len(lookup)==4*len(vms) and len(rows)==len(lookup),'VM_HK_PAIRING')
    return np.asarray([[lookup[v,h,k] for h,k in [(1,10),(1,20),(4,10),(4,20)]] for v in vms],dtype=np.float64)


def aggregate(rows,vms,indices):
    values={m:matrix_for(rows,vms,m) for m in METRICS[:3]};vm={m:a.mean(axis=1) for m,a in values.items()}
    point={m:float(a.mean()) for m,a in vm.items()};boots={m:a[indices].mean(axis=1) for m,a in vm.items()}
    point['CVaR90_case']=float(tailmean(values['MSE'].ravel()));point['CVaR90_VM_supplement']=float(tailmean(vm['MSE']))
    repeated=values['MSE'][indices].reshape(len(indices),-1)
    boots['CVaR90_case']=tailmean(repeated,axis=1);boots['CVaR90_VM_supplement']=tailmean(vm['MSE'][indices],axis=1)
    for m,b in boots.items():
        lo,hi=np.quantile(b,[.025,.975],method='linear');point[m+'_ci_low']=float(lo);point[m+'_ci_high']=float(hi)
    return point,boots,vm,values

