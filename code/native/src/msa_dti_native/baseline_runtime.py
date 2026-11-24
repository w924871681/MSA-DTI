from __future__ import annotations
import copy,gc,hashlib,io,json,random,time
from pathlib import Path
import numpy as np
import torch
from .io import sha,read,write,digest
from .runtime import CFG,ARCH,DEVICE
from . import baseline_methods as native
from . import hmeta as hm
from core.space import build_model,profile_arch
from core.methods.ours.stage2_runtime import candidate_backend_context
from core.methods.ours.weight_bank import make_bank_key_shared
from core.utils.metrics import eval_mae,eval_weighted_mse,eval_worst10

def arrhash(a):
    a=np.ascontiguousarray(a)
    return digest(dict(dtype=str(a.dtype),shape=list(a.shape),bytes=hashlib.sha256(a.tobytes()).hexdigest()))

def state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}


def statehash(st):return digest({k:arrhash(v.numpy()) for k,v in sorted(st.items())})


def savept(p,st):
 b=io.BytesIO();torch.save(st,b);p.write_bytes(b.getvalue())


def loadpt(p):return torch.load(io.BytesIO(p.read_bytes()),map_location='cpu',weights_only=True)


def build(idx,H,st,seed):
 native.seed_all(seed,DEVICE);m=build_model(ARCH[idx],25,H,96,device='cuda')
 if st is not None:m.load_state_dict(st,strict=True)
 return m


def metric_pred(m,x,y,idx):
 m.eval()
 with candidate_backend_context(ARCH[idx],DEVICE,'gru-native'),torch.no_grad():
  p=m(x.to(DEVICE).contiguous());yy=y.to(DEVICE).contiguous()
  val={'weighted_mse':float(eval_weighted_mse(p,yy)),'mae':float(eval_mae(p,yy)),'worst10':float(eval_worst10(p,yy,p=.10))}
  pred=p.detach().cpu().numpy()
 if not np.isfinite(pred).all() or not all(np.isfinite(list(val.values()))):raise RuntimeError('NONFINITE_VALIDATION')
 return val,pred


def key_for(idx,seed,recipe,source):return digest(dict(prefix=CURRENT['prefix_sha'],H=CURRENT['case']['H'],arch=idx,seed=seed,recipe=recipe,source=source,native=CURRENT['native_hash'],torch=torch.__version__,backend='cuda-gru-native'))


def cached_adapt(idx,st,source,seed,recipe,xs,ys,xv,yv):
 key=key_for(idx,seed,recipe,source);folder=CURRENT['cache']/key;folder.mkdir(parents=True,exist_ok=True);receipt=folder/'receipt.json'
 CURRENT['logical_adaptations']+=1
 if receipt.exists():
  r=read(receipt)
  for n,h in r['files'].items():assert sha(folder/n)==h,'CACHE_HASH_CHANGED'
  m=build(idx,CURRENT['case']['H'],loadpt(folder/'state.pt'),seed)
  CURRENT['cache_hits']+=1;CURRENT['events'].append(dict(kind='CANDIDATE_CACHE_REUSE',key=key,arch=idx,recipe=recipe,seed=seed))
  return m,r['validation'],r
 tick=time.perf_counter()
 if recipe=='hmeta_sgd_mse_50':
  m,_=hm._adapt_and_validate(spec=ARCH[idx],state=st,Xs=xs,ys=ys,Xv=xv,yv=yv,H=CURRENT['case']['H'],L=96,input_dim=25,device=DEVICE,seed=seed)
 else:
  m=build(idx,CURRENT['case']['H'],st,seed)
  with candidate_backend_context(ARCH[idx],DEVICE,'gru-native'):
   if recipe=='adam_huber_50':native._search_target_adapt(m,xs,ys,seed=seed)
   else:native._fixed_target_adapt(m,xs,ys,seed=seed)
 torch.cuda.synchronize();adapt_seconds=time.perf_counter()-tick
 val,pv=metric_pred(m,xv,yv,idx);ss=state(m);savept(folder/'state.pt',ss)
 b=io.BytesIO();np.savez_compressed(b,predictions=pv);(folder/'validation.npz').write_bytes(b.getvalue())
 r={'identity':key,'arch':idx,'seed':seed,'recipe':recipe,'source':source,'steps':50,'state_sha256':statehash(ss),'validation':val,'adapt_seconds':adapt_seconds,'files':{n:sha(folder/n) for n in ['state.pt','validation.npz']}}
 write(receipt,r);CURRENT['actual_adaptations']+=1;CURRENT['events'].append(dict(kind='ADAPT_COMPLETE',key=key,arch=idx,recipe=recipe,seed=seed,steps=50,seconds=adapt_seconds))
 return m,val,r


def search(method,H,tier,seed,xs,ys,xv,yv):
 # Call the exact original search routine. Its Adam adaptation passes seed
 # only to minibatch generation, so dropout RNG carries across shortlist
 # positions. Candidate-seed-only caching would change that native procedure.
 bank={make_bank_key_shared(H=H,arch_key=ARCH[i].arch_key,input_dim=25,L=96):st for i,st in C1[H].items()}
 model,spec,row,detail,count,adapted=native._run_search_case(method,{'bank':bank,'pi50':PI},CFG,ARCH,DEVICE,'gru-native',xs,ys,xv,yv,tier,H,25,96,seed)
 CURRENT['logical_adaptations']+=adapted;CURRENT['actual_adaptations']+=adapted
 CURRENT['events'].append({'kind':'NATIVE_SEARCH_COMPLETE','method':method,'budget':tier,'candidate_count':count,'actual_adaptations':adapted,'cross_candidate_rng':'original uninterrupted sequence','steps_each':50 if adapted else 0})
 selected=state(model);idx=row['arch_idx'];del model
 detail['feasible_arch_indices']=native.feasible_indices(CFG,ARCH,tier,96,25,H)
 return selected,idx,detail


def hmeta(H,tier,seed,xs,ys,xv,yv):
 feas=native.feasible_indices(CFG,ARCH,tier,96,25,H);rng=random.Random(seed);pop=rng.sample(feas,min(6,len(feas)));visited=set();rows=[];best=None;bst=None
 for generation in range(2):
  gr=[]
  for idx in pop:
   if idx in visited:continue
   st=HM['states'][hm._state_key(H,idx)];m,val,rec=cached_adapt(idx,st,HMHASH,seed+idx+10000*generation,'hmeta_sgd_mse_50',xs,ys,xv,yv)
   pa,op=profile_arch(ARCH[idx],96,25,H);row={'arch_idx':idx,'params':pa,'operations':op,'validation':val,'generation':generation,'adapt_identity':rec['identity']};rows.append(row);gr.append(row);visited.add(idx);lex=(val['weighted_mse'],pa,op,idx)
   if best is None or lex<best:best=lex;bst=state(m)
   del m
  if generation==0:
   pop=[]
   for p in sorted(gr,key=lambda r:(r['validation']['weighted_mse'],r['arch_idx']))[:3]:
    mut=hm._mutants(p['arch_idx'],feas,visited|set(pop),ARCH,rng)
    if mut:pop.append(mut[0])
   remain=[i for i in feas if i not in visited and i not in pop];rng.shuffle(remain);pop+=remain[:max(0,6-len(pop))]
 return bst,best[3],{'evaluated_candidates':rows,'feasible_arch_indices':feas}

