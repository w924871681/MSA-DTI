"""Repeat target adaptation and selection on one CUDA device using the frozen source assets."""
import argparse
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import pandas as pd

from inputs import CODE, WORK, METHODS, TIERS, carrier, cases, digest, file_hash, metrics, read, truth
from verify_results import compare
sys.path.insert(0,str(CODE/"native/src"))
from msa_dti import runtime as u, baseline_runtime as br
from msa_dti.prefix import prefix_selection, DTYPE
from msa_dti.features import causal_features
import torch
WEIGHTS=WORK/"weights"
SETTINGS=CODE/"protocol"
PLAN=read(CODE/"verification/pipeline_plan.json")
POLICY=read(CODE/"data/frozen_policy.json")

@lru_cache(maxsize=2)
def search_weights(h):
    return {u.make_bank_key_shared(H=h,arch_key=a.arch_key,input_dim=25,L=96):
        load(WEIGHTS/f"search/H{h}/A{i}.pt")["state_dict"] for i,a in enumerate(u.ARCH)}

def load(path):
    return torch.load(path,map_location="cpu",weights_only=True)

def fit(i,h,seed,state,xs,ys,xv):
    model=u.build(i,h,state,seed)
    with u.candidate_backend_context(u.ARCH[i],u.DEVICE,"gru-native"):
        u.seed(seed)
        opt=torch.optim.SGD(model.parameters(),lr=.01)
        for _ in range(50):
            model.train();opt.zero_grad(set_to_none=True)
            ((model(xs)-ys)**2).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        model.eval()
        with torch.no_grad():val=model(xv).cpu().numpy()
    return model,val

def target_features(stage,c,raw_path,root):
    prefix=carrier(stage,c,raw_path,prefix=True)
    if digest(prefix)!=c["prefix_hash"]:raise ValueError("Wrong prefix")
    path=root/(c["slug"]+".bin")
    path.write_bytes(prefix)
    return prefix_selection(c,path,read(CODE/"protocol/E/normalization.json"))

def select(c,features,method,root):
    h,seed=c["H"],c["seed"]
    xs,ys,xv,yv=[torch.from_numpy(features[k]) for k in ("Xs","ys","Xv","yv")]
    tiers=list(TIERS)
    if method=="msa" or method in ("pt","medet","scratch"):
        xx,yy,vv=[x.to(u.DEVICE).contiguous() for x in (xs,ys,xv)]
        if method!="msa":
            spec=read(SETTINGS/(method+".json"))
            i=spec["selected"]["arch_id"]
            state=None if method=="scratch" else load(WEIGHTS/f"{method}/H{h}.pt")
            if state is not None:state=state.get("state_dict",state)
            model,_=fit(i,h,seed,state,xx,yy,vv)
            return {t:(i,model) for t in tiers}
        profiles={i:tuple(u.profile_arch(u.ARCH[i],96,25,h)) for i in PLAN["bank"]}
        out={}; cache={}
        for tier in tiers:
            ids=[i for i in profiles if all(x<=y for x,y in zip(profiles[i],PLAN["caps"][tier]))]
            for i in ids:
                if i not in cache:
                    state=load(WEIGHTS/f"msa/strong_H{h}_A{i}.pt")["state_dict"]
                    model,val=fit(i,h,seed,state,xx,yy,vv)
                    loss=np.square(val.astype(np.float64)-features["yv"].astype(np.float64)).mean(1).mean()
                    cache[i]=(loss,model)
            ref=POLICY["reference"]
            valid=[i for i in ids if i==ref or (POLICY["tau"] is not None and cache[i][0]<=(1-POLICY["tau"])*cache[ref][0])]
            i=min(valid,key=lambda j:(cache[j][0],*profiles[j],j));out[tier]=(i,cache[i][1])
        return out
    if method=="h_meta_nas":
        if not hasattr(br,"HM"):
            br.HM=load(WEIGHTS/"hmeta.pt");br.HMHASH=file_hash(WEIGHTS/"hmeta.pt")
        br.CURRENT=dict(case=c,prefix_sha=c["prefix_hash"],native_hash=file_hash(Path(br.__file__)),cache=root/"cache"/c["slug"],
                        logical_adaptations=0,actual_adaptations=0,cache_hits=0,events=[],proxies={})
        out={}
        for t in ("medium","tight","loose"):
            state,i,_=br.hmeta(h,t,seed,xs,ys,xv,yv)
            out[t]=(i,u.build(i,h,state,seed))
        return out
    pi=read(SETTINGS/"pi50.json")["pi50"] if method=="meta_nas_lite" else None
    out={}
    for tier in tiers:
        model,_,row,details,_,_=u.native._run_search_case(method,{"bank":search_weights(h),"pi50":pi},u.CFG,u.ARCH,u.DEVICE,
            "gru-native",xs,ys,xv,yv,tier,h,25,96,seed)
        i=int(row["arch_idx"])
        if i not in u.native.feasible_indices(u.CFG,u.ARCH,tier,96,25,h):raise ValueError("Over-budget search")
        out[tier]=(i,model)
    return out

def main():
    global WEIGHTS,SETTINGS,POLICY,PLAN
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=["bitbrains","azure"],required=True)
    parser.add_argument("--bitbrains",type=Path)
    parser.add_argument("--method",choices=list(METHODS),nargs="+",default=list(METHODS))
    parser.add_argument("--tasks",type=int,default=120)
    parser.add_argument("--trained",type=Path,help="Use a newly trained export directory for weights, reference, bank, margin, source prior and fixed backbones")
    parser.add_argument("--strict",action="store_true",help="Require exact architecture and numeric agreement at archive replay tolerance")
    args=parser.parse_args()
    if args.trained:
        SETTINGS=args.trained.resolve();WEIGHTS=SETTINGS/"weights"
        POLICY=read(SETTINGS/"policy.json");PLAN={**PLAN,"bank":POLICY["bank"]}
    if args.dataset=="bitbrains" and args.bitbrains is None:parser.error("Bitbrains needs --bitbrains PATH")
    if not 1<=args.tasks<=120 or len(args.method)!=len(set(args.method)):parser.error("Invalid task count or duplicate method")
    if not torch.cuda.is_available():raise SystemExit("CUDA required for the original comparison kernels")
    stage="E" if args.dataset=="bitbrains" else "azure"; cohort=cases(stage)[:args.tasks]; rows=[]
    WORK.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="target-",dir=WORK) as tmp:
        root=Path(tmp); actions=[]
        # Save model states for every requested task before opening future labels.
        for c in cohort:
            d=target_features(stage,c,args.bitbrains,root)
            for method in args.method:
                chosen=select(c,d,method,root); saved={}
                for tier,(i,model) in chosen.items():
                    path=root/f"{c['slug']}_{method}_{tier}.pt"
                    if id(model) not in saved:
                        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path)
                        saved[id(model)]=(path,file_hash(path))
                    path,sha=saved[id(model)]
                    actions.append((c,method,tier,i,path,sha))
                del chosen
            print("selected",c["slug"],flush=True)
        last=None
        for c,method,tier,i,path,sha in actions:
            if c["slug"]!=last:
                raw=carrier(stage,c,args.bitbrains);y,valid=truth(c,raw)
                r=np.frombuffer(raw,dtype=DTYPE)
                feat=causal_features(r["timestamp"],r["raw"],read(CODE/"protocol/E/normalization.json"))
                origins=np.arange(c["T"],c["T"]+c["B"])
                x=torch.from_numpy(np.stack([feat[o-95:o+1] for o in origins]).astype(np.float32)).to(u.DEVICE)
                last=c["slug"]
            if file_hash(path)!=sha:raise RuntimeError("Adapted state changed")
            m=u.build(i,c["H"],load(path),c["seed"]);m.eval()
            with u.candidate_backend_context(u.ARCH[i],u.DEVICE,"gru-native"),torch.no_grad():
                pred=m(x).cpu().numpy()[valid]
            rows.append(dict(method=METHODS[method],budget=tier,VM=c["VM"],H=c["H"],K=c["K"],arch=i,**metrics(pred,y)))
    frame=pd.DataFrame(rows);dest=WORK/"reruns";dest.mkdir(exist_ok=True)
    suffix="-".join(args.method) if len(args.method)<8 else "all"
    frame.to_csv(dest/f"{args.dataset}-{args.tasks}-{suffix}.csv",index=False)
    compare(args.dataset,frame,exact=args.strict)
if __name__=="__main__":
    main()

