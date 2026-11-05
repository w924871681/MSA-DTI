"""Group C source training, 50-shot-budget updates and held-out assessment.

Each work item owns one method, architecture and horizon. It publishes all
40 VM/support results together. Source training can resume at epoch/outer boundaries.
"""
import argparse
from contextlib import contextmanager
import copy
import csv
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import time

for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/"native"))
import numpy as np
import torch
from configs.main_cfg import CFG
from core.space import enumerate_A_base,build_model,profile_arch
from core.methods.ours.stage2_runtime import configure_stage2_runtime,candidate_backend_context
from msa_dti.features import causal_features,array_sha256
from msa_dti.prefix import prefix_selection,DTYPE

DEVICE=torch.device("cuda")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
ARCH=enumerate_A_base(CFG.arch)
D=ROOT/"inputs"
OUT=ROOT/"results"


def read(p):
    return json.loads(p.read_bytes().decode("utf-8-sig"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f".{os.getpid()}.pending")
    tmp.write_bytes((json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode())
    os.replace(tmp,p)


def save_pt(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f".{os.getpid()}.pending")
    torch.save(obj,tmp)
    os.replace(tmp,p)


def seed(s):
    random.seed(s);np.random.seed(s%(2**32-1));torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def model(idx,h,s,state=None):
    seed(s)
    m=build_model(ARCH[idx],input_dim=25,H=h,L=96,device=str(DEVICE))
    if state is not None:m.load_state_dict(state,strict=True)
    return m


def cpu_state(m):
    return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}


def state_hash(m):
    d={k:array_sha256(v.detach().cpu().numpy()) for k,v in m.state_dict().items()}
    return hashlib.sha256(json.dumps(d,separators=(",",":"),sort_keys=True).encode()).hexdigest()


def scalar_loss(m,x,y):
    value=((m(x)-y)**2).mean()
    if not torch.isfinite(value):raise ValueError("nonfinite MSE")
    return value


def step(m,opt,x,y,clip):
    m.train();opt.zero_grad(set_to_none=True)
    scalar_loss(m,x,y).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),clip,error_if_nonfinite=True)
    opt.step()


@lru_cache(maxsize=2)
def source_pt_data(h):
    with np.load(D/"pt_source.npz",allow_pickle=False) as z:
        return [(torch.from_numpy(z[f"X_source_{c}_H{h}"]),torch.from_numpy(z[f"y_source_{c}_H{h}"])) for c in range(20)]


@lru_cache(maxsize=80)
def episode(c,h,k):
    with np.load(D/"source_tasks"/f"s{c}_h{h}_k{k}.npz",allow_pickle=False) as z:
        return tuple(torch.from_numpy(z[n]).to(DEVICE).contiguous() for n in ("Xs","ys","Xq","yq"))


def train_source(method,i,h,identity,tag):
    dest=OUT/"source"/method/f"H{h}"/f"A{i}.pt"
    pending=dest.with_suffix(".progress.pt")
    done=dest.with_suffix(".json")
    if done.exists():
        r=read(done)
        if r["signature"]!=identity or sha(dest)!=r["sha256"]:raise ValueError(f"Bad source receipt: {dest}")
        return dest
    s=2904+(101 if method=="pt" else 211)*h+i
    m=model(i,h,s)
    opt=torch.optim.Adam(m.parameters(),lr=.001)
    rng=random.Random(s); start=0
    if pending.exists():
        partial=torch.load(pending,map_location="cpu",weights_only=True)
        if partial["signature"]!=identity:raise ValueError("Changed source training inputs")
        m.load_state_dict(partial["state_dict"]);opt.load_state_dict(partial["optimizer"])
        rng.setstate(partial["sampler_state"]);torch.set_rng_state(partial["cpu_rng"])
        torch.cuda.set_rng_state(partial["cuda_rng"]);start=partial["completed"]
    epochs=50 if method=="pt" else 400
    vmnames=read(D/"source_vms.json")
    with candidate_backend_context(ARCH[i],DEVICE,"gru-native"):
        for it in range(start,epochs):
            if method=="pt":
                data=source_pt_data(h)
                centers=list(range(20));random.Random(2904+10007*h+it if i==57 else s+it).shuffle(centers)
                for cid in centers:
                    subs=2904+101*h+1009*cid+it if i==57 else s+it+int(hashlib.sha256(vmnames[cid].encode()).hexdigest()[:8],16)
                    order=torch.randperm(100,generator=torch.Generator().manual_seed(subs))
                    x,y=data[cid]
                    for left in (0,64):
                        ix=order[left:left+64];step(m,opt,x[ix].to(DEVICE).contiguous(),y[ix].to(DEVICE).contiguous(),float("inf"))
            else:
                tasks=rng.sample(range(20),8)
                opt.zero_grad(set_to_none=True)
                grads=[torch.zeros_like(v) for v in m.parameters()]
                for cid in tasks:
                    xs,ys,xq,yq=episode(cid,h,(10,20)[(it+cid)%2])
                    seed(2904+1009*cid+37*h+it)
                    fast=copy.deepcopy(m); inner=torch.optim.SGD(fast.parameters(),lr=.01)
                    for _ in range(5):step(fast,inner,xs,ys,1.)
                    gg=torch.autograd.grad(scalar_loss(fast,xq,yq),tuple(fast.parameters()))
                    for v,g in zip(grads,gg):v.add_(g.detach()/8)
                    del fast,inner,gg
                for par,grad in zip(m.parameters(),grads):par.grad=grad
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
            if (it+1)%(5 if method=="pt" else 10)==0:
                save_pt(pending,dict(signature=identity,completed=it+1,state_dict=cpu_state(m),
                        optimizer=opt.state_dict(),sampler_state=rng.getstate(),cpu_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state()))
                save(OUT/"progress"/f"{tag}.json",dict(method=method,arch=i,H=h,phase="source",iteration=it+1,total=epochs,time=time.time()))
    save_pt(dest,dict(state_dict=cpu_state(m),arch_index=i,H=h,recipe=method,source_only=True))
    save(done,dict(signature=identity,arch=i,H=h,recipe=method,seed=s,epochs_or_meta_outer=epochs,sha256=sha(dest)))
    pending.unlink(missing_ok=True)
    return dest


@lru_cache(maxsize=8)
def features(slug,h,k,t,b,seed_value,nval,phash):
    p=D/"prefix"/(slug+".bin")
    if sha(p)!=phash:raise ValueError("Changed C prefix")
    data=prefix_selection(dict(slug=slug,H=h,K=k,T=t,B=b,seed=seed_value),p,read(D/"normalization.json"))
    assert len(data["Xs"])==k and len(data["Xv"])==nval
    return data


def assess_inputs(c,d):
    p=D/"assessment"/(c["slug"]+".bin")
    if sha(p)!=c["post_sha256"]:raise ValueError("Changed C assessment carrier")
    rows=np.fromfile(p,dtype=DTYPE)
    raw=np.concatenate((d["raw"],rows["raw"]));ts=np.concatenate((d["timestamps"],rows["timestamp"]))
    t,b,h=c["T"],c["B"],c["H"];assert len(raw)==t+b+h
    o=np.arange(t,t+b);ix=o[:,None]+np.arange(1,h+1);assert ix.min()==t+1
    cpu=raw[ix,3];keep=np.all(np.isfinite(cpu)&(cpu>=0),axis=1);assert keep.sum()==c["assessment_count"]
    full=causal_features(ts,raw,read(D/"normalization.json"))
    x=np.stack([full[q-95:q+1] for q in o]).astype(np.float32)
    return torch.from_numpy(x).to(DEVICE), (cpu[keep]/100).astype(np.float32),keep


def score(p,y):
    if p.shape!=y.shape:raise ValueError("Prediction/target shape")
    v=float(np.square(p.astype(np.float64)-y.astype(np.float64)).mean(axis=1).mean())
    if not np.isfinite(v) or v<0:raise ValueError("Invalid held-out loss")
    return v


def work(method,i,h,identity,cases,registry):
    tag=f"{method}_A{i}_H{h}";folder=OUT/"jobs"/tag;done=folder/"scores.json"
    if done.exists():
        j=read(done)
        if j["signature"]!=identity or sha(folder/"predictions.npz")!=j["predictions_sha256"]:raise ValueError("Invalid completed job")
        return
    key=f"{method}/{h}/{i}"
    if key in registry:
        ent=registry[key];p=ROOT/ent["path"];assert sha(p)==ent["sha256"];kind="verified_reuse"
    elif method in ("pt","medet"):
        p=train_source(method,i,h,identity,tag);kind="new_source"
    elif method=="scratch":
        p=None;kind="random"
    else:raise ValueError("Missing MSA C1 initialization")
    obj=torch.load(p,map_location="cpu",weights_only=True) if p else None
    st=obj.get("state_dict",obj) if obj is not None else None
    source_hash=sha(p) if p else None
    folder.mkdir(parents=True,exist_ok=True)
    rows=[];preds={}; start=time.monotonic()
    for c in cases:
        if c["H"]!=h:continue
        slug=c["slug"];s=c["seed"]
        d=features(slug,h,c["K"],c["T"],c["B"],s,c["validation_count"],c["prefix_sha256"])
        xs,ys,xv=(torch.from_numpy(d[k]).to(DEVICE).contiguous() for k in ("Xs","ys","Xv"))
        m=model(i,h,s,st)
        with candidate_backend_context(ARCH[i],DEVICE,"gru-native"):
            seed(s);opt=torch.optim.SGD(m.parameters(),lr=.01)
            for _ in range(50):step(m,opt,xs,ys,1.)
            m.eval()
            with torch.no_grad():v=m(xv).detach().cpu().numpy()
            vloss=score(v,d["yv"]);sh=state_hash(m)
            # Once weights are fixed, later observed values may feed rolling inference only.
            xc,yc,valid=assess_inputs(c,d)
            with torch.no_grad():f=m(xc).detach().cpu().numpy()[valid]
            if state_hash(m)!=sh:raise ValueError("Parameters changed during assessment")
        preds[slug+"__val"]=v;preds[slug+"__assess"]=f
        rows.append(dict(slug=slug,vm=c["VM"],h=h,k=c["K"],validation_mse=vloss,assessment_mse=score(f,yc),
                         adapted_sha256=sh,validation_count=len(v),assessment_count=len(f)))
        del m,opt,xs,ys,xv,xc
        if len(rows)%5==0:save(OUT/"progress"/f"{tag}.json",dict(phase="C",method=method,arch=i,H=h,completed=len(rows),time=time.time()))
    assert len(rows)==40
    tmp=folder/f"predictions.{os.getpid()}.pending"
    with tmp.open("wb") as stream:np.savez_compressed(stream,**preds)
    os.replace(tmp,folder/"predictions.npz")
    par,ops=profile_arch(ARCH[i],96,25,h)
    save(done,dict(signature=identity,method=method,arch=i,H=h,params=par,operations=ops,source_mode=kind,
                   source_path=p.relative_to(ROOT).as_posix() if p else None,source_sha256=source_hash,
                   predictions_sha256=sha(folder/"predictions.npz"),rows=rows,assessment_work_seconds=time.monotonic()-start))
    tmp_progress=OUT/"progress"/f"{tag}.json";tmp_progress.unlink(missing_ok=True)
    print("DONE",tag,kind,flush=True)


@contextmanager
def lock(path):
    import msvcrt
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a+b") as f:
        if f.tell()==0:f.write(b"0");f.flush()
        f.seek(0)
        try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            yield False;return
        try:yield True
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)


def tasks(protocol):
    # Start the MSA reference screen and a missing source of each recipe promptly.
    alljobs=[(m,i,h) for m in ("msa","pt","scratch","medet") for i in protocol["eligible"][m] for h in (1,4)]
    first=[("msa",22,1),("medet",0,1),("pt",4,1),("scratch",0,1)]
    return first+[j for j in alljobs if j not in first]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--worker",type=int,default=0);parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--one",nargs=3,metavar=("METHOD","ARCH","H"))
    args=parser.parse_args();assert 0<=args.worker<args.workers
    assert torch.cuda.is_available()
    configure_stage2_runtime(DEVICE,"gru-native")
    bundle=read(ROOT/"run_lock.json")
    for rel,digest in bundle["files"].items():
        if sha(ROOT/rel)!=digest:raise ValueError("Frozen dependency changed: "+rel)
    for rel,digest in read(D/"files.json").items():
        if sha(ROOT/rel)!=digest:raise ValueError("Changed bundled input: "+rel)
    if (str(torch.__version__),np.__version__)!=("2.13.0+cu126","2.1.3"):raise ValueError("Unexpected runtime versions")
    proto=read(ROOT/"protocol.json"); jobs=tasks(proto)
    if args.one:
        j=(args.one[0],int(args.one[1]),int(args.one[2]));assert j in jobs;jobs=[j]
    # Dynamic stealing: another single worker can also finish the queue after a peer exits.
    order=jobs[args.worker::args.workers]+[j for k in range(args.workers) if k!=args.worker for j in jobs[k::args.workers]]
    cases=read(D/"cases.json");registry=read(D/"reused_weights.json"); sig=sha(ROOT/"run_lock.json")
    for m,i,h in order:
        tag=f"{m}_A{i}_H{h}"
        if (OUT/"jobs"/tag/"scores.json").exists():continue
        with lock(OUT/"locks"/(tag+".guard")) as own:
            if own:work(m,i,h,sig,cases,registry)


if __name__=="__main__":
    main()
