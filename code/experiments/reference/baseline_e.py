"""Evaluate frozen C-selected baseline backbones on the same 120 E tasks."""
import os
import statistics
from pathlib import Path
import sys
import numpy as np

from prepare import ROOT,LEGACY,read,sha
import evaluation as ev

E=ROOT/"E_stage"
OUT=E/"baseline_results"
METHODS=("pt","medet","scratch")

def receipt(method):
    path=ROOT/"outputs"/f"{method}_selected.json"
    if not path.is_file():return None
    data=read(path)
    if data["method"]!=method:raise ValueError("C baseline receipt mismatch")
    return data

def select(method,c):
    import compute as u
    from core.methods.ours.stage2_runtime import candidate_backend_context
    cp=receipt(method);assert cp is not None
    folder=OUT/method/c["slug"];folder.mkdir(parents=True,exist_ok=True)
    done=folder/"selected.json"
    if done.exists():return
    h,s=c["H"],c["seed"];i=cp["selected"]["arch_id"]
    prefix=E/"inputs/prefix"/(c["slug"]+".bin");assert sha(prefix)==c["prefix_hash"]
    d=u.prefix_selection(c,prefix,read(E/"inputs/normalization.json"))
    xs,ys,xv=(u.torch.from_numpy(d[k]).to(u.DEVICE).contiguous() for k in ("Xs","ys","Xv"))
    source=None;source_sha=None
    if method!="scratch":
        ent=cp["source"][str(h)];path=ROOT/ent["path"]
        assert sha(path)==ent["sha256"]
        source_sha=ent["sha256"];obj=u.torch.load(path,map_location="cpu",weights_only=True)
        source=obj.get("state_dict",obj)
    m=u.model(i,h,s,source)
    with candidate_backend_context(u.ARCH[i],u.DEVICE,"gru-native"):
        u.seed(s);opt=u.torch.optim.SGD(m.parameters(),lr=.01)
        for _ in range(50):u.step(m,opt,xs,ys,1.)
        m.eval()
        with u.torch.no_grad():pred=m(xv).cpu().numpy()
    loss=u.score(pred,d["yv"]);state_hash=u.state_hash(m)
    u.save_pt(folder/"model.pt",u.cpu_state(m));ev.panel(folder/"validation.npz",{"prediction":pred})
    params,ops=u.profile_arch(u.ARCH[i],96,25,h)
    u.save(done,dict(method=method,slug=c["slug"],arch=i,H=h,K=c["K"],seed=s,params=params,operations=ops,
                     source_sha256=source_sha,reference_sha256=sha(ROOT/"outputs"/f"{method}_selected.json"),
                     E_lock=sha(E/"run_lock.json"),baseline_lock=sha(E/"baseline_run_lock.json"),prefix_sha256=c["prefix_hash"],validation_mse=loss,
                     state_sha256=state_hash,model_sha256=sha(folder/"model.pt"),panel_sha256=sha(folder/"validation.npz")))

def score(method,c):
    import compute as u
    from core.methods.ours.stage2_runtime import candidate_backend_context
    assert ev.GLOBAL.exists()
    seal=read(ev.GLOBAL);assert seal["signature"]==sha(E/"run_lock.json") and len(seal["actions"])==120
    folder=OUT/method/c["slug"];result=folder/"scores.json"
    if result.exists():return
    pick=read(folder/"selected.json");cp=receipt(method)
    assert cp and pick["reference_sha256"]==sha(ROOT/"outputs"/f"{method}_selected.json")
    assert pick["arch"]==cp["selected"]["arch_id"] and pick["E_lock"]==sha(E/"run_lock.json") and pick["baseline_lock"]==sha(E/"baseline_run_lock.json")
    assert sha(folder/"model.pt")==pick["model_sha256"] and sha(folder/"validation.npz")==pick["panel_sha256"]
    origin=LEGACY/"data/future/evaluation"/c["slug"]/"future.bin"
    assert sha(origin)==c["future_hash"]
    raw=np.fromfile(origin,dtype=u.DTYPE);t,h=c["T"],c["H"];assert len(raw)==t+c["B"]+h
    assert raw[:t+1].tobytes()==(E/"inputs/prefix"/(c["slug"]+".bin")).read_bytes()
    origins=np.arange(t,t+c["B"]);cpu=raw["raw"][origins[:,None]+np.arange(1,h+1),3]
    valid=np.all(np.isfinite(cpu)&(cpu>=0),axis=1);y=(cpu[valid]/100).astype(np.float32)
    features=u.causal_features(raw["timestamp"],raw["raw"],read(E/"inputs/normalization.json"))
    x=u.torch.from_numpy(np.stack([features[o-95:o+1] for o in origins]).astype(np.float32)).to(u.DEVICE)
    model=u.model(pick["arch"],h,c["seed"],u.torch.load(folder/"model.pt",map_location="cpu",weights_only=True))
    assert u.state_hash(model)==pick["state_sha256"]
    model.eval()
    with candidate_backend_context(u.ARCH[pick["arch"]],u.DEVICE,"gru-native"),u.torch.no_grad():pred=model(x).cpu().numpy()[valid]
    assert u.state_hash(model)==pick["state_sha256"]
    ev.panel(folder/"forecast.npz",{"prediction":pred})
    feasible={tier:pick["params"]<=p and pick["operations"]<=op for tier,(p,op) in ev.inputs()[0]["budgets"].items()}
    u.save(result,dict(method=method,slug=c["slug"],VM=c["VM"],H=h,K=c["K"],arch=pick["arch"],
          selected_sha256=sha(folder/"selected.json"),global_seal_sha256=sha(ev.GLOBAL),future_sha256=c["future_hash"],
          forecast_sha256=sha(folder/"forecast.npz"),n_valid=len(y),feasible=feasible,**ev.metrics(pred,y)))

def summarize(method,cases):
    from msa_dti.prefix import DTYPE
    from supervise import save
    cp=receipt(method);assert cp is not None and len(cases)==120
    rows=[]
    for c in cases:
        folder=OUT/method/c["slug"];r=read(folder/"scores.json");a=read(folder/"selected.json")
        assert r["selected_sha256"]==sha(folder/"selected.json") and r["global_seal_sha256"]==sha(ev.GLOBAL)
        assert sha(folder/"forecast.npz")==r["forecast_sha256"] and sha(folder/"validation.npz")==a["panel_sha256"]
        assert sha(folder/"model.pt")==a["model_sha256"] and a["reference_sha256"]==sha(ROOT/"outputs"/f"{method}_selected.json")
        assert a["baseline_lock"]==sha(E/"baseline_run_lock.json")
        assert (r["VM"],r["H"],r["K"],r["arch"])==(c["VM"],c["H"],c["K"],cp["selected"]["arch_id"])
        prefix=E/"inputs/prefix"/(c["slug"]+".bin");assert sha(prefix)==c["prefix_hash"]
        h,t=c["H"],c["T"];raw=np.fromfile(prefix,dtype=DTYPE)
        origins=np.arange(c["bounds"]["validation"][0]+95,t+1-h)
        val=raw["raw"][origins[:,None]+np.arange(1,h+1),3];val=(val[np.all(np.isfinite(val)&(val>=0),axis=1)]/100).astype(np.float32)
        with np.load(folder/"validation.npz",allow_pickle=False) as z:assert ev.metrics(z["prediction"],val)["MSE"]==a["validation_mse"]
        future=LEGACY/"data/future/evaluation"/c["slug"]/"future.bin";assert sha(future)==r["future_sha256"]==c["future_hash"]
        raw=np.fromfile(future,dtype=DTYPE);origins=np.arange(t,t+c["B"])
        later=raw["raw"][origins[:,None]+np.arange(1,h+1),3];later=(later[np.all(np.isfinite(later)&(later>=0),axis=1)]/100).astype(np.float32)
        with np.load(folder/"forecast.npz",allow_pickle=False) as z:
            assert all(r[k]==v for k,v in ev.metrics(z["prediction"],later).items())
        assert r["feasible"]=={tier:a["params"]<=p and a["operations"]<=op for tier,(p,op) in ev.inputs()[0]["budgets"].items()}
        rows.append(r)
    vms={r["VM"] for r in rows};assert len(vms)==30 and all(sum(r["VM"]==vm for r in rows)==4 for vm in vms)
    def mean(key):return statistics.mean(statistics.mean(r[key] for r in rows if r["VM"]==vm) for vm in sorted(vms))
    summary={key:mean(key) for key in ("MSE","MAE","Worst10")}
    summary["feasible"]={tier:statistics.mean(r["feasible"][tier] for r in rows) for tier in ev.inputs()[0]["budgets"]}
    save(E/"outputs"/f"baseline_{method}_verified.json",dict(status="PASS",method=method,tasks=len(rows),VMs=len(vms),summary=summary))
