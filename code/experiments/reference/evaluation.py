"""Budget-first deployment on the original E tasks, then sealed rolling forecast scoring."""
import csv
import io
import os
from pathlib import Path
import statistics
import sys
from functools import lru_cache

import numpy as np
from prepare import ROOT,LEGACY,read,sha,emit,save
from calibration import choose
from supervise import save as publish

E=ROOT/"E_stage"
sys.path.insert(0,str(ROOT/"native"))
GLOBAL=E/"outputs/all_actions.json"
FINAL=E/"outputs/verified_summary.json"

@lru_cache(None)
def inputs():
    return (read(E/"protocol.json"),read(E/"inputs/bank.json"),read(E/"inputs/frozen_policy.json"))

def budgets(h):
    protocol,bank,_=inputs();entries=bank[str(h)]
    for tier,(pmax,omax) in protocol["budgets"].items():
        yield tier,[i for i in protocol["bank"] if entries[str(i)]["parameters"]<=pmax and entries[str(i)]["operations"]<=omax]

def actions(loss,profile):
    _,_,policy=inputs();ref=policy["reference"]
    return dict(msa=ref if policy["policy"]=="pure_reference" else choose(loss,ref,policy["tau"],profile),
                direct_minimum=min(loss,key=lambda i:(loss[i],*profile[i],i)),reference_only=ref)

def panel(path,arrays):
    tmp=path.with_name(path.name+f".{os.getpid()}.pending")
    with tmp.open("wb") as f:np.savez_compressed(f,**arrays)
    os.replace(tmp,path)

def select(c,progress):
    import compute as u
    from core.methods.ours.stage2_runtime import candidate_backend_context
    _,bank,_=inputs();slug=c["slug"];h=c["H"];s=c["seed"]
    folder=E/"results"/slug
    if (folder/"selected.json").exists():return
    p=E/"inputs/prefix"/(slug+".bin");assert sha(p)==c["prefix_hash"]
    data=u.prefix_selection(c,p,read(E/"inputs/normalization.json"))
    xs,ys,xv=(u.torch.from_numpy(data[key]).to(u.DEVICE).contiguous() for key in ("Xs","ys","Xv"))
    complete=0;locks={}
    for tier,ids in budgets(h):
        out=folder/tier;out.mkdir(parents=True,exist_ok=True);receipt=out/"actions.json"
        if not receipt.exists():
            rows=[];states={};preds={};loss={};profile={}
            for n,i in enumerate(ids,1):
                entry=bank[str(h)][str(i)];path=E/"inputs"/entry["path"]
                assert sha(path)==entry["sha256"]
                state=u.torch.load(path,map_location="cpu",weights_only=True)
                m=u.model(i,h,s,state.get("state_dict",state))
                with candidate_backend_context(u.ARCH[i],u.DEVICE,"gru-native"):
                    u.seed(s);opt=u.torch.optim.SGD(m.parameters(),lr=.01)
                    for _ in range(50):u.step(m,opt,xs,ys,1.)
                    m.eval()
                    with u.torch.no_grad():v=m(xv).cpu().numpy()
                loss[i]=u.score(v,data["yv"]);assert loss[i]>0
                profile[i]=u.profile_arch(u.ARCH[i],96,25,h)
                assert profile[i]==(entry["parameters"],entry["operations"])
                preds[str(i)]=v;states[i]=u.cpu_state(m)
                rows.append(dict(arch=i,validation=loss[i],source_sha256=entry["sha256"],state_sha256=u.state_hash(m)))
                progress(complete+n,55)
            picks=actions(loss,profile)
            u.save_pt(out/"models.pt",{i:states[i] for i in set(picks.values())})
            panel(out/"validation.npz",preds)
            u.save(receipt,dict(slug=slug,tier=tier,feasible_before_loading=ids,adapted=[r["arch"] for r in rows],rows=rows,picks=picks,
                     validation_sha256=sha(out/"validation.npz"),models_sha256=sha(out/"models.pt"),signature=sha(E/"run_lock.json")))
        complete+=len(ids);locks[tier]=sha(receipt)
    u.save(folder/"selected.json",dict(slug=slug,actions=locks))

def verified_action(c,tier,ids,truth=None):
    f=E/"results"/c["slug"];out=f/tier
    a=read(out/"actions.json")
    assert sha(out/"actions.json")==read(f/"selected.json")["actions"][tier]
    assert (a["slug"],a["tier"],a["signature"])==(c["slug"],tier,sha(E/"run_lock.json"))
    assert a["feasible_before_loading"]==a["adapted"]==[r["arch"] for r in a["rows"]]==ids
    assert sha(out/"models.pt")==a["models_sha256"] and sha(out/"validation.npz")==a["validation_sha256"]
    bank=inputs()[1][str(c["H"])]
    assert all(r["source_sha256"]==bank[str(r["arch"])]["sha256"] for r in a["rows"])
    losses={r["arch"]:r["validation"] for r in a["rows"]}
    if truth is not None:
        with np.load(out/"validation.npz",allow_pickle=False) as z:
            assert set(z.files)==set(map(str,ids))
            for i in ids:
                pred=z[str(i)];assert pred.shape==truth.shape and losses[i]==metrics(pred,truth)["MSE"]
    assert a["picks"]==actions(losses,{i:(bank[str(i)]["parameters"],bank[str(i)]["operations"]) for i in ids})
    return a

def seal(cases):
    from msa_dti.prefix import DTYPE
    actions_by_case={}
    for c in cases:
        slug=c["slug"];p=E/"inputs/prefix"/(slug+".bin");assert sha(p)==c["prefix_hash"]
        raw=np.fromfile(p,dtype=DTYPE);h=c["H"];starts=np.arange(c["bounds"]["validation"][0]+95,c["T"]+1-h)
        cpu=raw["raw"][starts[:,None]+np.arange(1,h+1),3]
        y=(cpu[np.all(np.isfinite(cpu)&(cpu>=0),axis=1)]/100).astype(np.float32)
        assert len(y)>=c["B"]
        for tier,ids in budgets(h):verified_action(c,tier,ids,y)
        actions_by_case[slug]=sha(E/"results"/slug/"selected.json")
    assert len(actions_by_case)==120
    publish(GLOBAL,dict(actions=actions_by_case,signature=sha(E/"run_lock.json"),future_opened=False,decisions=360))

def metrics(pred,y):
    assert pred.shape==y.shape and len(y)
    err=pred.astype(np.float64)-y.astype(np.float64);loss=np.square(err).mean(1)
    assert np.isfinite(loss).all()
    return dict(MSE=float(loss.mean()),MAE=float(np.abs(err).mean()),Worst10=float(np.sort(loss)[-max(1,round(.1*len(loss))):].mean()))

def score(c,progress):
    import compute as u
    from core.methods.ours.stage2_runtime import candidate_backend_context
    slug=c["slug"];f=E/"results"/slug
    if (f/"scores.json").exists():return
    seal_doc=read(GLOBAL);assert seal_doc["signature"]==sha(E/"run_lock.json") and len(seal_doc["actions"])==120
    assert sha(f/"selected.json")==seal_doc["actions"][slug]
    src=E/"inputs/future"/(slug+".bin")
    if not src.exists():
        origin=LEGACY/"data/future/evaluation"/slug/"future.bin";assert sha(origin)==c["future_hash"]
        tmp=src.with_suffix(".pending");tmp.parent.mkdir(parents=True,exist_ok=True);tmp.write_bytes(origin.read_bytes());os.replace(tmp,src)
    assert sha(src)==c["future_hash"]
    raw=np.fromfile(src,dtype=u.DTYPE);t,h=c["T"],c["H"];assert len(raw)==t+c["B"]+h
    assert raw[:t+1].tobytes()==(E/"inputs/prefix"/(slug+".bin")).read_bytes()
    origins=np.arange(t,t+c["B"]);ix=origins[:,None]+np.arange(1,h+1)
    cpu=raw["raw"][ix,3];valid=np.all(np.isfinite(cpu)&(cpu>=0),axis=1);y=(cpu[valid]/100).astype(np.float32)
    feature=u.causal_features(raw["timestamp"],raw["raw"],read(E/"inputs/normalization.json"))
    x=u.torch.from_numpy(np.stack([feature[o-95:o+1] for o in origins]).astype(np.float32)).to(u.DEVICE)
    preds={};results=[]
    for n,(tier,ids) in enumerate(budgets(h),1):
        a=verified_action(c,tier,ids);states=u.torch.load(f/tier/"models.pt",map_location="cpu",weights_only=True)
        assert set(states)==set(a["picks"].values())
        stats={}
        for i,st in states.items():
            model=u.model(i,h,c["seed"],st);model.eval()
            assert u.state_hash(model)==next(r["state_sha256"] for r in a["rows"] if r["arch"]==i)
            with candidate_backend_context(u.ARCH[i],u.DEVICE,"gru-native"),u.torch.no_grad():pred=model(x).cpu().numpy()[valid]
            preds[f"{tier}_{i}"]=pred;stats[i]=metrics(pred,y)
        for method,i in a["picks"].items():
            results.append(dict(VM=c["VM"],H=h,K=c["K"],budget=tier,method=method,arch=i,feasible=len(ids),n_valid=len(y),**stats[i]))
        progress(n,3)
    panel(f/"forecasts.npz",preds);u.save(f/"scores.json",dict(global_sha256=sha(GLOBAL),future_sha256=sha(src),panel_sha256=sha(f/"forecasts.npz"),rows=results))

def summarize(cases):
    from msa_dti.prefix import DTYPE
    rows=[];global_sha=sha(GLOBAL)
    for c in cases:
        folder=E/"results"/c["slug"];j=read(folder/"scores.json")
        src=E/"inputs/future"/(c["slug"]+".bin");assert sha(src)==c["future_hash"]==j["future_sha256"]
        assert j["global_sha256"]==global_sha and sha(folder/"forecasts.npz")==j["panel_sha256"]
        h,t=c["H"],c["T"];raw=np.fromfile(src,dtype=DTYPE)
        y=raw["raw"][np.arange(t,t+c["B"])[:,None]+np.arange(1,h+1),3];y=(y[np.all(np.isfinite(y)&(y>=0),axis=1)]/100).astype(np.float32)
        with np.load(folder/"forecasts.npz",allow_pickle=False) as z:
            for tier,ids in budgets(h):
                a=verified_action(c,tier,ids);rr=[r for r in j["rows"] if r["budget"]==tier]
                assert len(rr)==3 and {r["method"]:r["arch"] for r in rr}==a["picks"]
                for r in rr:
                    assert (r["VM"],r["H"],r["K"],r["n_valid"])==(c["VM"],h,c["K"],len(y))
                    assert all(r[k]==v for k,v in metrics(z[f"{tier}_{r['arch']}"],y).items())
                ref=next(r for r in rr if r["method"]=="reference_only")
                for r in rr:
                    sw=r["arch"]!=ref["arch"];rows.append(dict(**r,switch=int(sw),harm=int(sw and r["MSE"]>ref["MSE"]),benefit=int(sw and r["MSE"]<ref["MSE"])))
    assert len(rows)==1080
    summary=[]
    for tier in inputs()[0]["budgets"]:
        for method in inputs()[0]["actions"]:
            group=[r for r in rows if r["budget"]==tier and r["method"]==method]
            vms={r["VM"] for r in group};assert len(group)==120 and len(vms)==30
            def mean(key):return statistics.mean(statistics.mean(r[key] for r in group if r["VM"]==vm) for vm in sorted(vms))
            summary.append(dict(budget=tier,method=method,**{k:mean(k) for k in ("MSE","MAE","Worst10","switch","harm","benefit")}))
    buffer=io.StringIO(newline="");w=csv.DictWriter(buffer,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    emit(E/"outputs/task_metrics.csv",buffer.getvalue().encode("utf8"))
    publish(FINAL,dict(status="PASS",VMs=30,tasks=120,budget_runs=360,physical_adaptations=120*55,global_sha256=global_sha,summary=summary))
