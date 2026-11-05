"""D calibration. The verified C default is the only reference. E stays reserved."""
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare import ROOT,read,sha

D=ROOT/"D_stage"


def choose(va,ref,tau,profile):
    allowed=[i for i in va if i==ref or va[i]<=(1-tau)*va[ref]]
    return min(allowed,key=lambda i:(va[i],*profile[i],i))


def run_case(c,cp):
    import compute as u
    import torch
    from core.space import profile_arch
    from core.methods.ours.stage2_runtime import candidate_backend_context
    protocol=read(D/"protocol.json");ref=cp["selected"]["arch_id"]
    assert cp["method"]=="msa" and ref in protocol["eligible"]
    slug,h,k,t,b,n=c["slug"],c["H"],c["K"],c["T"],c["B"],c["validation_count"]
    folder=D/"results"/slug
    if (folder/"scores.json").is_file():return
    weights=read(D/"inputs/weights.json");eligible=protocol["eligible"];old=u.D
    assert len(eligible)==20 and len(set(eligible))==20
    folder.mkdir(exist_ok=True,parents=True)
    try:
        u.D=D/"inputs"
        data=u.features(slug,h,k,t,b,c["seed"],n,c["prefix_sha256"])
        xs,ys,xv=(torch.from_numpy(data[name]).to(u.DEVICE).contiguous() for name in ("Xs","ys","Xv"))
        rows=[];states={};panel={};scores={};profile={}
        for i in eligible:
            pa,op=profile_arch(u.ARCH[i],96,25,h);assert pa<=100000 and op<=5000000
            path=ROOT/weights[f"msa/{h}/{i}"]["path"];assert sha(path)==weights[f"msa/{h}/{i}"]["sha256"]
            obj=torch.load(path,map_location="cpu",weights_only=True)
            m=u.model(i,h,c["seed"],obj.get("state_dict",obj))
            with candidate_backend_context(u.ARCH[i],u.DEVICE,"gru-native"):
                u.seed(c["seed"]);opt=torch.optim.SGD(m.parameters(),lr=.01)
                for _ in range(50):u.step(m,opt,xs,ys,1.)
                m.eval()
                with torch.no_grad():v=m(xv).cpu().numpy()
            scores[i]=u.score(v,data["yv"])
            if scores[i]<=0:raise ValueError("The reference selector requires positive validation MSE")
            profile[i]=(pa,op);states[i]=u.cpu_state(m)
            rows.append(dict(arch=i,validation_mse=scores[i],parameters=pa,operations=op,
                             source_sha256=weights[f"msa/{h}/{i}"]["sha256"],state_sha256=u.state_hash(m)))
            panel[f"v{i}"]=v
            u.save(ROOT/"results/progress"/f"d_{slug}.json",dict(phase="D验证",completed=len(rows),total=len(eligible),time=u.time.time()))
        picks=[dict(tau=tau,arch=choose(scores,ref,tau,profile)) for tau in protocol["grid"]]
        seal=folder/"validation_seal.json"
        u.save(seal,dict(slug=slug,reference=ref,reference_receipt_sha256=sha(ROOT/"outputs/msa_selected.json"),rows=rows,
                         choices=picks,prefix_sha256=c["prefix_sha256"],signature=sha(D/"run_lock.json")))
        # Assessment labels are accessed only after every feasible candidate and every margin decision have been sealed.
        xc,yc,valid=u.assess_inputs(c,data)
        later={}
        for i in eligible:
            m=u.model(i,h,c["seed"],states[i]);m.eval()
            with candidate_backend_context(u.ARCH[i],u.DEVICE,"gru-native"),torch.no_grad():
                f=m(xc).cpu().numpy()[valid]
            panel[f"a{i}"]=f;later[str(i)]=u.score(f,yc)
        tmp=folder/"predictions.pending"
        with tmp.open("wb") as out:np.savez_compressed(out,**panel)
        tmp.replace(folder/"predictions.npz")
        u.save(folder/"scores.json",dict(slug=slug,VM=c["VM"],H=h,K=k,assessment_mse=later,
               seal_sha256=sha(seal),panel_sha256=sha(folder/"predictions.npz"),assessment_sha256=c["post_sha256"]))
        (ROOT/"results/progress"/f"d_{slug}.json").unlink(missing_ok=True)
    finally:u.D=old


def summarize():
    protocol=read(D/"protocol.json"); tasks=read(D/"inputs/cases.json");default=read(ROOT/"outputs/msa_selected.json")
    ref=default["selected"]["arch_id"];gp=protocol["grid"];ids=protocol["eligible"]
    roles=read(D/"inputs/roles.json");vm=set(roles["vm_ids"]);assert len(vm)==20 and vm.isdisjoint(set().union(*roles["excluded"].values()))
    assert len(tasks)==80 and len({(c["VM"],c["H"],c["K"]) for c in tasks})==80
    totals={t:{} for t in gp};reference={}
    dt=np.dtype([("timestamp","<i8"),("raw","<f8",(10,))])
    for c in tasks:
        slug,h,k,t,b=(c[s] for s in ("slug","H","K","T","B"))
        folder=D/"results"/slug;meta=read(folder/"scores.json");sealed=read(folder/"validation_seal.json")
        assert sha(folder/"predictions.npz")==meta["panel_sha256"] and sha(folder/"validation_seal.json")==meta["seal_sha256"]
        assert sealed["reference"]==ref and sealed["reference_receipt_sha256"]==sha(ROOT/"outputs/msa_selected.json") and sealed["signature"]==sha(D/"run_lock.json")
        pre=D/"inputs/prefix"/(slug+".bin");post=D/"inputs/assessment"/(slug+".bin");assert sha(pre)==c["prefix_sha256"] and sha(post)==c["post_sha256"]
        raw=np.concatenate((np.fromfile(pre,dtype=dt),np.fromfile(post,dtype=dt)))
        truth=[]
        for origins in (np.arange(96+h+k-1+95,t-h+1),np.arange(t,t+b)):
            cpu=raw["raw"][origins[:,None]+np.arange(1,h+1),3]; keep=np.all(np.isfinite(cpu)&(cpu>=0),axis=1);truth.append((cpu[keep]/100).astype(np.float32))
        assert [len(q) for q in truth]==[c["validation_count"],c["assessment_count"]]
        with np.load(folder/"predictions.npz",allow_pickle=False) as preds:
            assert set(preds.files)=={q+str(i) for i in ids for q in "va"}
            def scores(stage,y):
                result={}
                for i in ids:
                    p=preds[f"{stage}{i}"];assert p.shape==y.shape
                    value=float(np.square(p.astype(np.float64)-y.astype(np.float64)).mean(1).mean());assert math.isfinite(value)
                    result[i]=value
                return result
            validation=scores("v",truth[0]);later=scores("a",truth[1])
        assert [r["arch"] for r in sealed["rows"]]==ids and all(validation[r["arch"]]==r["validation_mse"] for r in sealed["rows"])
        assert all(later[i]==meta["assessment_mse"][str(i)] for i in ids)
        prof={r["arch"]:(r["parameters"],r["operations"]) for r in sealed["rows"]}
        reference.setdefault(c["VM"],[]).append(later[ref])
        assert len(sealed["choices"])==len(gp)
        for pick,tau in zip(sealed["choices"],gp):
            i=choose(validation,ref,tau,prof);assert pick==dict(tau=tau,arch=i)
            totals[tau].setdefault(c["VM"],[]).append((later[i],int(i!=ref and later[i]>later[ref]),int(i!=ref)))
    assert all(len(v)==4 for v in reference.values())
    risk_ref=statistics.mean(statistics.mean(s) for s in reference.values());table=[]
    for tau,groups in totals.items():
        assert set(groups)==vm and all(len(g)==4 for g in groups.values())
        risk,harm,switch=(statistics.mean(statistics.mean(row[k] for row in g) for g in groups.values()) for k in range(3))
        table.append(dict(tau=tau,mse=risk,harm_rate=harm,switch_rate=switch))
    valid=[r for r in table if r["harm_rate"]<=protocol["harm_limit"]];best=min(valid,key=lambda r:(r["mse"],-r["tau"])) if valid else None
    use=bool(best and best["mse"]<risk_ref)
    result=dict(status="PASS",stage="D",budget="medium",reference=ref,policy="selector" if use else "pure_reference",
                tau=best["tau"] if use else None,reference_only_mse=risk_ref,trials=table,D_tasks=80,D_vms=20,
                candidate_task_pairs=80*len(ids),assessment_scope="calibration only; no E results",source_reference_sha256=sha(ROOT/"outputs/msa_selected.json"))
    output=D/"outputs";output.mkdir(exist_ok=True);p=output/"frozen_policy.json";tmp=p.with_suffix(".pending")
    tmp.write_bytes((json.dumps(result,indent=2,ensure_ascii=False)+"\n").encode());tmp.replace(p)
    print(result["policy"],result["tau"])


if __name__=="__main__":summarize()
