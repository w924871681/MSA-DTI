"""Azure medium-budget search with one-task, three-budget validation controls."""
import concurrent.futures as cf
import csv
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE / "azure_run"
ROOT.mkdir(exist_ok=True)
AZ = ROOT / "azure_inputs"
I = HERE.parent / "budget_fair_20261105/inputs"
TIERS = ("tight", "medium", "loose")
GPUS = {"meta_nas_lite": 0, "zero_nas": 1, "zero_nas_ft": 2}
WORKERS = 12
MEMO = {}
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def read(p):
    return json.loads(p.read_bytes())


def save(p, doc):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".pending")
    tmp.write_text(json.dumps(doc, allow_nan=False), encoding="utf8")
    os.replace(tmp, p)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def unpack():
    AZ.mkdir(exist_ok=True)
    with zipfile.ZipFile(HERE / "pilot_controls.zip") as z:
        manifest = json.loads(z.read("manifest.json"))
        for n, digest in manifest.items():
            p = (AZ/n).resolve()
            assert AZ in p.parents
            if p.is_file() and sha(p) == digest:
                continue
            raw = z.read(n)
            assert hashlib.sha256(raw).hexdigest() == digest
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)


def setup(selection):
    sys.path.insert(0, str(I/"native/src"))
    from msa_dti import runtime as u
    from msa_dti.prefix import prefix_selection, DTYPE
    from msa_dti.features import causal_features
    MEMO.update(u=u, prefix=prefix_selection, dtype=DTYPE, features=causal_features,
                norm=read(AZ/"normalization.json"), caps=read(I/"protocol.json")["budgets"],
                rank=list(csv.DictReader((I/"pt_rank.csv").open(encoding="utf-8-sig"))),
                cases={x["slug"]:x for x in read(AZ/"cases.json")}, banks={})
    if selection:
        def fence(event,args):
            if event == "open" and isinstance(args[0], (str, bytes)) and (AZ/"future") in Path(os.fsdecode(args[0])).resolve().parents:
                raise PermissionError("Future label access before sealing")
        sys.addaudithook(fence)


def select(method, tier, slug, smoke):
    u = MEMO["u"]
    import numpy as np
    c = MEMO["cases"][slug]
    h, seed = c["H"], c["seed"]
    prefix = AZ/"prefix"/(slug+".bin")
    assert sha(prefix) == c["prefix_hash"]
    d = MEMO["prefix"](c,prefix,MEMO["norm"])
    xs,ys,xv,yv = [u.torch.from_numpy(d[k]) for k in ("Xs","ys","Xv","yv")]
    cap = MEMO["caps"][tier]
    feasible = [i for i,a in enumerate(u.ARCH) if all(x<=lim for x,lim in zip(u.profile_arch(a,96,25,h),cap))]
    native = u.native.feasible_indices(u.CFG,u.ARCH,tier,96,25,h)
    assert feasible == native and len(native) == {"tight":26,"medium":48,"loose":66}[tier]
    if method == "pt":
        i = next(int(r["arch_id"]) for r in MEMO["rank"] if int(r["max_parameters"])<=cap[0] and int(r["max_operations"])<=cap[1])
        assert i == (57 if tier=="tight" else 39) and i in feasible
        cp = u.torch.load(I/f"pt/H{h}/A{i}.pt",map_location="cpu",weights_only=True)
        m = u.build(i,h,cp.get("state_dict",cp),seed)
        with u.candidate_backend_context(u.ARCH[i],u.DEVICE,"gru-native"):
            u.seed(seed)
            opt=u.torch.optim.SGD(m.parameters(),lr=.01)
            xx,yy=xs.to(u.DEVICE).contiguous(),ys.to(u.DEVICE).contiguous()
            for _ in range(50):
                m.train()
                opt.zero_grad(set_to_none=True)
                ((m(xx)-yy)**2).mean().backward()
                u.torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
                opt.step()
        shortlist=[i]
    else:
        if h not in MEMO["banks"]:
            MEMO["banks"][h]={u.make_bank_key_shared(H=h,arch_key=a.arch_key,input_dim=25,L=96):
                u.torch.load(I/f"weights/H{h}/A{j}.pt",map_location="cpu",weights_only=True)["state_dict"] for j,a in enumerate(u.ARCH)}
        calls=[]
        original=u.native.feasible_indices
        def observed(cfg,arches,requested,*args):
            choices=original(cfg,arches,requested,*args)
            assert requested==tier and choices==feasible
            calls.append(requested)
            return choices
        u.native.feasible_indices=observed
        try:
            m,_,row,details,_,_=u.native._run_search_case(method,{"bank":MEMO["banks"][h],"pi50":read(I/"pi50.json")["pi50"] if method=="meta_nas_lite" else None},u.CFG,u.ARCH,u.DEVICE,"gru-native",xs,ys,xv,yv,tier,h,25,96,seed)
        finally:
            u.native.feasible_indices=original
        i=int(row["arch_idx"]);shortlist=details["shortlist"]
        assert calls==[tier] and set(shortlist)<=set(feasible) and i in shortlist
    out=ROOT/("pilot" if smoke else "batch")/method/tier/slug
    out.mkdir(parents=True,exist_ok=True)
    m.eval()
    with u.candidate_backend_context(u.ARCH[i],u.DEVICE,"gru-native"),u.torch.no_grad():
        val=m(xv.to(u.DEVICE).contiguous()).cpu().numpy()
    if not np.isfinite(val).all():raise ValueError("Nonfinite validation")
    if smoke and tier=="loose":
        ref=read(AZ/f"control/{method}.json")
        with np.load(AZ/f"control/{method}.npz") as panel:
            assert i==ref["arch"] and np.array_equal(val,panel["prediction"]), (method,i,ref["arch"],float(np.max(abs(val-panel["prediction"]))))
    u.torch.save({k:v.detach().cpu().clone() for k,v in m.state_dict().items()},out/"model.pt")
    par,ops=u.profile_arch(u.ARCH[i],96,25,h)
    rec=dict(method=method,tier=tier,slug=slug,arch=i,parameters=int(par),operations=int(ops),
             prefix_sha256=c["prefix_hash"],eligible=feasible,shortlist=shortlist,model_sha256=sha(out/"model.pt"))
    save(out/"selected.json",rec)
    return sha(out/"selected.json")


def score(method,tier,slug):
    import numpy as np
    u=MEMO["u"]; c=MEMO["cases"][slug]; h,t=c["H"],c["T"]
    out=ROOT/"batch"/method/tier/slug
    assert read(ROOT/"SEAL.json")[method][tier+"/"+slug]==sha(out/"selected.json")
    pick=read(out/"selected.json");assert sha(out/"model.pt")==pick["model_sha256"]
    f=AZ/"future"/(slug+".bin"); assert sha(f)==c["future_hash"]
    raw=np.fromfile(f,dtype=MEMO["dtype"]);assert len(raw)==t+c["B"]+h and raw[:t+1].tobytes()==(AZ/"prefix"/(slug+".bin")).read_bytes()
    origins=np.arange(t,t+c["B"]);cpu=raw["raw"][origins[:,None]+np.arange(1,h+1),3]
    valid=np.all(np.isfinite(cpu)&(cpu>=0),axis=1); y=(cpu[valid]/100).astype(np.float32)
    feats=MEMO["features"](raw["timestamp"],raw["raw"],MEMO["norm"])
    x=u.torch.from_numpy(np.stack([feats[o-95:o+1] for o in origins]).astype(np.float32)).to(u.DEVICE)
    m=u.build(pick["arch"],h,u.torch.load(out/"model.pt",weights_only=True,map_location="cpu"),c["seed"])
    pred=u.pred(m,x,pick["arch"])[valid]; assert pred.shape==y.shape and np.isfinite(pred).all()
    np.savez_compressed(out/"forecast.npz",prediction=pred)
    err=pred.astype(np.float64)-y.astype(np.float64); loss=np.square(err).mean(axis=1)
    r=dict(weighted_mse=float(loss.mean()),mae=float(abs(err).mean()),worst10=float(np.sort(loss)[-max(1,round(.1*len(loss))):].mean()))
    save(out/"score.json",dict(method=method,tier=tier,slug=slug,VM=c["VM"],H=h,K=c["K"],arch=pick["arch"],
         MSE=r["weighted_mse"],MAE=r["mae"],Worst10=r["worst10"],forecast_sha256=sha(out/"forecast.npz"),future_sha256=c["future_hash"]))


def lane(method,mode):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(GPUS[method])
    cases=read(AZ/"cases.json")
    pilot=mode=="pilot"
    tiers=TIERS if pilot else ("medium",)
    tasks=[(t,c["slug"]) for t in tiers for c in (cases[:1] if pilot else cases)]
    start=time.time();total=len(tasks);done=0
    save(ROOT/f"{mode}_{method}.json",dict(done=0,total=total,start=start))
    with cf.ProcessPoolExecutor(max_workers=1 if pilot else WORKERS,mp_context=mp.get_context("spawn"),initializer=setup,initargs=(mode!="score",)) as pool:
        futures=[pool.submit(score,method,t,s) if mode=="score" else pool.submit(select,method,t,s,pilot) for t,s in tasks]
        for task in cf.as_completed(futures):
            task.result();done+=1
            save(ROOT/f"{mode}_{method}.json",dict(done=done,total=total,start=start,updated=time.time()))
    save(ROOT/f"{mode}_{method}.json",dict(done=done,total=total,start=start,end=time.time()))


def phase(mode):
    save(HERE/"azure_status.json",{"phase":mode,"updated":time.time()})
    procs=[]
    try:
        for method in GPUS:
            with (ROOT/f"{mode}_{method}.log").open("ab") as log:
                procs.append(subprocess.Popen([sys.executable,"-B",__file__,"lane",method,mode],stdout=log,stderr=log))
        while any(p.poll() is None for p in procs):
            if any(p.poll() not in (0,None) for p in procs):raise RuntimeError(mode+" lane failed")
            if shutil.disk_usage(ROOT).free<1.5*1024**3:raise RuntimeError("disk guard below 1.5 GiB")
            save(HERE/"azure_status.json",dict(phase=mode, progress={m:read(ROOT/f"{mode}_{m}.json")["done"] for m in GPUS if (ROOT/f"{mode}_{m}.json").exists()}))
            time.sleep(5)
        if any(p.returncode for p in procs):raise RuntimeError(mode+" lane failed")
    finally:
        for p in procs:
            if p.poll() is None:p.terminate()


def supervise():
    with (ROOT/"lock").open("a+") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            unpack()
            for gpu in GPUS.values():
                if subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-compute-apps=pid","--format=csv,noheader"]).strip(): raise RuntimeError("Assigned GPU occupied")
            phase("pilot")
            save(ROOT/"PILOT_PASS.json",{"case":read(AZ/"cases.json")[0]["slug"],"methods":list(GPUS),"tiers":TIERS,"loose_validation_controls":"exact arrays"})
            phase("select")
            seals={m:{p.parent.parent.name+"/"+p.parent.name:sha(p) for p in (ROOT/"batch"/m).glob("*/*/selected.json")} for m in GPUS}
            assert all(len(s)==120 for m,s in seals.items())
            save(ROOT/"SEAL.json",seals)
            phase("score")
            with tarfile.open(ROOT/"result_bundle.partial","w") as a:
                for p in sorted(ROOT.glob("*.json")):
                    a.add(p,arcname=p.name)
                for folder in ("pilot","batch"):
                    for p in (ROOT/folder).rglob("*"):
                        if p.is_file() and p.suffix in (".json",".npz"):a.add(p,arcname=p.relative_to(ROOT))
            os.replace(ROOT/"result_bundle.partial",HERE/"azure_results.tar")
            save(HERE/"azure_status.json",{"phase":"complete","sha256":sha(HERE/"azure_results.tar")})
        except BaseException:
            save(HERE/"azure_status.json",{"phase":"failed","error":traceback.format_exc(),"lane_errors":{p.name:p.read_text(errors="replace")[-1800:] for p in ROOT.glob("*.log") if p.stat().st_size and p.name!="supervisor.log"}})
            raise


if __name__=="__main__":
    if sys.argv[1]=="lane":lane(*sys.argv[2:])
    elif sys.argv[1]=="supervise":supervise()
    elif sys.argv[1]=="dispatch":
        if (HERE/"azure_status.json").exists():raise RuntimeError("Existing experiment; inspect status before retrying")
        save(HERE/"azure_status.json",{"phase":"starting"})
        with (ROOT/"supervisor.log").open("ab") as log:
            subprocess.Popen([sys.executable,"-B",__file__,"supervise"],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    else:
        status=read(HERE/"azure_status.json");print(status)
        phase_name=status["phase"]
        for method in GPUS:
            for mode in ([phase_name] if phase_name!="failed" else ["pilot","select","score"]):
                p=ROOT/f"{mode}_{method}.json"
                if p.is_file():print(method,mode,read(p))
                log=ROOT/f"{mode}_{method}.log"
                if phase_name=="failed" and log.is_file() and log.stat().st_size:print(log.read_text(errors="replace")[-2400:])
