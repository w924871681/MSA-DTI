"""Matched E-stage ablations. Every score waits for 120 prefix-only decisions."""
import concurrent.futures as cf, hashlib, json, multiprocessing as mp, os, subprocess, sys, tarfile, time, traceback, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/"timing_s50_flow_20261104/C_real"
E=BASE/"fresh_E_stage";SRC=HERE.parent/"timing_s50_source_20261104/weights"
OUT=HERE/"module_run";OUT.mkdir(exist_ok=True)
STATUS=HERE/"modules_status.json"
for k in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]:os.environ[k]="1"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
sys.path.insert(0,str(BASE/"scripts"))
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    tmp=p.with_suffix(".pending");tmp.write_text(json.dumps(v,allow_nan=False));os.replace(tmp,p)
def config():
    with zipfile.ZipFile(HERE/"module_inputs.zip") as z:return json.loads(z.read("protocol.json"))
def init(mode):
    global u,conf,bank,norm,torch,np,source,seal,archive_seal
    if mode=="select":
        def fence(event,args):
            if event=="open" and isinstance(args[0],(str,bytes)):
                p=Path(os.fsdecode(args[0]));parts=p.parts
                if "future" in parts or p.name in ("scores.json","forecasts.npz"):raise PermissionError("Outcome access before selection")
        sys.addaudithook(fence)
    import compute as u
    import numpy as np
    torch=u.torch
    u.configure_stage2_runtime(u.DEVICE,"gru-native")
    conf=config();bank=read(E/"inputs/bank.json");norm=read(E/"inputs/normalization.json");source={}
    archive_seal=read(E/"outputs/all_actions.json")["actions"]
    if mode=="score":seal=read(OUT/"seal.json")["decisions"]
def init_model(i,h,seed):
    if (i,h) not in source:
        f=SRC/f"strong_H{h}_A{i}.pt";assert sha(f)==conf["weights"][f"{h}/{i}"]
        obj=torch.load(f,map_location="cpu",weights_only=True);source[i,h]=obj.get("state_dict",obj)
    return u.model(i,h,seed,source[i,h])
def fit(i,h,s,xs,ys,xv,yv,steps):
    m=init_model(i,h,s)
    with u.candidate_backend_context(u.ARCH[i],u.DEVICE,"gru-native"):
        if steps:
            u.seed(s);opt=torch.optim.SGD(m.parameters(),lr=.01)
            for _ in range(steps):u.step(m,opt,xs,ys,1.)
        m.eval()
        with torch.no_grad():pred=m(xv).cpu().numpy()
    assert np.all(np.isfinite(pred))
    return u.score(pred,yv),m
def select(c,pilot):
    s=c["slug"];out=OUT/s;out.mkdir(exist_ok=True);h=c["H"];seed=c["seed"]
    orig=E/"results"/s;assert sha(orig/"selected.json")==archive_seal[s]
    assert sha(orig/"medium/actions.json")==read(orig/"selected.json")["actions"]["medium"]
    a=read(orig/"medium/actions.json")
    old={r["arch"]:r for r in a["rows"]}; ids=sorted(old)
    prof={i:u.profile_arch(v,96,25,h) for i,v in enumerate(u.ARCH)}
    cap=read(E/"protocol.json")["budgets"]["medium"];all_ids=[i for i in prof if all(v<=lim for v,lim in zip(prof[i],cap))]
    assert len(all_ids)==48 and len(ids)==17 and set(ids)<set(all_ids) and 57 in ids
    prefix=E/"inputs/prefix"/(s+".bin");assert sha(prefix)==c["prefix_hash"]
    ds=u.prefix_selection(c,prefix,norm);xs,ys,xv=[torch.from_numpy(ds[k]).to(u.DEVICE).contiguous() for k in ("Xs","ys","Xv")]
    if pilot:
        loss,m=fit(57,h,seed,xs,ys,xv,ds["yv"],50)
        assert loss==old[57]["validation"] and u.state_hash(m)==old[57]["state_sha256"]
    archived=orig/"medium/models.pt";assert sha(archived)==a["models_sha256"]
    states=torch.load(archived,map_location="cpu",weights_only=True)
    def key(loss,i):return (loss,*prof[i],i)
    def decide(losses):
        ref=conf["reference"];tau=conf["tau"];ok=[i for i in losses if i==ref or losses[i]<=(1-tau)*losses[ref]]
        return min(ok,key=lambda i:key(losses[i],i))
    los={i:r["validation"] for i,r in old.items()};assert decide(los)==a["picks"]["msa"]
    best=min(los,key=lambda i:key(los[i],i));beststate=states[best]
    for i in all_ids:
        if i in old:continue
        loss,m=fit(i,h,seed,xs,ys,xv,ds["yv"],50);los[i]=loss
        if key(loss,i)<key(los[best],best):best=i;beststate=u.cpu_state(m)
    pick=decide(los);assert pick in (best,57)
    models={"no_preselection":(pick,beststate if pick==best else states[57])}
    u.save_pt(out/"models.pt",{m:state for m,(_,state) in models.items()})
    save(out/"selected.json",dict(tier="medium",slug=s,prefix_hash=c["prefix_hash"],picks={k:i for k,(i,_) in models.items()},full=a["picks"]["msa"],validation={"no_preselection":los},eligible=all_ids,bank_eligible=ids,models_sha=sha(out/"models.pt"),a57_replay=pilot))
def score(c,pilot):
    s=c["slug"];out=OUT/s;sel=read(out/"selected.json");assert sha(out/"selected.json")==seal[s] and sha(out/"models.pt")==sel["models_sha"]
    t,h=c["T"],c["H"];f=E/"inputs/future"/(s+".bin");assert sha(f)==c["future_hash"]
    raw=np.fromfile(f,dtype=u.DTYPE);assert len(raw)==t+c["B"]+h and hashlib.sha256(raw[:t+1].tobytes()).hexdigest()==c["prefix_hash"]
    orig=np.arange(t,t+c["B"]);y=raw["raw"][orig[:,None]+np.arange(1,h+1),3];valid=np.all(np.isfinite(y)&(y>=0),axis=1)
    feats=u.causal_features(raw["timestamp"],raw["raw"],norm)
    x=torch.from_numpy(np.stack([feats[o-95:o+1] for o in orig]).astype(np.float32)).to(u.DEVICE)
    states=torch.load(out/"models.pt",map_location="cpu",weights_only=True)
    predictions={}
    for method in ("no_preselection",):
        i=sel["picks"][method]
        m=u.model(i,h,c["seed"],states[method]);m.eval()
        with u.candidate_backend_context(u.ARCH[i],u.DEVICE,"gru-native"),torch.no_grad():v=m(x).cpu().numpy()[valid]
        assert v.shape==y[valid].shape and np.isfinite(v).all();predictions[method]=v
    np.savez_compressed(out/"forecast.npz",**predictions)
    save(out/"score.json",dict(future_hash=c["future_hash"],panel_sha=sha(out/"forecast.npz")))
def lane(gpu,mode):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    cases=read(E/"inputs/cases.json")
    vms=sorted({c["VM"] for c in cases});assigned=[c for c in cases if vms.index(c["VM"])%4==gpu]
    fn=select if mode=="select" else score
    with cf.ProcessPoolExecutor(max_workers=10,initializer=init,initargs=(mode,),mp_context=mp.get_context("spawn")) as pool:
        pool.submit(fn,assigned[0],True).result();done=1
        save(OUT/f"{mode}_{gpu}.json",dict(done=done,total=len(assigned)))
        for f in cf.as_completed([pool.submit(fn,c,False) for c in assigned[1:]]):
            f.result();done+=1;save(OUT/f"{mode}_{gpu}.json",dict(done=done,total=len(assigned)))
def phase(mode):
    running={}
    while True:
        azure=read(HERE/"azure_status.json")["phase"]
        for gpu in range(4):
            if gpu in running or (gpu!=3 and azure!="complete"):continue
            if subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-compute-apps=pid","--format=csv,noheader"]).strip():continue
            with (OUT/f"{mode}_{gpu}.log").open("ab") as log:
                running[gpu]=subprocess.Popen([sys.executable,"-B",__file__,"lane",str(gpu),mode],stdout=log,stderr=log)
        for gpu,p in running.items():
            if p.poll() not in (0,None):raise RuntimeError((OUT/f"{mode}_{gpu}.log").read_text(errors="replace")[-3000:])
        counts={str(g):read(OUT/f"{mode}_{g}.json")["done"] for g in running if (OUT/f"{mode}_{g}.json").exists()}
        save(STATUS,dict(phase=mode,gpus=list(running),done=sum(counts.values()),per_gpu=counts,total=120))
        if len(running)==4 and all(p.poll()==0 for p in running.values()):return
        time.sleep(8)
def run():
    try:
        conf=config()
        for n,h in conf["cases"].items():assert sha(E/n)==h,n
        assert sha(BASE/"scripts/compute.py")==conf["compute_sha"]
        phase("select")
        sel={p.parent.name:sha(p) for p in OUT.glob("*/selected.json")};assert len(sel)==120
        save(OUT/"seal.json",dict(decisions=sel,protocol_sha=sha(HERE/"module_inputs.zip"),timestamp=time.time()))
        phase("score")
        with tarfile.open(HERE/"modules_results.tar","w") as a:
            a.add(OUT/"seal.json",arcname="seal.json")
            for s in sel:
                for n in ("selected.json","forecast.npz","score.json"):a.add(OUT/s/n,arcname=s+"/"+n)
        save(STATUS,dict(phase="complete",done=120,sha256=sha(HERE/"modules_results.tar")))
    except BaseException:save(STATUS,dict(phase="failed",error=traceback.format_exc()));raise
if __name__=="__main__":
    if sys.argv[1]=="lane":lane(int(sys.argv[2]),sys.argv[3])
    elif sys.argv[1]=="dispatch":
        if STATUS.exists():raise RuntimeError("Existing module run")
        save(STATUS,dict(phase="starting"))
        with (HERE/"modules.log").open("ab") as log:subprocess.Popen([sys.executable,"-B",__file__,"run"],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    else:run()
