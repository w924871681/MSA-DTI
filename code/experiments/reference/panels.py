"""Task computations for the new-bank supplementary experiments."""
from pathlib import Path
import hashlib,json,os,sys,time
from functools import lru_cache
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
C=ROOT/"timing_s50_flow_20261104/C_real"
A=ROOT/"timing_s50_ab_20261104/A_fresh/native_replay/inputs"
B=ROOT/"timing_s50_ab_20261104/B_engine/cohort"
D=C/"fresh_D_stage/inputs"
E=C/"fresh_E_stage/inputs"
DIR={"A":A,"B":B,"C":C/"inputs","D":D,"E":E}
OUT=HERE/"panels"
CURRENT="n20_s2904"
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[k]="1"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+f".{os.getpid()}.pending")
    temp.write_text(json.dumps(x,separators=(",",":"),allow_nan=False));os.replace(temp,p)
def outfile(var,stage,slug):return OUT/var/stage/(slug+".json")
@lru_cache(None)
def cases(stage):
    data=read(DIR[stage]/("tasks.json" if stage=="A" else "cases.json"))
    return data["cases"] if stage=="B" else data
@lru_cache(None)
def plan():return read(HERE/"pipeline_plan.json")
def setup(gpu,stage=None):
    global u,torch,np
    if gpu is not None:os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    if stage=="E":
        def fence(event,args):
            if event=="open" and isinstance(args[0],(str,bytes)):
                p=Path(os.fsdecode(args[0]))
                if "future" in p.parts or p.name in ("forecasts.npz","scores.json"):raise PermissionError("E prefix-only phase")
        sys.addaudithook(fence)
    sys.path.insert(0,str(C/"scripts"))
    import compute as u
    import numpy as np
    torch=u.torch;u.configure_stage2_runtime(u.DEVICE,"gru-native")
@lru_cache(None)
def resource(i,h):return tuple(map(int,u.profile_arch(u.ARCH[i],96,25,h)))
def feasible(i,h,tier):
    return all(a<=b for a,b in zip(resource(i,h),plan()["caps"][tier]))
@lru_cache(None)
def checkpoint(v,i,h):
    if v==CURRENT:
        p=ROOT/"timing_s50_source_20261104/weights"/f"strong_H{h}_A{i}.pt"
        assert sha(p)==plan()["original_weights"][f"{h}/{i}"]
    else:
        p=HERE/"sources"/v/f"H{h}_A{i}.pt"
        assert sha(p)==read(p.with_suffix(".json"))["sha256"]
    obj=torch.load(p,map_location="cpu",weights_only=True);return obj["state_dict"]
@lru_cache(None)
def norm(stage):return read(DIR[stage]/("source_normalization.json" if stage=="B" else "normalization.json"))
def inputs(stage,c):
    p=DIR[stage]/("prefixes" if stage in "AB" else "prefix")/(c["slug"]+".bin")
    digest=sha(p)
    expected=(a_hashes()["prefixes/"+p.name] if stage=="A" else c.get("prefix_sha256",c.get("prefix_hash")))
    assert digest==expected
    d=u.prefix_selection(c,p,norm(stage))
    return p,d,[torch.from_numpy(d[k]).to(u.DEVICE).contiguous() for k in ("Xs","ys","Xv")]
@lru_cache(None)
def a_hashes():return read(A/"sha256.json")
def future(stage,c,p,d):
    h,t,b=c["H"],c["T"],c["B"];slug=c["slug"]
    if stage in ("C","D"):
        u.D=DIR[stage]
        return u.assess_inputs(c,d)
    f=DIR[stage]/("check" if stage=="B" else "future")/(slug+".bin")
    assert sha(f)==c.get("check_sha256",c.get("future_hash"))
    raw=f.read_bytes();assert raw.startswith(p.read_bytes())
    a=np.frombuffer(raw,dtype=u.DTYPE);assert len(a)==t+b+h
    origins=np.arange(t,t+b);y=a["raw"][origins[:,None]+np.arange(1,h+1),3];ok=np.all(np.isfinite(y)&(y>=0),axis=1)
    features=u.causal_features(a["timestamp"],a["raw"],norm(stage))
    x=np.stack([features[o-95:o+1] for o in origins]).astype(np.float32)
    return torch.from_numpy(x).to(u.DEVICE),(y[ok]/100).astype(np.float32),ok
def metrics(p,y):
    assert p.shape==y.shape and len(y) and np.isfinite(p).all()
    error=p.astype(np.float64)-y.astype(np.float64);loss=np.square(error).mean(1)
    return [float(loss.mean()),float(abs(error).mean()),float(np.sort(loss)[-max(1,round(len(y)*.1)):].mean())]
def fits(v,i,c,tensors,steps,optimizer):
    h,s=c["H"],c["seed"];xs,ys,xv=tensors
    m=u.model(i,h,s,checkpoint(v,i,h))
    with u.candidate_backend_context(u.ARCH[i],u.DEVICE,"gru-native"):
        u.seed(s)
        opt=torch.optim.SGD(m.parameters(),lr=.01) if optimizer=="sgd" else torch.optim.Adam(m.parameters(),lr=.001)
        for n in range(max(steps)+1):
            if n in steps:
                m.eval()
                with torch.no_grad():pred=m(xv).cpu().numpy()
                yield n,pred,m
            if n<max(steps):u.step(m,opt,xs,ys,1.)
def config(v,stage):
    if stage=="A":return list(range(66))
    if stage=="B":return list(range(66))
    if v==CURRENT:return plan()["bank"]
    return read(OUT/v/"bank.json")["bank"]
def task(v,stage,c):
    dest=outfile(v,stage,c["slug"])
    if dest.is_file():return
    h=c["H"];ids=config(v,stage);p,d,tensors=inputs(stage,c)
    if stage=="B":ids=[i for i in ids if feasible(i,h,c["tier"])]
    if stage=="C":ids=[i for i in ids if all(feasible(i,hh,"tight") for hh in (1,4))]
    if stage in ("D","E") and v!=CURRENT:ids=[i for i in ids if feasible(i,h,"medium")]
    # In the original-bank diagnostic, record every source checkpoint once to allow
    # comparisons under either individual cap and the nested medium-budget subsets.
    schedules={"sgd":[50]}
    if stage in ("D","E") and v==CURRENT:schedules={"sgd":plan()["steps"],"adam":[50]}
    validation={};states={}
    for opt,steps in schedules.items():
        for i in ids:
            for n,pred,m in fits(v,i,c,tensors,steps,opt):
                tag=f"{opt}{n}";validation.setdefault(tag,{})[i]=u.score(pred,d["yv"])
                if stage in ("B","C","D"):states[tag,i]=u.cpu_state(m)
    record=dict(VM=c["VM"],H=h,K=c["K"],slug=c["slug"],prefix=sha(p),validation=validation)
    if stage in ("B","C","D"):
        seal=dest.with_suffix(".prefix.json");save(seal,record)
        x,y,valid=future(stage,c,p,d);assessment={}
        for (tag,i),st in states.items():
            m=u.model(i,h,c["seed"],st);m.eval()
            with u.candidate_backend_context(u.ARCH[i],u.DEVICE,"gru-native"),torch.no_grad():pred=m(x).cpu().numpy()[valid]
            assessment.setdefault(tag,{})[i]=u.score(pred,y)
        record.update(assessment=assessment,prefix_seal=sha(seal))
    save(dest,record)
def score_task(v,c):
    dest=outfile(v,"scores",c["slug"])
    if dest.exists():return
    sealed=read(OUT/v/"E_seal.json");source=outfile(v,"E",c["slug"]);assert sha(source)==sealed["prefixes"][c["slug"]]
    choices=sealed["actions"][c["slug"]]
    p,d,tensors=inputs("E",c);data=read(source)
    x,y,valid=future("E",c,p,d);needed={}
    for spec in choices.values():
        for i in (spec["arch"],spec["ref"]):
            needed.setdefault((spec["optimizer"],i),set()).add(spec["steps"])
    predictions={};stats={}
    for (opt,i),ss in needed.items():
        for n,val,m in fits(v,i,c,tensors,ss,opt):
            tag=f"{opt}{n}";assert u.score(val,d["yv"])==data["validation"][tag][str(i)]
            with torch.no_grad():pred=m(x).cpu().numpy()[valid]
            key=f"{tag}_A{i}";predictions[key]=pred;stats[key]=metrics(pred,y)
    panel=dest.with_suffix(".npz");panel.parent.mkdir(exist_ok=True,parents=True)
    with panel.with_suffix(".pending").open("wb") as f:np.savez_compressed(f,**predictions)
    os.replace(panel.with_suffix(".pending"),panel)
    save(dest,dict(VM=c["VM"],H=c["H"],K=c["K"],stats=stats,panel=sha(panel),future=c["future_hash"],seal=sha(OUT/v/"E_seal.json")))

def hardware():
    import platform
    import subprocess
    setup(0)
    folder=OUT/"hardware";folder.mkdir(parents=True,exist_ok=True)
    cpu=next((line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),platform.machine())
    save(folder/"system.json",dict(cpu=cpu,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,threads=torch.get_num_threads(),hardware=plan()["hardware"]))
    for device in ("cpu","cuda"):
        dev=torch.device(device);x=torch.zeros((1,96,25),device=dev)
        for h in (1,4):
            for i in plan()["bank"]:
                file=folder/f"latency_{device}_H{h}_A{i}.json"
                if file.is_file():continue
                m=u.model(i,h,2904,checkpoint(CURRENT,i,h)).to(dev).eval()
                sync=torch.cuda.synchronize if device=="cuda" else lambda:None
                batches=[]
                with torch.inference_mode(),u.candidate_backend_context(u.ARCH[i],dev,"gru-native"):
                    for _ in range(100):m(x)
                    sync()
                    for _ in range(5):
                        values=[]
                        for _ in range(1000):
                            sync();start=time.perf_counter_ns();m(x);sync()
                            values.append(time.perf_counter_ns()-start)
                        batches.append(values)
                save(file,dict(arch=i,H=h,device=device,warmup=100,unit="ns",batches=batches))
    vm=min({c["VM"] for c in cases("D")},key=lambda x:int(x.split(":")[-1]))
    for c in [r for r in cases("D") if r["VM"]==vm]:
        p,d,tensors=inputs("D",c);h=c["H"]
        for tier in ("tight","medium","loose"):
            for method in ("msa","pt"):
                dest=folder/f"time_{c['slug']}_{tier}_{method}.json"
                if dest.exists():continue
                measurements=[];ch=[]
                for rep in range(6):
                    torch.cuda.synchronize();start=time.perf_counter_ns()
                    ids=[i for i in plan()["bank"] if feasible(i,h,tier)] if method=="msa" else [39]
                    loss={}
                    for i in ids:
                        f=(ROOT/"timing_s50_source_20261104/weights"/f"strong_H{h}_A{i}.pt" if method=="msa" else ROOT/"budget_fair_20261105/inputs/pt"/f"H{h}"/f"A{i}.pt")
                        st=torch.load(f,map_location="cpu",weights_only=True)
                        m=u.model(i,h,c["seed"],st.get("state_dict",st))
                        with u.candidate_backend_context(u.ARCH[i],u.DEVICE,"gru-native"):
                            u.seed(c["seed"]);opt=torch.optim.SGD(m.parameters(),lr=.01)
                            for _ in range(50):u.step(m,opt,tensors[0],tensors[1],1.)
                            m.eval()
                            with torch.no_grad():pred=m(tensors[2]).cpu().numpy()
                        loss[i]=u.score(pred,d["yv"])
                    if method=="msa":
                        choice=min((i for i in ids if i==57 or loss[i]<=.8*loss[57]),key=lambda i:(loss[i],*resource(i,h),i))
                    else:choice=39
                    torch.cuda.synchronize();elapsed=(time.perf_counter_ns()-start)/1e9
                    if rep:measurements.append(elapsed);ch.append(choice)
                assert len(set(ch))==1
                save(dest,dict(VM=vm,H=h,K=c["K"],budget=tier,method=method,selected=ch[0],feasible=feasible(ch[0],h,tier),times_seconds=measurements,adapted=len(ids)))
