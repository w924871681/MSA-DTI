"""Run the prespecified source-pool/seed variants with the original 50-epoch recipe."""
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ORIGINAL=ROOT/"original"
DATA=ORIGINAL/"data/bitbrains/data/bitbrains_dataset.npz"
MANIFEST=DATA.with_name("bitbrains_dataset_manifest.json")
OUT=HERE/"sources"
OUT.mkdir(exist_ok=True)
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[k]="1"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
def read(p):return json.loads(p.read_bytes())
def save(p,obj):
    p.parent.mkdir(exist_ok=True,parents=True)
    tmp=p.with_name(p.name+f".{os.getpid()}.pending");tmp.write_text(json.dumps(obj,allow_nan=False));os.replace(tmp,p)
def sha(p):
    digest=hashlib.sha256()
    with p.open("rb") as stream:
        for b in iter(lambda:stream.read(2**20),b""):digest.update(b)
    return digest.hexdigest()
def init():
    global src,torch,ids,arrays
    os.environ["MSADTI_OUTPUT"]=str(ORIGINAL/"fresh_source_run")
    sys.path.insert(0,str(ORIGINAL/"src"))
    import numpy as np
    import torch
    from msa_dti import source_training as src
    ids=read(MANIFEST)["source_machine_ids"]
    arrays={}
    with np.load(DATA,allow_pickle=False) as archive:
        for h in (1,4):
            arrays[h]=[(torch.from_numpy(archive[f"X_source_{j}_H{h}"].copy()),torch.from_numpy(archive[f"y_source_{j}_H{h}"].copy())) for j in range(20)]
def train(job):
    pool,base_seed,i,h=job
    folder=OUT/f"n{pool}_s{base_seed}";folder.mkdir(exist_ok=True)
    p=folder/f"H{h}_A{i}.pt";receipt=p.with_suffix(".json")
    if receipt.exists():
        assert sha(p)==read(receipt)["sha256"];return
    seed=base_seed+101*h+i
    m=src.build(i,h,s=seed)
    with src.candidate_backend_context(src.ARCH[i],src.DEVICE,"gru-native"):
        opt=torch.optim.Adam(m.parameters(),lr=.001,weight_decay=0.,betas=(.9,.999),eps=1e-8,amsgrad=False)
        for epoch in range(50):
            order=list(range(pool));random.Random(seed+epoch).shuffle(order)
            for j in order:
                x,y=arrays[h][j];machine_seed=int(hashlib.sha256(ids[j].encode()).hexdigest()[:8],16)
                g=torch.Generator(device="cpu").manual_seed(seed+epoch+machine_seed)
                shuffle=torch.randperm(100,generator=g)
                for batch in (shuffle[:64],shuffle[64:]):
                    src.step(m,opt,x[batch],y[batch])
    tmp=p.with_suffix(".pending")
    src.save_torch(tmp,{"state_dict":src.cpu_state(m),"source_only":True,"seed":seed,"source_ids":ids[:pool]})
    os.replace(tmp,p)
    save(receipt,dict(sha256=sha(p),pool=pool,source_ids=ids[:pool],base_seed=base_seed,model_seed=seed,H=h,arch=i,epochs=50,updates=100*pool,visits=5000*pool))
def lane(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    with zipfile.ZipFile(HERE/"supp_inputs.zip") as z:plan=json.loads(z.read("plan.json"))
    jobs=[(n,s,i,h) for n,s in plan["variants"] for i in range(66) for h in (1,4)]
    # Assign equal numbers of architectures across lanes and both horizons to each lane.
    jobs=[j for k,j in enumerate(jobs) if k%4==gpu]
    with cf.ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context("spawn"),initializer=init) as p:
        for future in cf.as_completed([p.submit(train,j) for j in jobs]):future.result()
def run():
    import fcntl
    live={}
    try:
        assert sha(DATA)=="62f7b942fc584b2fc0e3894367cde6dfe1303d0f1a38ba3f70cc6de76196880c"
        assert len(set(read(MANIFEST)["source_machine_ids"]))==20
        with (OUT/"lock").open("a+") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            while True:
                for gpu in range(4):
                    if gpu in live:continue
                    if gpu<3 and read(HERE/"azure_status.json")["phase"]!="complete":continue
                    if subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-compute-apps=pid","--format=csv,noheader"]).strip():continue
                    with (OUT/f"gpu{gpu}.log").open("ab") as log:
                        live[gpu]=subprocess.Popen([sys.executable,"-B",__file__,"lane",str(gpu)],stdout=log,stderr=log,start_new_session=True)
                count=sum(1 for p in OUT.glob("n*/H*_A*.json"))
                for gpu,p in live.items():
                    if p.poll() not in (None,0):raise RuntimeError((OUT/f"gpu{gpu}.log").read_text(errors="replace")[-2500:])
                if shutil.disk_usage(OUT).free<1.30*2**30:raise RuntimeError("free storage below 1.30 GiB")
                if len(live)==4 and all(p.poll()==0 for p in live.values()):break
                save(HERE/"supp_status.json",dict(phase="source_training",done=count,total=660,assigned_gpus=list(live),updated=time.time()))
                time.sleep(12)
            assert count==660
            save(HERE/"supp_status.json",dict(phase="source_complete",done=count,total=660))
    except BaseException:
        save(HERE/"supp_status.json",dict(phase="failed",error=traceback.format_exc()))
        for p in live.values():
            if p.poll() is None:
                import signal
                os.killpg(p.pid,signal.SIGTERM)
        raise
if __name__=="__main__":
    op=sys.argv[1]
    if op=="lane":lane(int(sys.argv[2]))
    elif op=="run":run()
    elif op=="dispatch":
        if (HERE/"supp_status.json").exists():raise RuntimeError("Existing source job")
        save(HERE/"supp_status.json",{"phase":"starting"})
        with (OUT/"supervisor.log").open("ab") as log:
            subprocess.Popen([sys.executable,"-B",__file__,"run"],start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
