"""Continue frozen source conditions through distinct A/B/C/D/E cohorts."""
import concurrent.futures as cf
import csv
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import traceback
import panels as q

HERE=q.HERE; OUT=q.OUT; STATUS=HERE/"pipeline_status.json"
def prof(i,h):return tuple(q.plan()["profiles"][str(h)][str(i)])
def ids_at(ids,h,tier,mask="both"):
    a,b=q.plan()["caps"][tier]
    return [i for i in ids if (mask=="operations" or prof(i,h)[0]<=a) and (mask=="parameters" or prof(i,h)[1]<=b)]
def joint(ids):return [i for i in ids if all(i in ids_at([i],h,"tight") for h in (1,4))]
def docs(v,stage):
    return [q.read(q.outfile(v,stage,c["slug"])) for c in q.cases(stage)]
def reference(v,stage,ids):
    rows=docs(v,stage); key="validation" if stage=="A" else "assessment"
    result={}
    for i in ids:
        grouped={}
        for r in rows:grouped.setdefault(r["VM"],[]).append(r[key]["sgd50"][str(i)])
        assert len(grouped)==20 and all(len(g)==4 for g in grouped.values())
        result[i]=statistics.mean(statistics.mean(g) for g in grouped.values())
    ranking=sorted(ids,key=lambda i:(result[i],max(prof(i,h)[0] for h in (1,4)),max(prof(i,h)[1] for h in (1,4)),i))
    q.save(OUT/v/(stage+"_ref.json"),dict(ref=ranking[0],ranking=ranking,means=result))
    return ranking[0]
def screen(v,ref):
    counts=[[0,0] for _ in range(66)]
    for r in docs(v,"B"):
        va,ck=r["validation"]["sgd50"],r["assessment"]["sgd50"]
        others=[int(i) for i in va if int(i)!=ref];assert str(ref) in va
        for k,loss in enumerate((ck,va)):
            i=min(others,key=lambda i:(loss[str(i)],*prof(i,r["H"]),i))
            if ck[str(i)]<ck[str(ref)] and (k==0 or va[str(i)]<va[str(ref)]):counts[i][k]+=1
    ids=[i for i,row in enumerate(counts) if max(row)>=2 or i==ref]
    q.save(OUT/v/"bank.json",dict(bank=ids,screening_reference=ref,counts=counts))
    return ids
def pick(loss,ids,ref,tau,h):
    if tau is None:return ref
    choices=[i for i in ids if i==ref or loss[str(i)]<=(1-tau)*loss[str(ref)]]
    return min(choices,key=lambda i:(loss[str(i)],*prof(i,h),i))
def rules(v,bank):
    base=dict(steps=50,optimizer="sgd",subset=bank,tier="medium",mask="both",calibrate=True)
    result={"full":base}
    if v==q.CURRENT:
        for n in q.plan()["steps"]:
            if n!=50:result[f"steps{n}"]={**base,"steps":n}
        # Restrict in an order established exclusively by B evidence.
        medium=[i for i in q.plan()["bank_order"] if all(i in ids_at(bank,h,"medium") for h in (1,4))]
        for n in (1,4,8,12):result[f"subset{n}"]={**base,"subset":medium[:n]}
        result["adam"]={**base,"optimizer":"adam"}
        result["adam_direct"]={**result["adam"],"calibrate":False,"fixed":0}
        result["sgd_direct"]={**base,"calibrate":False,"fixed":0}
        for tier in q.plan()["caps"]:
            for mask in ("parameters","operations","both"):
                result[f"{tier}_{mask}"]={**base,"tier":tier,"mask":mask,"calibrate":False,"fixed":"primary"}
    return result
def calibrate(v,ref,book):
    rows=docs(v,"D"); policies={}
    for name,r in book.items():
        if not r["calibrate"]:
            policies[name]=dict(tau=policies["full"]["tau"] if r["fixed"]=="primary" else r["fixed"]);continue
        tag=r["optimizer"]+str(r["steps"]);curve=[]
        base=statistics.mean(x["assessment"][tag][str(ref)] for x in rows)
        for tau in q.plan()["grid"]:
            errors=[];harms=[]
            for x in rows:
                ids=ids_at(r["subset"],x["H"],r["tier"],r["mask"])
                i=pick(x["validation"][tag],ids,ref,tau,x["H"])
                checks=x["assessment"][tag]
                errors.append(checks[str(i)]);harms.append(i!=ref and checks[str(i)]>checks[str(ref)])
            curve.append(dict(tau=tau,mse=statistics.mean(errors),harm=statistics.mean(harms)))
        valid=[x for x in curve if x["harm"]<=.05];best=min(valid,key=lambda x:(x["mse"],-x["tau"])) if valid else None
        policies[name]=dict(tau=best["tau"] if best and best["mse"]<base else None,curve=curve,reference_mse=base)
    if v==q.CURRENT:assert policies["full"]["tau"]==.2,policies["full"]
    q.save(OUT/v/"calibrated.json",dict(ref=ref,rules=book,policies=policies))
    return policies
def seal(v,ref,book,policies):
    actions={};prefixes={}
    for x in docs(v,"E"):
        slug=x["slug"];h=x["H"];prefixes[slug]=q.sha(q.outfile(v,"E",slug));acts={}
        for name,r in book.items():
            ids=ids_at(r["subset"],h,r["tier"],r["mask"]);assert ref in ids
            a=pick(x["validation"][r["optimizer"]+str(r["steps"])],ids,ref,policies[name]["tau"],h)
            acts[name]=dict(arch=a,ref=ref,steps=r["steps"],optimizer=r["optimizer"],eligible=len(ids))
        actions[slug]=acts
    assert len(actions)==120
    q.save(OUT/v/"E_seal.json",dict(prefixes=prefixes,actions=actions,calibration=q.sha(OUT/v/"calibrated.json")))
def lane(v,stage,gpu):
    cases=q.cases("E" if stage=="scores" else stage)
    assigned=cases[gpu::4]
    fn=q.score_task if stage=="scores" else q.task
    ctx=mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=7,mp_context=ctx,initializer=q.setup,initargs=(gpu,stage)) as pool:
        pending=[pool.submit(fn,v,c) if stage=="scores" else pool.submit(fn,v,stage,c) for c in assigned]
        for f in cf.as_completed(pending):f.result()
def phase(v,stage):
    children={}
    try:
        for gpu in range(4):
            log=OUT/v/f"{stage}_{gpu}.log";log.parent.mkdir(parents=True,exist_ok=True)
            with log.open("ab") as f:
                children[gpu]=subprocess.Popen([sys.executable,"-B",__file__,"lane",v,stage,str(gpu)],stdout=f,stderr=f,start_new_session=True)
        while True:
            for gpu,p in children.items():
                if p.poll() not in (None,0):raise RuntimeError((OUT/v/f"{stage}_{gpu}.log").read_text(errors="replace")[-3200:])
            done=len([p for p in (OUT/v/stage).glob("*.json") if not p.name.endswith(".prefix.json")])
            total=120 if stage in ("E","scores") else 80
            q.save(STATUS,dict(phase="panels",condition=v,stage=stage,done=done,total=total,updated=time.time()))
            if all(p.poll()==0 for p in children.values()):assert done==total;return
            if shutil.disk_usage(HERE).free<1.25*2**30:raise RuntimeError("free storage below 1.25 GiB")
            time.sleep(10)
    finally:
        import signal
        for p in children.values():
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
def archive():
    result={}
    for name,source in [("pipeline_results.tar",OUT),("source_weights.tar",HERE/"sources")]:
        tmp=HERE/(name+".partial")
        with tarfile.open(tmp,"w") as t:
            for p in sorted(source.rglob("*")):
                if p.is_file() and p.suffix in (".json",".npz",".csv",".pt"):t.add(p,arcname=p.relative_to(HERE))
            for n in ("pipeline.py","panels.py","pipeline_plan.json","supp_retry.py","supp_inputs.zip"):
                t.add(HERE/n,arcname=n)
        os.replace(tmp,HERE/name);result[name]=q.sha(HERE/name)
    return result
def supervise():
    with (HERE/"pipeline.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while q.read(HERE/"supp_status.json")["phase"]!="source_complete":
                st=q.read(HERE/"supp_status.json")
                if st["phase"]=="failed":raise RuntimeError(st["error"])
                q.save(STATUS,dict(phase="waiting_for_sources",done=st.get("done",0),total=660,updated=time.time()));time.sleep(20)
            while subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"]).strip():
                q.save(STATUS,dict(phase="waiting_for_idle_devices"));time.sleep(20)
            for v in [q.CURRENT]+[f"n{n}_s{s}" for n,s in q.plan()["variants"]]:
                if v==q.CURRENT:
                    ids=q.plan()["bank"];ref=57
                else:
                    phase(v,"A");pre=reference(v,"A",joint(range(66)))
                    phase(v,"B");ids=screen(v,pre)
                    phase(v,"C");ref=reference(v,"C",joint(ids))
                book=rules(v,ids);phase(v,"D");policies=calibrate(v,ref,book)
                phase(v,"E");seal(v,ref,book,policies);phase(v,"scores")
            q.save(STATUS,dict(phase="hardware"))
            subprocess.run([sys.executable,"-B",__file__,"hardware"],check=True)
            q.save(STATUS,dict(phase="complete",archives=archive()))
        except BaseException:
            q.save(STATUS,dict(phase="failed",error=traceback.format_exc()));raise
if __name__=="__main__":
    op=sys.argv[1]
    if op=="lane":lane(sys.argv[2],sys.argv[3],int(sys.argv[4]))
    elif op=="run":supervise()
    elif op=="hardware":q.hardware()
    elif op=="dispatch":
        if STATUS.exists():raise RuntimeError("Existing pipeline")
        q.save(STATUS,dict(phase="starting"))
        with (HERE/"pipeline.log").open("ab") as log:
            subprocess.Popen([sys.executable,"-B",__file__,"run"],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
