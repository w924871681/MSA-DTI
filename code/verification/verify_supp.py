"""Recompute supplementary decisions from validation panels and errors from predictions."""
import csv,hashlib,io,json,statistics,sys,tarfile
from collections import Counter
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def digest(b):return hashlib.sha256(b).hexdigest()
def savecsv(path,rows):
    if not rows:raise ValueError("empty report: "+str(path))
    with path.open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def verify(data,source):
    table=data/"supplement";table.mkdir(exist_ok=True)
    plan=json.loads((HERE/"pipeline_plan.json").read_bytes())
    cases_path="timing_s50_flow_20261104/C_real/fresh_E_stage/inputs/"
    variants=["n20_s2904"]+[f"n{n}_s{s}" for n,s in plan["variants"]]
    def prof(i,h):return plan["profiles"][str(h)][str(i)]
    def valid(i,h,tier,mask):
        a,b=prof(i,h);p,o=plan["caps"][tier]
        return (mask=="operations" or a<=p) and (mask=="parameters" or b<=o)
    def choose(r,loss,ref,tau,h):
        ids=[i for i in r["subset"] if valid(i,h,r["tier"],r["mask"])]
        selected=ref if tau is None else min((i for i in ids if i==ref or loss[str(i)]<=(1-tau)*loss[str(ref)]),key=lambda i:(loss[str(i)],*prof(i,h),i))
        return selected,len(ids)
    details=[];conditions=[];latencies=[];timings=[]
    source_set=set(json.loads((data/"roles.json").read_bytes())["source"])
    with tarfile.open(HERE/"source_weights.tar") as weights:
        full_ids=json.load(weights.extractfile("sources/n20_s2905/H1_A0.json"))["source_ids"]
        assert len(full_ids)==20 and set(full_ids)==source_set
        for v in variants[1:]:
            for h in (1,4):
                for i in range(66):
                    p=f"sources/{v}/H{h}_A{i}"
                    receipt=json.load(weights.extractfile(p+".json"))
                    assert digest(weights.extractfile(p+".pt").read())==receipt["sha256"]
                    assert receipt["pool"]*100==receipt["updates"] and receipt["epochs"]==50
                    n,seed=map(int,v[1:].split("_s"))
                    assert (receipt["pool"],receipt["base_seed"],receipt["arch"],receipt["H"])==(n,seed,i,h)
                    assert len(set(receipt["source_ids"]))==n and receipt["model_seed"]==seed+101*h+i
                    assert full_ids[:n]==receipt["source_ids"]
    with tarfile.open(HERE/"pipeline_results.tar") as result,tarfile.open(source) as archive:
        def blob(n):return result.extractfile(n).read()
        def doc(n):return json.loads(blob(n))
        tasks=json.load(archive.extractfile(cases_path+"cases.json"))
        truth={}
        for c in tasks:
            b=archive.extractfile(cases_path+"future/"+c["slug"]+".bin").read();assert digest(b)==c["future_hash"]
            a=np.frombuffer(b,dtype=[("ts","<i8"),("raw","<f8",(10,))]);h=c["H"];t=c["T"];assert len(a)==t+c["B"]+h
            cpu=a["raw"][np.arange(t,t+c["B"])[:,None]+np.arange(1,h+1),3]
            truth[c["slug"]]=(cpu[np.all(np.isfinite(cpu)&(cpu>=0),axis=1)]/100).astype(np.float32)
        assert len(truth)==120
        for v in variants:
            root=f"panels/{v}/";seal=doc(root+"E_seal.json");cal=doc(root+"calibrated.json")
            assert seal["calibration"]==digest(blob(root+"calibrated.json"))
            assert set(seal["prefixes"])==set(seal["actions"])==set(truth)
            ref=cal["ref"]; rules=cal["rules"]
            bank=plan["bank"] if v==variants[0] else doc(root+"bank.json")["bank"]
            pre=55 if v==variants[0] else doc(root+"A_ref.json")["ref"]
            if v!=variants[0]:
                def cohort(stage):
                    return [doc(k) for k in result.getnames() if k.startswith(root+stage+"/") and k.endswith(".json") and not k.endswith(".prefix.json")]
                for stage,loss,ids,expected in [("A","validation",range(66),pre),("C","assessment",bank,ref)]:
                    records=cohort(stage);assert len(records)==80 and len({a["VM"] for a in records})==20
                    eligible=[i for i in ids if all(valid(i,h,"tight","both") for h in (1,4))]
                    def criterion(i):
                        means={}
                        for a in records:means.setdefault(a["VM"],[]).append(a[loss]["sgd50"][str(i)])
                        assert all(len(m)==4 for m in means.values())
                        return (statistics.mean(statistics.mean(m) for m in means.values()),*map(max,zip(prof(i,1),prof(i,4))),i)
                    assert min(eligible,key=criterion)==expected
                counter=[[0,0] for _ in range(66)];records=cohort("B");assert len(records)==80
                for a in records:
                    va=a["validation"]["sgd50"];ck=a["assessment"]["sgd50"];h=a["H"]
                    challengers=[int(i) for i in va if int(i)!=pre]
                    for j,scores in enumerate((ck,va)):
                        winner=min(challengers,key=lambda i:(scores[str(i)],*prof(i,h),i))
                        if ck[str(winner)]<ck[str(pre)] and (j==0 or va[str(winner)]<va[str(pre)]):counter[winner][j]+=1
                assert counter==doc(root+"bank.json")["counts"] and bank==[i for i,row in enumerate(counter) if i==pre or max(row)>=2]
            conditions.append(dict(variant=v,screen_ref=pre,candidates=len(bank),reference=ref,medium_candidates=sum(valid(i,1,"medium","both") for i in bank),tau=cal["policies"]["full"]["tau"]))
            # Recheck threshold selection on calibration tasks separately from E.
            drows=[doc(k) for k in result.getnames() if k.startswith(root+"D/") and k.endswith(".json") and not k.endswith(".prefix.json")]
            assert len(drows)==80 and len({x["VM"] for x in drows})==20
            for name,r in rules.items():
                if not r["calibrate"]:continue
                t=r["optimizer"]+str(r["steps"]);entries=[]
                for tau in plan["grid"]:
                    pairs=[(x,choose(r,x["validation"][t],ref,tau,x["H"])[0]) for x in drows]
                    m=statistics.mean(x["assessment"][t][str(i)] for x,i in pairs)
                    harm=statistics.mean(i!=ref and x["assessment"][t][str(i)]>x["assessment"][t][str(ref)] for x,i in pairs)
                    if harm<=.05:entries.append((m,-tau))
                baseline=statistics.mean(x["assessment"][t][str(ref)] for x in drows)
                expected=-min(entries)[1] if entries and min(entries)[0]<baseline else None
                assert expected==cal["policies"][name]["tau"],(v,name,expected)
            for c in tasks:
                slug=c["slug"];h=c["H"];y=truth[slug];path=root+"E/"+slug+".json"
                assert seal["prefixes"][slug]==digest(blob(path))
                predata=doc(path);assert predata["prefix"]==c["prefix_hash"]
                receipt=doc(root+"scores/"+slug+".json");buf=blob(root+"scores/"+slug+".npz")
                assert digest(buf)==receipt["panel"] and receipt["future"]==c["future_hash"] and receipt["seal"]==digest(blob(root+"E_seal.json"))
                with np.load(io.BytesIO(buf)) as panel:
                    stats={}
                    for k in panel.files:
                        p=panel[k];assert p.shape==y.shape and np.isfinite(p).all()
                        e=p.astype(np.float64)-y;loss=(e**2).mean(1)
                        m=[float(loss.mean()),float(np.abs(e).mean()),float(np.sort(loss)[-max(1,round(.1*len(loss))):].mean())]
                        assert m==receipt["stats"][k];stats[k]=m
                for name,r in rules.items():
                    a=seal["actions"][slug][name];tau=cal["policies"][name]["tau"];tag=r["optimizer"]+str(r["steps"])
                    i,count=choose(r,predata["validation"][tag],ref,tau,h);assert (i,count)==(a["arch"],a["eligible"])
                    m=stats[f"{tag}_A{i}"];refm=stats[f"{tag}_A{ref}"][0]
                    details.append(dict(variant=v,rule=name,VM=c["VM"],H=h,K=c["K"],arch=i,reference=ref,steps=r["steps"],optimizer=r["optimizer"],eligible=count,tau=tau,budget=r["tier"],parameters=prof(i,h)[0],operations=prof(i,h)[1],feasible=int(valid(i,h,r["tier"],"both")),MSE=m[0],MAE=m[1],Worst10=m[2],harm=int(i!=ref and m[0]>refm),switch=int(i!=ref)))
        for p in result.getnames():
            if not p.startswith("panels/hardware/") or not p.endswith(".json"):continue
            d=doc(p)
            if "batches" in d:
                values=np.asarray(d["batches"]);assert values.shape==(5,1000) and (values>0).all()
                latencies.append(dict(arch=d["arch"],H=d["H"],device=d["device"],median_ms=np.median(values)/1e6,p95_ms=np.quantile(values,.95)/1e6,calls=values.size))
            elif "times_seconds" in d:
                values=d.pop("times_seconds");assert len(values)==5 and min(values)>0
                timings.append(dict(**d,mean_seconds=np.mean(values),sd_seconds=np.std(values,ddof=1),min_seconds=min(values),max_seconds=max(values)))
            elif p.endswith("/system.json"):system=d
    assert len(latencies)==19*2*2 and len(timings)==24
    original=list(csv.DictReader((data/"bitbrains_policy.csv").open(encoding="utf-8-sig")))
    original={(r["method"],r["budget"],r["VM"],int(r["H"]),int(r["K"])):r for r in original}
    anchors={"full":"msa","tight_both":"msa","medium_both":"msa","loose_both":"msa",
             "subset1":"reference_only","sgd_direct":"direct_minimum"}
    matched=[r for r in details if r["variant"]==variants[0] and r["rule"] in anchors]
    assert len(matched)==120*len(anchors)
    for r in matched:
        s=original[anchors[r["rule"]],r["budget"],r["VM"],r["H"],r["K"]]
        assert int(s["arch"])==r["arch"] and all(abs(float(s[m])-r[m])<1e-14 for m in ("MSE","MAE","Worst10"))
    assert system["threads"]==1
    (table/"hardware.json").write_text(json.dumps(system,indent=2),encoding="utf8")
    savecsv(table/"decisions.csv",details);savecsv(table/"sources.csv",conditions)
    savecsv(table/"latency.csv",latencies);savecsv(table/"target_time.csv",timings)
    print("Verified source checkpoints, calibration choices, E selections and per-window scores",len(details),len(latencies),len(timings))
if __name__=="__main__":verify(Path(sys.argv[1]),sys.argv[2])
