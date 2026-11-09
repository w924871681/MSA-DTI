"""Recompute each task from the score-time prediction and its original future labels."""
from pathlib import Path
import csv, hashlib, io, json, sys, tarfile
import numpy as np
HERE=Path(__file__).resolve().parent
D=Path(sys.argv[2])
def digest(b):return hashlib.sha256(b).hexdigest()
def raw(a,n):return a.extractfile(n).read()
def obj(a,n):return json.loads(raw(a,n))
def labels(b,c):
    assert digest(b)==c["future_hash"]
    dt=np.dtype([("timestamp","<i8"),("raw","<f8",(10,))])
    series=np.frombuffer(b,dtype=dt)
    t,h=c["T"],c["H"];assert len(series)==t+c["B"]+h
    y=series["raw"][np.arange(t,t+c["B"])[:,None]+np.arange(1,h+1),3]
    return (y[np.all(np.isfinite(y)&(y>=0),axis=1)]/100).astype(np.float32)
def metrics(pred,y):
    assert pred.shape==y.shape and np.isfinite(pred).all()
    err=pred.astype(np.float64)-y.astype(np.float64);loss=np.square(err).mean(1)
    return {"MSE":loss.mean(),"MAE":np.abs(err).mean(),"Worst10":np.sort(loss)[-max(1,round(.1*len(loss))):].mean()}
def write(p,rows):
    with p.open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def ablation():
    src=tarfile.open(sys.argv[3]);new=tarfile.open(HERE/"ablation_results.tar")
    plan=json.loads((HERE/"ablation_plan.json").read_bytes())
    seal=obj(new,"seal.json");assert seal["inputs_sha256"]==digest((HERE/"abl_inputs.zip").read_bytes())
    e="timing_s50_flow_20261104/C_real/fresh_E_stage/"
    rows=[]
    for c in obj(src,e+"inputs/cases.json"):
        s=c["slug"];v=plan["plans"][s];picks=v["picks"];assert picks==seal["decisions"][s]
        y=labels(raw(src,e+"inputs/future/"+s+".bin"),c)
        with np.load(io.BytesIO(raw(new,s+".npz"))) as pred:
            stat={int(i):metrics(pred[i],y) for i in pred.files}
            for m,i in picks.items():
                if i!=55:
                    orig=next(r for r in obj(src,e+"results/"+s+"/scores.json")["rows"] if r["budget"]=="medium" and r["arch"]==i)
                    assert all(orig[k]==x for k,x in stat[i].items())
                rows.append(dict(VM=c["VM"],H=c["H"],K=c["K"],budget="medium",method=m,arch=i,reference=55,tau=plan["configs"][m],harm=int(i!=55 and stat[i]["MSE"]>stat[55]["MSE"]),**stat[i]))
    assert len(rows)==360
    write(D/"reference_controls.csv",rows)
def azure():
    a=tarfile.open(HERE/"azure_results.tar");base=Path(sys.argv[3]);cases=json.loads((base/"inputs/cases.json").read_bytes())
    seal=obj(a,"SEAL.json");pilot=obj(a,"PILOT_PASS.json"); methods={"meta_nas_lite","zero_nas","zero_nas_ft"};assert set(seal)==set(pilot["methods"])==methods and len(cases)==120
    caps=json.loads((base/"inputs/bitbrains_protocol.json").read_bytes())["budgets"]["medium"]
    rows=[]
    for m in sorted(methods):
        for c in cases:
            s=c["slug"];p=f"batch/{m}/medium/{s}/";selection=obj(a,p+"selected.json");score=obj(a,p+"score.json")
            assert digest(raw(a,p+"selected.json"))==seal[m]["medium/"+s] and digest(raw(a,p+"forecast.npz"))==score["forecast_sha256"]
            assert selection["prefix_sha256"]==c["prefix_hash"] and score["future_sha256"]==c["future_hash"]
            assert len(selection["eligible"])==48 and set(selection["shortlist"])<=set(selection["eligible"]) and score["arch"]==selection["arch"] in selection["shortlist"]
            assert all(selection[key]<=limit for key,limit in zip(["parameters","operations"],caps))
            y=labels((base/"future_local"/(s+".bin")).read_bytes(),c)
            with np.load(io.BytesIO(raw(a,p+"forecast.npz"))) as z:stats=metrics(z["prediction"],y)
            assert all(stats[k]==score[k] for k in stats)
            rows.append(dict(VM=c["VM"],H=c["H"],K=c["K"],budget="medium",method=m,arch=score["arch"],parameters=selection["parameters"],operations=selection["operations"],feasible=1,**stats))
    write(D/"azure_external/medium_search_verified.csv",rows)
if __name__=="__main__":
    if sys.argv[1]=="ablation":ablation()
    else:azure()
