"""Independently reconstruct uploaded tight-budget task metrics from forecasts."""
import csv,hashlib,io,json,sys,tarfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def hashbytes(b):return hashlib.sha256(b).hexdigest()
def verify(data,inputs):
    plan=json.loads((HERE/"pipeline_plan.json").read_bytes())
    cases=json.loads((inputs/"inputs/cases.json").read_bytes());assert len(cases)==120
    rows=[]
    with tarfile.open(HERE/"azure_results.tar") as a:
        def blob(n):return a.extractfile(n).read()
        def doc(n):return json.loads(blob(n))
        seals=doc("SEAL.json");pilot=doc("PILOT_PASS.json")
        assert set(seals)==set(pilot["methods"])=={"meta_nas_lite","zero_nas","zero_nas_ft"}
        assert set(pilot["tiers"])=={"tight","medium","loose"}
        for method in sorted(seals):
            assert len(seals[method])==120
            for c in cases:
                key="tight/"+c["slug"];root="batch/"+method+"/"+key+"/"
                selection=doc(root+"selected.json");score=doc(root+"score.json")
                assert hashbytes(blob(root+"selected.json"))==seals[method][key]
                assert selection["prefix_sha256"]==c["prefix_hash"] and score["future_sha256"]==c["future_hash"]
                i=score["arch"];h=c["H"];p=plan["profiles"][str(h)];cap=plan["caps"]["tight"]
                feasible=[k for k in range(66) if all(x<=y for x,y in zip(p[str(k)],cap))]
                assert selection["eligible"]==feasible and len(feasible)==26
                assert set(selection["shortlist"])<=set(feasible) and i==selection["arch"] and i in selection["shortlist"]
                assert [selection["parameters"],selection["operations"]]==p[str(i)]
                b=(inputs/"future_local"/(c["slug"]+".bin")).read_bytes();assert hashbytes(b)==c["future_hash"]
                records=np.frombuffer(b,dtype=[("ts","<i8"),("raw","<f8",(10,))])
                t=c["T"];assert len(records)==t+c["B"]+h
                assert hashbytes(records[:t+1].tobytes())==c["prefix_hash"]
                cpu=records["raw"][np.arange(t,t+c["B"])[:,None]+np.arange(1,h+1),3]
                y=(cpu[np.all(np.isfinite(cpu)&(cpu>=0),axis=1)]/100).astype(np.float32)
                predraw=blob(root+"forecast.npz");assert hashbytes(predraw)==score["forecast_sha256"]
                with np.load(io.BytesIO(predraw)) as f:pred=f["prediction"]
                assert pred.shape==y.shape and np.isfinite(pred).all()
                err=pred.astype(np.float64)-y.astype(np.float64);loss=np.square(err).mean(1)
                assert [score[k] for k in ("MSE","MAE","Worst10")]==[loss.mean(),abs(err).mean(),np.sort(loss)[-max(1,round(.1*len(loss))):].mean()]
                rows.append(dict(VM=c["VM"],H=h,K=c["K"],budget="tight",method=method,arch=i,parameters=selection["parameters"],operations=selection["operations"],feasible=1,**{k:score[k] for k in ("MSE","MAE","Worst10")}))
    target=data/"azure_external/tight_search_verified.csv"
    with target.open("w",encoding="utf8",newline="") as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)
    assert len({(r["method"],r["VM"],r["H"],r["K"]) for r in rows})==360
    print("verified 360 sealed selections, model budget profiles and per-window scores")
if __name__=="__main__":verify(Path(sys.argv[1]),Path(sys.argv[2]))
