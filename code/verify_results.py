"""Score every published budget comparison from window-level forecasts and raw labels."""
import argparse, io, zipfile, json
from pathlib import Path
import numpy as np
import pandas as pd
from inputs import CODE, WORK, METHODS, cases, carrier, metrics, read, truth

KEY=["method","budget","VM","H","K"]
METRICS=["MSE","MAE","Worst10"]
def reference(dataset):
    if dataset=="bitbrains":
        scores=pd.read_csv(CODE/"data/task_results.csv")
        # Fixed PT-FT used A39 under all caps. The task-results file also contains a separate tight retraining control.
        mask=scores.method.eq("PT-FT")
        fixed=scores[mask & scores.budget.eq("medium")].copy()
        scores=pd.concat([scores[~(mask & scores.budget.eq("tight"))],fixed.assign(budget="tight")])
        return scores.set_index(KEY)
    base=CODE/"data/azure_external"
    df=pd.read_csv(base/"baseline_archive.csv")
    df=df[df.method.isin(set(METHODS)-{"msa"})]
    search=df.method.isin(["meta_nas_lite","zero_nas","zero_nas_ft"])
    frames=[df[~(search & df.budget.ne("loose"))],pd.read_csv(base/"frozen19.csv").query('method=="msa"')]
    frames.extend(pd.read_csv(base/(t+"_search_verified.csv")) for t in ["tight","medium"])
    df=pd.concat(frames,ignore_index=True)
    df["method"]=df.method.map(METHODS)
    return df.set_index(KEY)

def compare(dataset, frame, exact=True):
    actual=frame.set_index(KEY).sort_index()
    expected=reference(dataset).loc[actual.index]
    if not actual.index.is_unique or len(actual)!=len(expected):
        raise ValueError("Task keys differ")
    arch=int((actual.arch!=expected.arch).sum())
    delta=np.abs(actual[METRICS].to_numpy()-expected[METRICS].to_numpy())
    peak=delta.max(axis=0)
    print(dataset, len(actual), "rows; architecture mismatches:",arch,"max absolute differences:",dict(zip(METRICS,peak)))
    if exact and (arch or not np.allclose(actual[METRICS],expected[METRICS],rtol=1e-11,atol=1e-13)):
        raise ValueError("Published comparison differs")
    return expected

def run(dataset, raw_path):
    rows=[]; stage="azure" if dataset=="azure" else "E"
    with zipfile.ZipFile(WORK/"assets"/(dataset+"-forecasts.zip")) as panel:
        records=json.loads(panel.read("records.json")); by_slug={}
        for row in records:
            by_slug.setdefault(row["slug"],[]).append(row)
        for c in cases(stage):
            y,_=truth(c,carrier(stage,c,raw_path))
            batch=by_slug.pop(c["slug"])
            if len(batch)!=24:raise ValueError("Expected eight methods at three budgets")
            with np.load(io.BytesIO(panel.read(c["slug"]+".npz")),allow_pickle=False) as pred:
                for a in batch:
                    if a["future_sha256"]!=c["future_hash"]:raise ValueError("Future identity changed")
                    rows.append(dict(method=METHODS[a["method"]],budget=a["budget"],VM=c["VM"],H=c["H"],K=c["K"],
                                     arch=a["arch"],**metrics(pred[a["key"]],y)))
        if by_slug or len(rows)!=2880:raise ValueError("Unexpected coverage")
    frame=pd.DataFrame(rows)
    compare(dataset,frame)
    dest=WORK/"verified"; dest.mkdir(exist_ok=True,parents=True)
    frame.to_csv(dest/(dataset+".csv"),index=False)
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bitbrains",type=Path,required=True,help="original fastStorage.zip")
    args=parser.parse_args()
    for name in ("bitbrains","azure"):run(name,args.bitbrains)

