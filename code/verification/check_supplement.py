"""Recompute the supplementary and ablation tables using the archived experiment outputs."""
import io,json,shutil,sys,tarfile,tempfile,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inputs import CODE,WORK,cases,carrier,read,truth
from reproduce import py
import pandas as pd
import numpy as np

def same_csv(a,b):
    x,y=[pd.read_csv(p,float_precision="round_trip") for p in (a,b)]
    if list(x)!=list(y) or x.shape!=y.shape:raise ValueError(str(a))
    for col in x:
        if pd.api.types.is_numeric_dtype(x[col]):
            np.testing.assert_allclose(x[col],y[col],atol=1e-14,rtol=1e-12,equal_nan=True)
        else:
            pd.testing.assert_series_equal(x[col],y[col])
def member(archive,name,data):
    info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
def main(raw_path):
    with tempfile.TemporaryDirectory(dir=WORK) as tmp:
        stage=Path(tmp); data=stage/"data";shutil.copytree(CODE/"data",data)
        with zipfile.ZipFile(WORK/"assets/supplement-evidence.zip") as z:z.extractall(stage)
        for n in ("pipeline_plan.json","verify_supp.py","verify_tight.py","verify_modules.py"):shutil.copyfile(CODE/"verification"/n,stage/n)
        for dest in (stage/"verify.py",stage/"medium/verify.py"):shutil.copyfile(CODE/"verification/verify_medium.py",dest)
        prefix="timing_s50_flow_20261104/C_real/fresh_E_stage/"
        t=stage/"bitbrains.tar"
        with tarfile.open(t,"w") as a:
            for p in (stage/"controls").rglob("*"):
                if p.is_file():a.add(p,arcname=prefix+p.relative_to(stage/"controls").as_posix())
            for c in cases("E"):
                blob=carrier("E",c,raw_path);truth(c,blob)
                member(a,prefix+"inputs/future/"+c["slug"]+".bin",blob)
        azure=stage/"azure";(azure/"inputs").mkdir(parents=True);(azure/"future_local").mkdir()
        shutil.copyfile(CODE/"protocol/azure/cases.json",azure/"inputs/cases.json")
        (azure/"inputs/bitbrains_protocol.json").write_text(json.dumps({"budgets":read(CODE/"verification/pipeline_plan.json")["caps"]}))
        for c in cases("azure"):(azure/"future_local"/(c["slug"]+".bin")).write_bytes(carrier("azure",c,None))
        py(stage/"verify_supp.py",data,t);py(stage/"verify_tight.py",data,azure)
        py(stage/"medium/verify.py","azure",data,azure);py(stage/"verify_modules.py","modules",data,t,stage/"fair.tar")
        files=["supplement/"+n+".csv" for n in ("decisions","sources","latency","target_time")]
        files+=["azure_external/tight_search_verified.csv","azure_external/medium_search_verified.csv","no_preselection.csv"]
        for n in files:same_csv(data/n,CODE/"data"/n)
        if read(data/"supplement/hardware.json")!=read(CODE/"data/supplement/hardware.json"):raise ValueError("Hardware metadata differs")
        dest=WORK/"verified";dest.mkdir(exist_ok=True)
        for n in files:
            p=dest/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(data/n,p)
        print("Recomputed supplementary panels, 720 Azure search selections and 120 preselection ablations.")
if __name__=="__main__":main(Path(sys.argv[1]))

