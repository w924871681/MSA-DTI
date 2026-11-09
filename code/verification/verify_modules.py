"""Cross-check the 48-way decisions, seals, forecasts, and original 17-way controls."""
from pathlib import Path
import hashlib, io, json, sys, tarfile, zipfile
import numpy as np
from verify import labels, metrics, write
HERE=Path(__file__).resolve().parent; tar=tarfile.open(HERE/"modules_results.tar")
old=tarfile.open(sys.argv[3]);fair=tarfile.open(sys.argv[4]);D=Path(sys.argv[2])
e="timing_s50_flow_20261104/C_real/fresh_E_stage/"
def raw(a,n):return a.extractfile(n).read()
def get(a,n):return json.loads(raw(a,n))
def sha(b):return hashlib.sha256(b).hexdigest()
p=zipfile.ZipFile(HERE/"module_inputs.zip");proto=json.loads(p.read("protocol.json"));seal=get(tar,"seal.json");assert seal["protocol_sha"]==sha((HERE/"module_inputs.zip").read_bytes())
bank=get(old,e+"inputs/bank.json");rows=[];updated=0
for c in get(old,e+"inputs/cases.json"):
    s=c["slug"]; h=c["H"];folder=e+"results/"+s+"/"
    n=s+"/selected.json";assert sha(raw(tar,n))==seal["decisions"][s];sel=get(tar,n)
    mark=get(tar,s+"/score.json");future=raw(old,e+"inputs/future/"+s+".bin");y=labels(future,c)
    assert mark["future_hash"]==c["future_hash"] and sha(raw(tar,s+"/forecast.npz"))==mark["panel_sha"]
    base=get(old,folder+"medium/actions.json");search=get(fair,"batch/zero_nas/medium/"+s+"/selected.json")
    assert sel["eligible"]==search["eligible"] and len(sel["eligible"])==48 and sel["bank_eligible"]==sorted(base["adapted"]) and len(sel["bank_eligible"])==17
    v={int(i):x for i,x in sel["validation"]["no_preselection"].items()};assert sorted(v)==sel["eligible"] and all(np.isfinite(x) for x in v.values())
    assert all(v[row["arch"]]==row["validation"] for row in base["rows"])
    winner=sel["picks"]["no_preselection"];valid={i for i in v if i==57 or v[i]<=(1-proto["tau"])*v[57]};assert winner in valid and v[winner]==min(v[i] for i in valid)
    # No resource tie needs resolving on this run.
    assert len([i for i in valid if v[i]==v[winner]])==1
    with np.load(io.BytesIO(raw(tar,s+"/forecast.npz"))) as z,np.load(io.BytesIO(raw(old,folder+"forecasts.npz"))) as archive:
        pred=z["no_preselection"]
        if "medium_"+str(winner) in archive.files:assert np.array_equal(pred,archive["medium_"+str(winner)]);updated+=1
        vals=metrics(pred,y);ref=metrics(archive["medium_57"],y)["MSE"]
    rows.append(dict(VM=c["VM"],H=h,K=c["K"],method="no_preselection",budget="medium",arch=winner,feasible=1,adapted=48,**vals,switch=int(winner!=57),harm=int(winner!=57 and vals["MSE"]>ref)))
assert len(rows)==len(seal["decisions"])==120 and updated>0
write(D/"no_preselection.csv",rows);print("120 decisions checked; exact historical forecasts where available:",updated)
