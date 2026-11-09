"""Rebuild the released Azure CPU subset from the 75 hash-locked Microsoft shards."""
import argparse,gzip,hashlib
from pathlib import Path
import numpy as np
from inputs import CODE,WORK,file_hash,read,cases,DTYPE,digest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("shards",type=Path);args=p.parse_args()
    cfg=CODE/"protocol/azure";release=read(cfg/"shards.json");order=read(cfg/"candidate_order.json")["candidates"]
    lookup={r["vm_id"].encode("ascii"):i for i,r in enumerate(order)}
    vals=np.full((len(order),3288),np.nan);counts=np.zeros_like(vals,dtype=np.uint8)
    for name in release["asset_names"][2:]:
        path=args.shards/name
        if file_hash(path)!=release["official_digests"][name].split(":")[-1]:raise ValueError("Unexpected shard: "+name)
        with gzip.open(path,"rb") as stream:
            for line in stream:
                ts,_,rest=line.partition(b",");vm,_,tail=rest.partition(b",")
                i=lookup.get(vm)
                if i is None:continue
                t=int(ts)
                if not 0<=t<=986100 or t%300:continue
                j=t//300;counts[i,j]=min(255,int(counts[i,j])+1)
                try:vals[i,j]=float(tail.rsplit(b",",1)[-1])
                except ValueError:vals[i,j]=np.nan
        print("Scanned",name,flush=True)
    ids=np.flatnonzero(np.all(counts==1,axis=1)&np.all(np.isfinite(vals)&(vals>=0),axis=1))
    selected=read(cfg/"sample_manifest.json")["selected"]
    if len(ids)!=89 or [order[i]["vm_id"] for i in ids[:30]]!=[r["vm_id"] for r in selected]:raise ValueError("Coverage selection differs")
    out=WORK/"azure_series";out.mkdir(parents=True,exist_ok=True)
    for i,r in zip(ids[:30],selected):
        raw=np.full((3000,10),np.nan);raw[:,3]=vals[i,288:3288];ts=np.arange(288,3288,dtype=np.int64)*300
        slug=hashlib.sha256(r["vm_id"].encode()).hexdigest()[:16]
        rows=np.empty(len(ts),dtype=DTYPE);rows["timestamp"]=ts;rows["raw"]=raw
        for c in cases("azure"):
            if c["VM"]==r["vm_id"] and (digest(rows[:c["T"]+1].tobytes())!=c["prefix_hash"] or
                    digest(rows[:c["T"]+c["B"]+c["H"]].tobytes())!=c["future_hash"]):raise ValueError("Azure task carrier differs")
        with np.load(CODE/"data/azure_series"/(slug+".npz"),allow_pickle=False) as frozen:
            np.testing.assert_array_equal(raw,frozen["raw"]);np.testing.assert_array_equal(ts,frozen["timestamp"])
        np.savez_compressed(out/(slug+".npz"),timestamp=ts,raw=raw)
    print("Matched 30 subscriptions and 120 tasks.")
if __name__=="__main__":main()

