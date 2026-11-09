"""Read frozen task identities and reconstruct the exact input carriers from the provider data."""
import hashlib
import io
import json
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
WORK = CODE.parent / "work"
DTYPE = np.dtype([("timestamp", "<i8"), ("raw", "<f8", (10,))])
TIERS = ("tight", "medium", "loose")
METHODS = {"msa":"MSA-DTI", "pt":"PT-FT", "medet":"MeDeT-style", "scratch":"Scratch50",
           "meta_nas_lite":"Meta+NAS-lite", "zero_nas":"Zero-shot NAS",
           "zero_nas_ft":"Zero-shot NAS+FT", "h_meta_nas":"H-Meta-NAS-based"}

def read(path):
    return json.loads(path.read_bytes())

def digest(data):
    return hashlib.sha256(data).hexdigest()

def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def cases(stage):
    data = read(CODE/"protocol"/stage/("tasks.json" if stage=="A" else "cases.json"))
    return data["cases"] if stage=="B" else data

@lru_cache(maxsize=2)
def provider(path):
    expected = read(CODE/"native/protocol/source_development.json")["archive_sha256"]
    if file_hash(path) != expected:
        raise ValueError(f"fastStorage.zip SHA-256 must be {expected}")
    return zipfile.ZipFile(path)

@lru_cache(maxsize=4)
def bit_rows(path, name, parser):
    data = provider(path).read(name)
    if parser == "pandas":
        a = pd.read_csv(io.BytesIO(data), sep=";").to_numpy(np.float64)
    else:
        def number(field):
            try:
                return float(field)
            except ValueError:
                return float("nan")
        a = np.array([[number(s) for s in line.split(b";")] for line in data.splitlines()[1:] if line.strip()])
    rows = np.empty(len(a), dtype=DTYPE)
    rows["timestamp"], rows["raw"] = a[:,0], a[:,1:]
    return rows

def carrier(stage, c, raw_path, *, prefix=False):
    if stage == "azure":
        slug = c["slug"].split("_")[0]
        with np.load(CODE/"data/azure_series"/(slug+".npz"), allow_pickle=False) as d:
            rows = np.empty(len(d["timestamp"]), dtype=DTYPE)
            rows["timestamp"], rows["raw"] = d["timestamp"], d["raw"]
    else:
        name = c.get("raw_member", f"fastStorage/2013-8/{c['VM'].split(':')[-1]}.csv")
        rows = bit_rows(raw_path, name, "pandas" if stage=="A" else "float")[c["native_start"]:]
    end = c["T"]+1 if prefix else c["T"]+c["B"]+c["H"]
    return rows[:end].tobytes()

def materialize(stage, raw_path, destination):
    for c in cases(stage):
        slug = c["slug"]+".bin"
        pre = carrier(stage,c,raw_path,prefix=True)
        if stage=="A":
            expected = read(CODE/"protocol/A/sha256.json")["prefixes/"+slug]
        else:
            expected = c.get("prefix_hash",c.get("prefix_sha256"))
        parts = [("prefixes" if stage in ("A","B") else "prefix", pre, expected)]
        if stage!="A":
            full = carrier(stage,c,raw_path)
            if stage in ("C","D"):
                parts.append(("assessment",full[len(pre):],c["post_sha256"]))
            elif stage=="B":
                parts.append(("check",full,c["check_sha256"]))
            else:
                parts.append(("future",full,c["future_hash"]))
        for folder, payload, expected in parts:
            if digest(payload)!=expected:
                raise ValueError(f"Input fingerprint mismatch: {stage}/{folder}/{slug}")
            out=destination/folder/slug
            out.parent.mkdir(parents=True,exist_ok=True)
            out.write_bytes(payload)

def truth(c, raw):
    if digest(raw)!=c["future_hash"]:
        raise ValueError("Incorrect scoring carrier for "+c["slug"])
    rows=np.frombuffer(raw,dtype=DTYPE)
    origins=np.arange(c["T"],c["T"]+c["B"])
    y=rows["raw"][origins[:,None]+np.arange(1,c["H"]+1),3]
    valid=np.all(np.isfinite(y)&(y>=0),axis=1)
    return (y[valid]/100).astype(np.float32),valid

def metrics(prediction, y):
    if prediction.shape!=y.shape or not np.isfinite(prediction).all():
        raise ValueError("Prediction shape or finite-value check failed")
    delta=prediction.astype(np.float64)-y.astype(np.float64)
    losses=np.square(delta).mean(1)
    worst=max(1,round(.10*len(losses)))
    return dict(MSE=losses.mean(),MAE=np.abs(delta).mean(),Worst10=np.sort(losses)[-worst:].mean())

