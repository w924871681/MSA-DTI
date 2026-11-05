"""Prepare the original cohorts, train on the source VMs and refit the selection rules."""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

from inputs import CODE,WORK,TIERS,cases,materialize,read,file_hash

RUN=WORK/"training"
ORIGINAL=RUN/"original"
REFERENCE=CODE/"experiments/reference"
EXPORT=RUN/"export"
SUP=RUN/"supplement"
VARIANT="n20_s2904"
def write(path,doc):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".pending");tmp.write_text(json.dumps(doc,indent=2),encoding="utf8");os.replace(tmp,path)
def copy(src,dest):
    dest.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(src,dest)

def prepare(raw):
    if not raw.is_file():raise FileNotFoundError(raw)
    contract=read(CODE/"native/protocol/source_development.json")
    if file_hash(raw)!=contract["archive_sha256"]:raise ValueError("Unexpected Bitbrains release")
    shutil.copytree(CODE/"native",ORIGINAL,dirs_exist_ok=True)
    target=ORIGINAL/"data/raw/fastStorage.zip";target.parent.mkdir(parents=True,exist_ok=True)
    if not target.is_file() or file_hash(target)!=contract["archive_sha256"]:copy(raw,target)
    env=dict(os.environ,PYTHONPATH=str(ORIGINAL/"src"),MSADTI_OUTPUT=str(ORIGINAL/"results"),PYTHONUTF8="1")
    if not (ORIGINAL/"results/BITBRAINS_INPUT_LOCK.json").exists():
        subprocess.run([sys.executable,"-B","-m","msa_dti.prepare"],cwd=ORIGINAL,env=env,check=True)
    for stage,rel in {"A":"timing_s50_ab_20261104/A_fresh/native_replay/inputs",
            "B":"timing_s50_ab_20261104/B_engine/cohort","C":"timing_s50_flow_20261104/C_real/inputs",
            "D":"timing_s50_flow_20261104/C_real/fresh_D_stage/inputs",
            "E":"timing_s50_flow_20261104/C_real/fresh_E_stage/inputs"}.items():
        dest=RUN/rel
        shutil.copytree(CODE/"protocol"/stage,dest,dirs_exist_ok=True)
        materialize(stage,raw,dest)
    flow=RUN/"timing_s50_flow_20261104/C_real"
    copy(REFERENCE/"compute.py",flow/"scripts/compute.py")
    shutil.copytree(CODE/"experiments/native",flow/"native",dirs_exist_ok=True)
    cinputs=flow/"inputs"
    copy(ORIGINAL/"data/bitbrains/data/bitbrains_dataset.npz",cinputs/"pt_source.npz")
    write(cinputs/"source_vms.json",read(ORIGINAL/"data/bitbrains/data/bitbrains_dataset_manifest.json")["source_machine_ids"])
    shutil.copytree(ORIGINAL/"results/source_tasks",cinputs/"source_tasks",dirs_exist_ok=True)
    for name in ["source_sensitivity","panels","pipeline"]:copy(REFERENCE/(name+".py"),SUP/(name+".py"))
    copy(CODE/"verification/pipeline_plan.json",SUP/"pipeline_plan.json")
    # Historical baseline workers locate the source-task carrier under flow/C/inputs.
    sub=RUN/"flow/C/inputs"
    shutil.copytree(cinputs/"source_tasks",sub/"source_tasks",dirs_exist_ok=True)
    for name in ["source_stage","teacher_fresh_stage","fresh_hmeta_source"]:copy(REFERENCE/(name+".py"),RUN/(name+".py"))
    write(RUN/"prepared.json",{"archive_sha256":file_hash(raw),"groups":{s:len(cases(s)) for s in "ABCDE"}})

def train_msa():
    import source_sensitivity as source
    source.init()
    for h in (1,4):
        for i in range(66):
            source.train((20,2904,i,h))
            copy(SUP/f"sources/{VARIANT}/H{h}_A{i}.pt",EXPORT/f"weights/msa/strong_H{h}_A{i}.pt")

def policy():
    import pipeline as p
    q=p.q;q.setup(None)
    # The non-anchor code path learns a new bank for this run. A/B/C/D never require E labels.
    name="trained20"
    dest=SUP/"sources"/name
    shutil.copytree(SUP/f"sources/{VARIANT}",dest,dirs_exist_ok=True)
    for stage in "ABCD":
        for task in q.cases(stage):q.task(name,stage,task)
        if stage=="A":ref=p.reference(name,"A",p.joint(range(66)))
        elif stage=="B":bank=p.screen(name,ref)
        elif stage=="C":deployment=p.reference(name,"C",p.joint(bank))
    rules=p.rules(name,bank);policies=p.calibrate(name,deployment,rules)
    write(EXPORT/"policy.json",dict(bank=bank,screening_reference=ref,reference=deployment,tau=policies["full"]["tau"]))

def train_search():
    import source_stage as source
    for h in (1,4):
        for i in range(66):
            dest=source.OUTPUT/f"weights/H{h}/A{i}.pt"
            if dest.exists():
                if file_hash(dest)!=read(source.OUTPUT/f"receipts/H{h}_A{i}.json")["weight_sha256"]:raise ValueError("Incomplete source asset")
            else:source.worker(i,h)
            copy(dest,EXPORT/f"weights/search/H{h}/A{i}.pt")
    import teacher_fresh_stage as teacher
    from msa_dti.teacher import _compute_pi50
    episodes=[];lock=file_hash(RUN/"prepared.json")
    for h in (1,4):
        for k in (10,20):
            for cid in range(20):
                path=teacher.OUT/"episodes"/f"s{cid}_h{h}_k{k}.json"
                if not path.exists():teacher.run_episode(cid,h,k,lock)
                doc=read(path)
                if doc["lock_sha256"]!=lock:raise ValueError("Source teacher input changed")
                episodes.append(doc)
    write(EXPORT/"pi50.json",_compute_pi50(episodes,66))

def train_fixed(method):
    import compute as u
    u.configure_stage2_runtime(u.DEVICE,"gru-native")
    u.OUT=u.ROOT/"baselines"/method
    ranks=[]
    for i in range(66):
        group={};profiles=[]
        for h in (1,4):
            u.work(method,i,h,file_hash(RUN/"prepared.json"),cases("C"),{})
            r=read(u.OUT/f"jobs/{method}_A{i}_H{h}/scores.json")
            profiles.append((r["params"],r["operations"]))
            for row in r["rows"]:group.setdefault(row["vm"],[]).append(row["assessment_mse"])
        if len(group)!=20 or any(len(v)!=4 for v in group.values()):raise ValueError("Incomplete C assessment")
        ranks.append((statistics.mean(statistics.mean(v) for v in group.values()),max(p for p,o in profiles),max(o for p,o in profiles),i))
    idx=min(ranks)[-1];write(EXPORT/(method+".json"),dict(selected=dict(arch_id=idx),ranking=sorted(ranks)))
    if method!="scratch":
        for h in (1,4):copy(u.OUT/f"source/{method}/H{h}/A{idx}.pt",EXPORT/f"weights/{method}/H{h}.pt")

def train_hmeta():
    import fresh_hmeta_source as source
    source.OUT.mkdir(parents=True,exist_ok=True);source.worker()
    copy(source.OUT/"hmeta.pt",EXPORT/"weights/hmeta.pt")

def supplementary_training():
    """Repeat the source variants and the original supplementary rules on one device."""
    import source_sensitivity as source
    plan=read(CODE/"verification/pipeline_plan.json")
    source.init()
    for n,s in plan["variants"]:
        for h in (1,4):
            for i in range(66):source.train((n,s,i,h))
    for h in (1,4):
        for i in range(66):
            name=f"strong_H{h}_A{i}.pt"
            path=WORK/"weights/msa"/name
            if file_hash(path)!=plan["original_weights"][f"{h}/{i}"]:raise ValueError("Wrong control checkpoint")
            copy(path,RUN/"timing_s50_source_20261104/weights"/name)
        copy(WORK/f"weights/pt/H{h}.pt",RUN/f"budget_fair_20261105/inputs/pt/H{h}/A39.pt")
    subprocess.run([sys.executable,"-B",__file__,"supplement-panels"],check=True)

def supplementary_panels():
    import pipeline as p
    q=p.q;q.setup(None)
    for variant in [q.CURRENT]+[f"n{n}_s{s}" for n,s in q.plan()["variants"]]:
        if variant==q.CURRENT:
            bank=q.plan()["bank"];ref=57
        else:
            for stage in "ABC":
                for c in q.cases(stage):q.task(variant,stage,c)
                if stage=="A":screen_ref=p.reference(variant,"A",p.joint(range(66)))
                elif stage=="B":bank=p.screen(variant,screen_ref)
                else:ref=p.reference(variant,"C",p.joint(bank))
        rules=p.rules(variant,bank)
        for c in q.cases("D"):q.task(variant,"D",c)
        policies=p.calibrate(variant,ref,rules)
        for c in q.cases("E"):q.task(variant,"E",c)
        p.seal(variant,ref,rules,policies)
        for c in q.cases("E"):q.score_task(variant,c)
    q.hardware()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["prepare","all","source-smoke","msa","policy","search","pt","medet","scratch","hmeta","supplement","supplement-panels"])
    p.add_argument("--bitbrains",type=Path)
    args=p.parse_args()
    if args.stage in ("prepare","all"):
        if args.bitbrains is None:p.error("provide --bitbrains PATH_TO_FASTSTORAGE_ZIP")
        prepare(args.bitbrains.resolve())
        if args.stage=="prepare":return
    if not (RUN/"prepared.json").is_file():p.error("run prepare first")
    if args.stage=="all":
        for s in ["msa","policy","search","pt","medet","scratch","hmeta"]:
            subprocess.run([sys.executable,"-B",__file__,s],check=True)
        return
    sys.path[:0]=[str(SUP),str(RUN),str(RUN/"timing_s50_flow_20261104/C_real/scripts")]
    if args.stage=="source-smoke":
        import source_sensitivity as source
        source.OUT=RUN/"smoke-weights";source.OUT.mkdir(exist_ok=True)
        source.init();source.train((20,2904,0,1))
        print("A0, H=1 source training receipt:",read(source.OUT/"n20_s2904/H1_A0.json"))
    elif args.stage=="supplement":supplementary_training()
    elif args.stage=="supplement-panels":supplementary_panels()
    elif args.stage=="msa":train_msa()
    elif args.stage=="policy":policy()
    elif args.stage=="search":train_search()
    elif args.stage=="hmeta":train_hmeta()
    else:train_fixed(args.stage)
if __name__=="__main__":main()

