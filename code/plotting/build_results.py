"""Derive the manuscript tables and plot inputs from paired archived task records."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, TABLES = ROOT / "data", ROOT / "tables"
KEY = ["VM", "H", "K"]
TIERS = ["tight", "medium", "loose"]
METHODS = ["PT-FT","MeDeT-style","Scratch50","Meta+NAS-lite","Zero-shot NAS",
           "Zero-shot NAS+FT","H-Meta-NAS-based","MSA-DTI"]
FIELDS = ["MSE","MAE","Worst10"]
SEED = 20261026


def read(name):
    return pd.read_csv(DATA/name, float_precision="round_trip")


def indexed(df):
    out = df.set_index(KEY).sort_index()
    assert out.index.is_unique and len(out)==120
    assert (out.groupby(level="VM").size()==4).all()
    return out


def metrics(df):
    means = df.groupby(level="VM")[FIELDS].mean().mean().to_dict()
    means["CVaR90"] = df.MSE.nlargest(12).mean()
    return means


def table(name,label,caption,heads,rows,alignment=None,group_header=None):
    alignment = alignment or "l"+"r"*(len(heads)-1)
    position = "!t" if name=="azure_external" else "!htbp"
    lines=[r"\begin{table}[pos="+position+"]",r"\centering\footnotesize",
           r"\caption{"+caption+r"}\label{"+label+"}",r"\begin{tabular}{"+alignment+"}",
           r"\toprule"]
    if group_header:
        lines.append(group_header)
    lines.append(" & ".join(heads)+r"\\\midrule")
    lines += [" & ".join(map(str,row))+r"\\" for row in rows]
    lines += [r"\bottomrule",r"\end{tabular}"]
    (TABLES/(name+".tex")).write_text("\n".join(lines+[r"\end{table}",""]),encoding="utf-8")


raw=read("task_results.csv")
dep=read("deployment.csv")
pol=read("bitbrains_policy.csv")
pair=indexed(read("paired_new.csv"))
assert len(raw)==len(dep)==2880 and len(pol)==1080
med=raw[raw.budget=="medium"]
by={m:indexed(med[med.method==m]) for m in METHODS}
for m,v in by.items():
    z=indexed(dep[(dep.method==m)&(dep.budget=="medium")])
    assert z.index.equals(pair.index) and v.index.equals(pair.index)
    assert (z.feasible==1).all() and np.array_equal(z.arch,v.arch) and np.allclose(z.deployable_MSE,v.MSE,rtol=1e-11,atol=1e-17)
assert np.allclose(by["PT-FT"].MSE,pair.pt_ft_mse) and np.allclose(by["MSA-DTI"].MSE,pair.proposed_mse)
overall=pd.DataFrame([{"Method":m,**metrics(by[m])} for m in METHODS])
overall.to_csv(DATA/"overall_new.csv",index=False)

# Resample VMs as clusters, retaining the four paired task observations of each VM.
gain=100*(1-pair.proposed_mse/pair.pt_ft_mse)
draws=np.random.default_rng(SEED).integers(0,30,size=(10000,30))
def interval(values):
    return np.quantile(values[draws].mean(axis=1),[.025,.975])
sub=[]
def add_group(panel,name,series):
    v=series.groupby(level="VM").mean().sort_index().to_numpy()
    lo,hi=interval(v)
    sub.append(dict(panel=panel,group=name,gain_percent=v.mean(),ci_low_percent=lo,ci_high_percent=hi))
for h,k in [(1,10),(1,20),(4,10),(4,20)]:
    add_group("H/K",f"H={h}, K={k}",gain.loc[(slice(None),h,k)])
for t in TIERS:
    g=dep[(dep.method=="PT-FT")&(dep.budget==t)]
    if not g.feasible.all():
        assert t=="tight" and not g.feasible.any()
        continue
    pt=indexed(g).deployable_MSE
    ours=indexed(dep[(dep.method=="MSA-DTI")&(dep.budget==t)]).deployable_MSE
    add_group("Deployment budget",t.title(),100*(1-ours/pt))
pd.DataFrame(sub).to_csv(DATA/"subgroups.csv",index=False)
policy_by={m:indexed(pol[(pol.method==m)&(pol.budget=="medium")]) for m in ("msa","reference_only","direct_minimum")}
m,r,d=[policy_by[k] for k in ("msa","reference_only","direct_minimum")]
assert np.array_equal(m.arch,by["MSA-DTI"].arch) and np.allclose(m.MSE,by["MSA-DTI"].MSE)
def decisions(v):
    on=v.arch!=r.arch
    assert (v.loc[~on,"MSE"]==r.loc[~on,"MSE"]).all()
    return {"switched":int(on.sum()),"harm":int((on&(v.MSE>r.MSE)).sum()),"benefit":int((on&(v.MSE<r.MSE)).sum())}
# All four rows use the original 120 medium-budget tasks and A57 as the reference.
unfiltered=indexed(pol[(pol.method=="msa")&(pol.budget=="loose")])
expanded=indexed(read("no_preselection.csv"))
assert expanded.index.equals(m.index) and unfiltered.index.equals(m.index) and (expanded.adapted==48).all()
profiles=json.loads((DATA/"budget_resource_profiles_new.json").read_bytes())
def coverage(block):
    return sum(profiles[str(h)][str(i)]["parameters"]<=100000 and profiles[str(h)][str(i)]["operations"]<=5000000
               for (_,h,_),i in block.arch.items())
abl=[];ablation_stats={}
for code,name,block,n in [("full","MSA-DTI",m,17),("no_preselection","No preselection",expanded,48),
                           ("no_filter","No deployment filter",unfiltered,19),("no_margin","No margin",d,17)]:
    assert block.index.equals(r.index)
    vals=metrics(block);count=decisions(block);feasible=coverage(block) if code!="no_preselection" else int(expanded.feasible.sum())
    if code=="no_preselection":assert int(expanded.harm.sum())==count["harm"]
    assert feasible==(96 if code=="no_filter" else 120)
    ablation_stats[code]=dict(adapted=n,feasible=feasible,harm=count["harm"],**vals)
    abl.append([name,str(n),f"{feasible}/120",*[f"{vals[k]*100:.4f}" for k in FIELDS+["CVaR90"]],f"{100*count['harm']/120:.2f}"])
table("ablation","tab:ablation","Effect of removing one component at the medium Bitbrains budget.",
      ["Rule","Adapted","Feasible","MSE","MAE",r"Worst-10\%","CVaR90",r"Harm (\%)"],abl,
      group_header=r" & & & \multicolumn{4}{c}{Prediction error ($\times 10^{-2}$)} & \\\cmidrule(lr){4-7}")
pools=pol[pol.method=="msa"].copy()
pools.to_csv(DATA/"budget_selection_new.csv",index=False)
sens=[]
for t in TIERS:
    group=indexed(pools[pools.budget==t]); vals=metrics(group)
    sens.append([t.title(),int(group.feasible.iloc[0]),group.arch.nunique(),*[f"{vals[k]:.6f}" for k in FIELDS+["CVaR90"]]])
table("source_sensitivity","tab:source-sensitivity","Candidate availability and prediction errors under the three deployment budgets.",["Budget","Feasible","Used","MSE","MAE","Worst-10\\%","CVaR90"],sens)
clock=read("timing.csv").set_index("method");cost=[]
for name in METHODS:
    s=clock.loc[name];target=s.prior_target_min+(0 if name=="PT-FT" else s.new_target_min)
    cover=[int(dep[(dep.method==name)&(dep.budget==tier)].feasible.sum()) for tier in TIERS]
    cost.append([name,*[str(v) for v in cover],f"{s.offline_min:.2f}",f"{target:.2f}",f"{s.offline_min+target:.2f}"])
table("cost","tab:cost","Deployment coverage and recorded time through model selection.",
      ["Method","Tight","Medium","Loose","Source/devel.","Target","Total"],cost,
      group_header=r" & \multicolumn{3}{c}{Feasible tasks (out of 120)} & \multicolumn{3}{c}{Time (min)}\\\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
az=read("azure_external/baseline_archive.csv"); new=read("azure_external/frozen19.csv")
codes=["pt","medet","scratch","meta_nas_lite","zero_nas","zero_nas_ft","h_meta_nas","msa"]
search_methods={"meta_nas_lite","zero_nas","zero_nas_ft"}
search={tier:read(f"azure_external/{tier}_search_verified.csv") for tier in ("tight","medium")}
assert all(len(frame)==360 and set(frame.method)==search_methods for frame in search.values())
az_limits=json.loads((DATA/"azure_external/resources.json").read_text())
azstats={}; azrows=[]
for name,code in zip(METHODS,codes):
    good=(new if code=="msa" else az); vals={}
    cells=[name]
    for t in TIERS:
        source=search[t] if t in search and code in search_methods else good
        block=indexed(source[(source.method==code)&(source.budget==t)])
        assert block.index.equals(indexed(new[(new.method=="msa")&(new.budget==t)]).index)
        fits=[all(n<=cap for n,cap in zip(az_limits["profiles"][str(h)][str(int(i))],az_limits["caps"][t])) for (_,h,_),i in block.arch.items()]
        if code!="msa":assert fits==[bool(f) for f in block.feasible]
        count=sum(fits)
        assert count==(0 if code=="pt" and t=="tight" else 120)
        vals[t]={**metrics(block),"feasible":count}
        if count == 0:
            cells.append(r"\multicolumn{2}{c}{Over budget}")
        else:
            cells.extend(f"{1000 * vals[t][k]:.3f}" for k in ("MSE", "MAE"))
    azstats[name]=vals
    azrows.append(cells)
table("azure_external","tab:azure-external","Prediction errors on the same 30 Azure subscriptions under three deployment budgets.",
      [r"Method",r"MSE",r"MAE",r"MSE",r"MAE",r"MSE",r"MAE"],azrows,
      group_header=r"Errors ($\times 10^{-3}$) & \multicolumn{2}{c}{Tight} & \multicolumn{2}{c}{Medium} & \multicolumn{2}{c}{Loose}\\\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
ptv=pair.pt_ft_mse.groupby(level="VM").mean().to_numpy()
msv=pair.proposed_mse.groupby(level="VM").mean().to_numpy()
red=100*(1-msv[draws].mean(axis=1)/ptv[draws].mean(axis=1))
changed=(m.arch!=d.arch)
# The loose tier admits the full retained bank. Check those same choices against each cap.
profiles=json.loads((DATA/"budget_resource_profiles_new.json").read_bytes())
loose=indexed(pools[pools.budget=="loose"]).reset_index()
violations={}
for tier,(par,ops) in zip(TIERS,[(30000,1500000),(100000,5000000),(500000,20000000)]):
    selected=[profiles[str(row.H)][str(row.arch)] for row in loose.itertuples()]
    violations[tier]=sum(r["parameters"]>par or r["operations"]>ops for r in selected)
assert violations==dict(tight=50,medium=24,loose=0)
res={"ablation":ablation_stats,"without_target_filter_violations":violations,"aggregate_CI":np.quantile(red,[.025,.975]).tolist(),"relative":float(gain.mean()),"VMwins":int(sum(msv<ptv)),
     "taskwins":int(sum(gain>0)),"decision":{k:decisions(v) for k,v in policy_by.items()},"disagree":int(changed.sum()),
     "avoided":int((changed&(d.MSE>m.MSE)).sum()),"missed":int((changed&(d.MSE<m.MSE)).sum()),
     "subgroups":sub,"azure":azstats}
(DATA/"summary.json").write_text(json.dumps(res,indent=2),encoding="utf-8")
print({k:v for k,v in res.items() if k not in ("azure","subgroups")});print(sens);print(abl);print(azrows);print(sub)
