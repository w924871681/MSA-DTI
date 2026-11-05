"""Generate supplementary tables and figures from verified task records."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import unified_style as s

D=s.DATA/"supplement"; T=s.ROOT/"tables"; F=s.ROOT/"figures"; MAIN="n20_s2904"
def obj(name):return json.loads((s.DATA/name).read_bytes())
def read(name):return pd.read_csv(D/name,float_precision="round_trip")
def tex(file,text):(T/(file+".tex")).write_text(text+"\n",encoding="utf8")
def tab(file,title,head,rows,align,group_header=None):
    lines=[r"\begin{center}\begin{minipage}{\linewidth}\centering",r"\captionof{table}{"+title+"}",r"\small\setlength{\tabcolsep}{4.2pt}",
           r"\begin{tabular}{"+align+"}",r"\toprule"]
    if group_header:
        lines.append(group_header)
    lines.append(" & ".join(head)+r"\\\midrule")
    lines+=[" & ".join(map(str,row))+r"\\" for row in rows]
    lines.extend([r"\bottomrule\end{tabular}"])
    tex(file,"\n".join(lines+[r"\end{minipage}\end{center}"]))
def tau(x):return "Reference only" if pd.isna(x) else f"{100*x:g}"
def static():
    windows=[(1,10,201,1551,1552,627),(1,20,211,1556,1557,625),(4,10,204,1549,1553,624),(4,20,214,1555,1559,621)]
    tab("sup_windows","Task windows in zero-based record indices.",
        ["$H$","$K$","Validation origins","Windows before masking","$T$","Forecast origins"],
        [[h,k,f"{a}--{b}",b-a+1,t,f"{t}--{t+n-1}"] for h,k,a,b,t,n in windows],"rrrrrr")
    bank=obj("screening_bank.json"); profiles=obj("azure_external/resources.json")["profiles"]
    rows=[]
    for r in bank["members"]:
        i=r["arch_id"];a=r["architecture"];p,o=profiles["1"][str(i)];p4,o4=profiles["4"][str(i)]
        rows.append([f"A{i}",a["arch_key"].replace("_",r"\_"),r["check_wins"],r["validation_wins"],f"{p:,} / {p4:,}",f"{o/1e6:.3f} / {o4/1e6:.3f}"])
    tab("sup_bank","Retained source checkpoints and group-B evidence.",
        ["ID","Architecture","Check wins","Val. wins",r"\shortstack[r]{Parameters\\$H=1/4$}",r"\shortstack[r]{Est. operations ($10^6$)\\per forecast ($H=1/4$)}"],rows,"llrrrr")
    w=read("screening_cases.csv");v=sorted(w.vm.unique(),key=lambda x:int(x.split(":")[-1]))
    def members(group):
        y=w[w.vm.isin(group)]
        return {55}|set(y.check_point.value_counts().loc[lambda x:x>=2].index)|set(y.validation_point.value_counts().loc[lambda x:x>=2].index)
    alls=members(v);assert alls==set(bank["retained_including_comparator"])
    counts=[len(members(g)) for g in np.array_split(v,3)]
    tex("sup_b_subsets_text",f"As a check on how much evidence a smaller screening group supplies, we sorted the 20 group-B VMs by identifier and split them into groups of seven, seven, and six. Applying the same two-point requirement to these groups gave {counts[0]}, {counts[1]}, and {counts[2]} retained models, respectively, including A55. The full group retained 19. These counts describe screening only.")
    g=obj("frozen_policy.json")
    tab("sup_grid","Group-D calibration on 80 tasks at the medium budget.",
        [r"Margin (\%)",r"Assessment MSE ($10^{-2}$)","Harmful replacements","Replacements"],
        [[f"{100*r['tau']:g}",f"{100*r['mse']:.4f}",f"{round(r['harm_rate']*80)}/80",f"{round(r['switch_rate']*80)}/80"] for r in g["trials"]],"rrrr")
def summarize(frame):
    assert len(frame)==120 and frame.groupby("VM").size().eq(4).all()
    means=frame.groupby("VM")[["MSE","MAE","Worst10"]].mean().mean().to_dict()
    return dict(**means,CVaR90=frame.MSE.nlargest(12).mean(),harm=int(frame.harm.sum()),switch=int(frame.switch.sum()),feasible=int(frame.feasible.sum()),tau=frame.tau.iloc[0])
def stats():
    df=read("decisions.csv")
    out={(v,k):summarize(x) for (v,k),x in df.groupby(["variant","rule"])}
    def metric(rule):return out[MAIN,rule]
    sizes=[1,4,8,12,17];steps=[0,1,5,10,20,50]
    bank=[metric(f"subset{n}" if n<17 else "full") for n in sizes]
    updates=[metric(f"steps{n}" if n<50 else "full") for n in steps]
    cal=obj("frozen_policy.json");g=cal["trials"];fig=s.canvas(184)
    axes=[s.axis(fig,x,48,96,119) for x in (45,201,355)]
    axes[0].plot(sizes,[100*x["MSE"] for x in bank],"o-",markersize=3.4,color=s.BLUE)
    axes[0].set(xlabel="Candidates",ylabel=r"Test MSE ($\times10^{-2}$)",xticks=sizes)
    axes[1].plot(steps,[100*x["MSE"] for x in updates],"s-",markersize=3.4,color=s.GREEN)
    axes[1].set(xlabel="SGD updates",ylabel=r"Test MSE ($\times10^{-2}$)",xticks=[0,10,20,50])
    axes[2].plot([100*x["tau"] for x in g],[100*x["mse"] for x in g],"o-",markersize=3.4,color=s.BLUE)
    axes[2].axvline(100*cal["tau"],ls="--",lw=.8,color=s.ORANGE)
    axes[2].set(xlabel="Margin (%)",ylabel=r"Cal. MSE ($\times10^{-2}$)",xticks=[0,10,20])
    for ax in axes:
        s.clean(ax)
    s.panel_row(fig,zip(axes,["(a) Candidate subsets","(b) Target updates","(c) Calibration"]))
    s.save(fig,F,"figS1",["supplement/decisions.csv","frozen_policy.json"],{"stages":["E","E","D"]})
    tex("sup_curve_text",f"With only A57 available, mean test MSE was {bank[0]['MSE']:.6f}. With the full set of 17 feasible candidates it was {bank[-1]['MSE']:.6f}. With the same 17 candidates but no target updates, MSE was {updates[0]['MSE']:.6f}.")
    controls=[("SGD, calibrated","full"),("SGD, direct minimum","sgd_direct"),("Adam, calibrated","adam"),("Adam, direct minimum","adam_direct")]
    tab("sup_sensitivity","Optimizer-matched selection at the medium budget.",
        ["Update and selection rule",r"Margin (\%)",r"MSE ($10^{-2}$)",r"MAE ($10^{-2}$)","Harm / 120"],
        [[n,tau(metric(k)["tau"]),f"{100*metric(k)['MSE']:.4f}",f"{100*metric(k)['MAE']:.4f}",metric(k)["harm"]] for n,k in controls],"lrrrr")
    tex("sup_opt_text","We adapted all 17 models for 50 updates with either SGD (learning rate 0.01) or Adam (0.001). For each optimizer, direct selection chose the lowest validation MSE. The calibrated rule applied the replacement margin to the same validation scores.")
    sources=read("sources.csv");rows=[]
    for r in sources.itertuples():
        st=out[r.variant,"full"];n,seed=r.variant[1:].split("_s")
        rows.append([n,seed,f"A{r.screen_ref}",r.candidates,f"A{r.reference}",tau(st["tau"]),f"{100*st['MSE']:.4f}",f"{100*st['MAE']:.4f}",st["harm"]])
    rows.sort(key=lambda x:(int(x[1]),int(x[0])))
    tab("sup_sources","Effect of source-pool size and seed on medium-budget evaluation.",
        ["VMs","Seed","Screen ref.","Bank","Deploy ref.",r"Margin (\%)",r"MSE ($10^{-2}$)",r"MAE ($10^{-2}$)","Harm / 120"],rows,"rrlrlrrrr")
    rows=[]
    for tier in ["tight","medium","loose"]:
        for mask,label in [("parameters","Parameters"),("operations","Operations"),("both","Both")]:
            k=f"{tier}_{mask}";st=metric(k);x=df[(df.variant==MAIN)&(df.rule==k)]
            lo,hi=int(x.eligible.min()),int(x.eligible.max())
            rows.append([tier.title(),label,lo if lo==hi else f"{lo}--{hi}",f"{st['feasible']}/120",f"{100*st['MSE']:.4f}",st["harm"]])
    tab("sup_caps","Resource compliance and test error under each filtering rule.",
        ["Budget","Active limits","Candidates","Meet both limits",r"MSE ($10^{-2}$)","Harm / 120"],rows,"llrrrr")
    original=df.loc[df.variant==MAIN].set_index(["rule","VM","H","K"]).sort_index()
    tiers=["tight","medium","loose"]
    counts={};differences={}
    ids=sorted(obj("screening_bank.json")["retained_including_comparator"])
    for tier in tiers:
        both=original.loc[f"{tier}_both"]
        assert int(both.feasible.sum())==120
        for mask in ("parameters","operations"):
            active=original.loc[f"{tier}_{mask}"]
            assert active.index.equals(both.index)
            changed=int(active.arch.ne(both.arch).sum())
            violations=int(active.feasible.eq(0).sum())
            assert 0<=violations<=changed
            counts[tier,mask]=(changed,violations)
            differences[tier,mask]=active.arch.value_counts().reindex(ids,fill_value=0)-both.arch.value_counts().reindex(ids,fill_value=0)
    assert not any(counts["loose",mask][0] for mask in ("parameters","operations"))
    affected=[i for i in ids if any(d.loc[i] for d in differences.values())]
    if not affected:
        affected=ids
    fig=s.canvas(120+13*len(affected))
    axes=[s.axis(fig,x,49,161,42+13*len(affected)) for x in (46,278)]
    maximum=max(2,max(int(d.abs().max()) for d in differences.values())+1)
    for ax,tier in zip(axes,tiers[:2]):
        for offset,mask,label,marker,color in [
                (-.13,"parameters","Parameters only","o",s.BLUE),
                (.13,"operations","Operations only","s",s.ORANGE)]:
            values=differences[tier,mask].loc[affected].to_numpy()
            y=np.arange(len(affected))+offset
            ax.hlines(y,0,values,lw=1,color=color)
            ax.scatter(values,y,s=20,marker=marker,facecolor="white",edgecolor=color,label=label,zorder=3)
        ax.set_yticks(np.arange(len(affected)),[f"A{i}" for i in affected]);ax.invert_yaxis()
        ax.set_xlim(-maximum,maximum);ax.axvline(0,color=s.GREY,lw=.8)
        ax.set_xlabel("Change in number of selections")
        s.clean(ax,"x")
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=2,frameon=False)
    s.panel_row(fig,zip(axes,["(a) Tight budget","(b) Medium budget"]))
    s.save(fig,F,"figS2",["supplement/decisions.csv"],{"paired_tasks_per_budget":120,"comparison":"frequency relative to both limits"})
    p=counts["tight","parameters"];o=counts["tight","operations"]
    tex("sup_caps_text",f"At the tight budget, omitting the parameter limit produced {o[1]} outputs above that limit. Omitting the operation limit gave {p[1]} outputs above the operation limit. Test MSE in the table includes these outputs.")
def hardware():
    a=read("latency.csv");system=json.loads((D/"hardware.json").read_bytes());assert len(a)==76
    device_name=lambda x:str(x).replace("_",r"\_").replace("&",r"\&").replace("@","at")
    tex("sup_hardware_text",f"Latency was measured on an {device_name(system['cpu'])} CPU with one thread and a {device_name(system['gpu'])} GPU using PyTorch {device_name(system['torch'])}. A single zero-valued input has shape $1\\times96\\times25$. Each checkpoint receives 100 warmup calls followed by five blocks of 1,000 forward passes. We synchronize the GPU before and after each timed call. The input and model are already on the measurement device. GRUs use the native backend used during adaptation.")
    fig=s.canvas(302);axes=[s.axis(fig,x,49,161,224) for x in (50,278)]
    ids=sorted(a.arch.unique())
    for ax,kind in zip(axes,["cpu","cuda"]):
        for h,offset,m,color in [(1,.14,"o",s.BLUE),(4,-.14,"s",s.ORANGE)]:
            data=a[(a.device==kind)&(a.H==h)].set_index("arch").loc[ids]
            y=np.arange(len(ids))+offset
            ax.hlines(y,data.median_ms,data.p95_ms,colors=color,lw=1)
            ax.scatter(data.median_ms,y,s=11,marker=m,color=color,label=f"H = {h}")
        ax.set_xscale("log");ax.set_yticks(np.arange(len(ids)),["A"+str(i) for i in ids]);ax.invert_yaxis()
        ax.set_xlabel("Latency (ms)")
        s.clean(ax,"x")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=2,frameon=False)
    s.panel_row(fig,zip(axes,["(a) CPU, one thread","(b) GPU, synchronized"]))
    s.save(fig,F,"figS3",["supplement/latency.csv","supplement/hardware.json"],{"statistic":"median to p95","calls":5000})
    ti=read("target_time.csv"); vm=ti.VM.unique().item().split(":")[-1];rows=[]
    for h in (1,4):
        for k in (10,20):
            line=[h,k]
            for tier in ["tight","medium","loose"]:
                for method in ["msa","pt"]:
                    x=ti[(ti.H==h)&(ti.K==k)&(ti.budget==tier)&(ti.method==method)].iloc[0]
                    line.append(f"{x.mean_seconds:.3f}")
            rows.append(line)
    tex("sup_time_intro",f"We used VM {vm}, the lowest numbered VM in group D. For each task and budget, we ran one warmup and five timed selections on an otherwise idle GPU. Timing starts with resource filtering and checkpoint loading, with task features already constructed. It includes adaptation, validation, and the final choice. Fixed PT-FT adapts A39 at every budget. At the tight budget its time describes execution of a model that exceeds the operation limit.")
    tab("sup_time","Mean target selection time in seconds over five repetitions.",
        ["$H$","$K$","MSA-DTI","PT-FT","MSA-DTI","PT-FT","MSA-DTI","PT-FT"],rows,"rrrrrrrr",
        group_header=r" & & \multicolumn{2}{c}{Tight} & \multicolumn{2}{c}{Medium} & \multicolumn{2}{c}{Loose}\\\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}")
if __name__=="__main__":
    static();stats();hardware()
