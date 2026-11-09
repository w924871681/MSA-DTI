"""Refit the four medium-budget ablation arms, then score the sealed architecture choices."""
import argparse,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from inputs import CODE,WORK,cases,carrier,read,file_hash,truth,metrics
from run_targets import u,torch,PLAN,POLICY,load,target_features,fit,DTYPE,causal_features,WEIGHTS

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--bitbrains",type=Path,required=True)
    p.add_argument("--tasks",type=int,default=120);args=p.parse_args()
    if not 1<=args.tasks<=120:p.error("Choose 1 to 120 tasks")
    ref=POLICY["reference"];bank=PLAN["bank"];records=[]
    with tempfile.TemporaryDirectory(dir=WORK,prefix="ablation-") as temp:
        folder=Path(temp);decisions=[]
        for c in cases("E")[:args.tasks]:
            d=target_features("E",c,args.bitbrains,folder)
            xs,ys,xv=[torch.from_numpy(d[k]).to(u.DEVICE).contiguous() for k in ("Xs","ys","Xv")]
            h=c["H"];profile={i:tuple(u.profile_arch(a,96,25,h)) for i,a in enumerate(u.ARCH)}
            admissible=[i for i in profile if all(x<=y for x,y in zip(profile[i],PLAN["caps"]["medium"]))]
            eligible=[i for i in bank if i in admissible]
            losses={};states={}
            for i in sorted(set(admissible)|set(bank)):
                model,val=fit(i,h,c["seed"],load(WEIGHTS/f"msa/strong_H{h}_A{i}.pt")["state_dict"],xs,ys,xv)
                losses[i]=np.square(val.astype(np.float64)-d["yv"].astype(np.float64)).mean(1).mean()
                states[i]={k:v.detach().cpu() for k,v in model.state_dict().items()}
            choices={}
            for name,pool,tau in [("full",eligible,POLICY["tau"]),("no_preselection",admissible,POLICY["tau"]),
                    ("no_filter",bank,POLICY["tau"]),("no_margin",eligible,0)]:
                candidates=[i for i in pool if i==ref or losses[i]<=(1-tau)*losses[ref]]
                i=min(candidates,key=lambda j:(losses[j],*profile[j],j))
                choices[name]=(i,len(pool),i in admissible)
            path=folder/(c["slug"]+".pt");ids={v[0] for v in choices.values()}|{ref}
            torch.save({i:states[i] for i in ids},path);decisions.append((c,choices,path,file_hash(path)))
        norm=read(CODE/"protocol/E/normalization.json")
        for c,choices,path,sha in decisions:
            if file_hash(path)!=sha:raise ValueError("Sealed model state changed")
            raw=carrier("E",c,args.bitbrains);y,valid=truth(c,raw);r=np.frombuffer(raw,dtype=DTYPE)
            feat=causal_features(r["timestamp"],r["raw"],norm)
            x=torch.from_numpy(np.stack([feat[o-95:o+1] for o in range(c["T"],c["T"]+c["B"])]).astype(np.float32)).to(u.DEVICE)
            stats={}
            for i,st in load(path).items():
                model=u.build(i,c["H"],st,c["seed"]);model.eval()
                with u.candidate_backend_context(u.ARCH[i],u.DEVICE,"gru-native"),torch.no_grad():
                    stats[i]=metrics(model(x).cpu().numpy()[valid],y)
            for name,(i,n,feasible) in choices.items():
                records.append(dict(VM=c["VM"],H=c["H"],K=c["K"],arm=name,arch=i,eligible=n,feasible=int(feasible),
                                    harm=int(i!=ref and stats[i]["MSE"]>stats[ref]["MSE"]),**stats[i]))
    out=WORK/"reruns";out.mkdir(exist_ok=True);frame=pd.DataFrame(records)
    frame.to_csv(out/f"ablation-{args.tasks}.csv",index=False)
    print(frame.groupby("arm").agg(MSE=("MSE","mean"),MAE=("MAE","mean"),feasible=("feasible","sum"),harm=("harm","sum")))
if __name__=="__main__":main()

