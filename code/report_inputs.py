"""Join verified window-level scores back into the article's task and plotting inputs."""
import shutil
import pandas as pd
import numpy as np
from inputs import METHODS, WORK
from verify_results import KEY,METRICS,compare

def populate(data):
    verified=WORK/"verified"
    def load(path):return pd.read_csv(path,float_precision="round_trip")
    def save(df,path):df.to_csv(path,index=False)
    b,a=[load(verified/(name+".csv")) for name in ("bitbrains","azure")]
    for name,df in (("bitbrains",b),("azure",a)):
        if len(df)!=2880:raise ValueError("Run the complete forecast audit first")
        compare(name,df)
    bi,az=b.set_index(KEY),a.set_index(KEY)
    def update(file,actual,scope,convert=False):
        path=data/file;df=load(path)
        mask=scope(df)
        keys=df.loc[mask,KEY].copy()
        if convert:keys["method"]=keys.method.map(METHODS)
        rows=actual.loc[pd.MultiIndex.from_frame(keys)]
        if not np.array_equal(rows.arch,df.loc[mask,"arch"]):raise ValueError(str(file))
        df.loc[mask,METRICS]=rows[METRICS].to_numpy()
        save(df,path)
    update("task_results.csv",bi,lambda d:~(d.method.eq("PT-FT") & d.budget.eq("tight")))
    update("bitbrains_policy.csv",bi,lambda d:d.method.eq("msa"),True)
    update("azure_external/frozen19.csv",az,lambda d:d.method.eq("msa"),True)
    search=["meta_nas_lite","zero_nas","zero_nas_ft"]
    update("azure_external/baseline_archive.csv",az,
           lambda d:d.method.isin(set(METHODS)-{"msa"}) & (~d.method.isin(search)|d.budget.eq("loose")),True)
    for t in ("tight","medium"):
        update("azure_external/"+t+"_search_verified.csv",az,lambda d:d.method.isin(search),True)
    path=data/"deployment.csv";df=load(path);mask=df.feasible.eq(1)
    align=bi.loc[pd.MultiIndex.from_frame(df.loc[mask,KEY])]
    df.loc[mask,["deployable_MSE","deployable_MAE"]]=align[["MSE","MAE"]].to_numpy()
    save(df,path)
    path=data/"paired_new.csv";df=load(path)
    for method,columns in (("MSA-DTI",["proposed_mse","proposed_worst10"]),("PT-FT",["pt_ft_mse","pt_ft_worst10"])):
        keys=df.assign(method=method,budget="medium")[KEY]
        df[columns]=bi.loc[pd.MultiIndex.from_frame(keys),["MSE","Worst10"]].to_numpy()
    save(df,path)
    path=data/"azure_external/budget_paths_19.csv";df=load(path)
    for tier in ("tight","medium","loose"):
        keys=df.rename(columns={"vm":"VM","h":"H","k":"K"}).assign(method="MSA-DTI",budget=tier)[KEY]
        values=az.loc[pd.MultiIndex.from_frame(keys)]
        if not np.array_equal(df[tier],values.arch):raise ValueError("Azure selection differs")
        df["mse_"+tier]=values.MSE.to_numpy()
    save(df,path)
    for f in ["no_preselection.csv"]+["supplement/"+s+".csv" for s in ("decisions","sources","latency","target_time")]:
        shutil.copyfile(verified/f,data/f)

