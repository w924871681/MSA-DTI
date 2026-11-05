"""Deterministic runtime shared by source and target experiments."""
import os,sys,random,hashlib
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from .io import *
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
from core.config import load_and_merge
from core.space import build_model,profile_arch,enumerate_A_base
from core.methods.ours.stage2_runtime import candidate_backend_context,configure_stage2_runtime
from core.methods.ours.weight_bank import make_bank_key_shared
from . import baseline_methods as native
from . import raw_data as data
DEVICE=torch.device(os.environ.get('MSADTI_DEVICE','cuda'))
configure_stage2_runtime(DEVICE,'gru-native')
CFG=load_and_merge('ours',main_module='configs.main_cfg',methods_pkg='configs.methods',smoke=False)
CFG.main.sim.seed=2904
ARCH=enumerate_A_base(CFG.main.arch)
assert len(ARCH)==66
DS=read(DATA/'bitbrains/data/bitbrains_dataset_manifest.json')
ASSETS=read(ROOT/'protocol/source_recipes.json')['checkpoints']

def seed(s):
    random.seed(s);np.random.seed(s%(2**32-1));torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
def case_seed(c,base=2904):return base+int(hashlib.sha256(c['machine_id'].encode()).hexdigest()[:8],16)+37*c['H']+53*c['K']
def role(c,name):
    assert name in ['support','validation','check']
    norm=read(DATA/'bitbrains/data/source_normalization.json');lo,hi=c['roles'][name]['raw_segment_bounds_0based_halfopen']
    with np.load(DATA/c['target_series_path']) as z:
        x,y=data.windows(z['timestamps'][lo:hi],z['raw_metrics'][lo:hi],norm,c['H'],c['roles'][name]['origin_segment_indices'])
    return torch.from_numpy(x),torch.from_numpy(y)
def source_task(cid,H,K):
    with np.load(RUN/'source_tasks'/f's{cid}_h{H}_k{K}.npz',allow_pickle=False) as a:return tuple(torch.from_numpy(a[k].copy()) for k in ['Xs','ys','Xq','yq'])
def build(idx,H,state=None,s=2904):
    seed(s);m=build_model(ARCH[idx],input_dim=25,H=H,L=96,device=str(DEVICE))
    if state is not None:m.load_state_dict(state,strict=True)
    return m
def update(m,x,y,steps,s,search=False):
    if steps!=50:raise ValueError('The paper uses exactly 50 adaptation steps')
    with torch.backends.cudnn.flags(enabled=False):
        if search:native._search_target_adapt(m,x,y,seed=s)
        else:native._fixed_target_adapt(m,x,y,seed=s)
def pred(m,x,idx):
    m.eval()
    with candidate_backend_context(ARCH[idx],DEVICE,'gru-native'),torch.no_grad():return m(x.to(DEVICE).contiguous()).detach().cpu().numpy()
def metrics(p,y):
    z=np.asarray(p,dtype=float)-np.asarray(y,dtype=float);se=np.square(z).mean(1)
    return dict(weighted_mse=float(se.mean()),mae=float(abs(z).mean()),worst10=float(np.sort(se)[-max(1,round(.1*len(se))):].mean()))
def progress(j,done,total,detail):
    runtime=dict(pid=os.getpid())
    try:
        import psutil
        proc=psutil.Process();runtime.update(rss_bytes=proc.memory_info().rss,process_started=proc.create_time())
        if torch.cuda.is_initialized():runtime.update(cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except (ImportError,OSError):pass
    write(RUN/'progress'/f"{j['id']}.json",dict(completed=done,total=total,detail=detail,utc=utc(),runtime=runtime))
    print(j['id'],done,'/',total,detail,flush=True)
def finish(j,outputs,**kw):write(RUN/'receipts'/f"{j['id']}.json",dict(job_id=j['id'],status='PASS',utc=utc(),outputs={str(p.relative_to(ROOT)):sha(p) for p in outputs},**kw))
