"""Queue the exact native H-Meta-NAS source training after GPU 1 is idle.

The native 240 outer iterations update a shared meta bank sequentially, so
only one of the 12 available worker slots is active for this dependency.
The resume checkpoint includes Python-independent Torch CPU/CUDA RNG state.
"""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent; ORIG=ROOT/'original'; C=next((ROOT/'flow').glob('C*'))
OUT=ROOT/'fresh_hmeta_source'; GPU=1
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['PYTHONDONTWRITEBYTECODE']='1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.pending')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)

def lock_doc():
    original=read(ORIG/'data/bitbrains/data/bitbrains_dataset_manifest.json')['source_machine_ids']
    if original!=read(C/'inputs/source_vms.json'):raise ValueError('Source VM identities differ')
    registry=read(C/'inputs/files.json')
    files=[Path(__file__),ORIG/'src/msa_dti/meta_training.py',ORIG/'src/configs/methods/h_meta_nas_cfg.py',
           C/'inputs/source_vms.json',ORIG/'data/bitbrains/data/bitbrains_dataset_manifest.json']
    for h in (1,4):
        for k in (10,20):
            for cid in range(20):
                name=f'inputs/source_tasks/s{cid}_h{h}_k{k}.npz';p=C/name
                if sha(p)!=registry[name]:raise ValueError('Source task mismatch')
                files.append(p)
    return {'scope':'Native sequential H-Meta-NAS source training; one active task, pool capacity 12',
            'files':{str(p.relative_to(ROOT)):sha(p) for p in files},'gpu':GPU,'source_episodes':240}

def worker(smoke=False):
    os.environ['MSADTI_OUTPUT']=str(ORIG/'fresh_source_run');sys.path.insert(0,str(ORIG/'src'))
    import numpy as np
    import torch
    from msa_dti import runtime as u
    from msa_dti import meta_training as native
    original_save=native._save_torch; original_load=native._load_torch
    stage=OUT/'smoke' if smoke else OUT
    def get_task(cfg,cache,cid,H,K):
        with np.load(C/'inputs/source_tasks'/f's{cid}_h{H}_k{K}.npz',allow_pickle=False) as z:
            values=[torch.from_numpy(z[n].copy()) for n in ('Xs','ys','Xq','yq')]
        return (*values,None,None,None,None)
    def checked_save(obj,path):
        if shutil.disk_usage(ROOT).free<2.5*1024**3:raise RuntimeError('disk guard at source checkpoint')
        if isinstance(obj,dict):obj['_resume_cpu_rng']=torch.get_rng_state();obj['_resume_cuda_rng']=torch.cuda.get_rng_state()
        original_save(obj,path)
        save(stage/'progress.json',{'iterations':obj.get('completed_iterations',0),'total':obj.get('target_iterations',240),
             'updated_unix':time.time(),'checkpoint_sha256':sha(path),'disk_free_bytes':shutil.disk_usage(ROOT).free})
    def checked_load(path):
        obj=original_load(path)
        if '_resume_cpu_rng' not in obj or '_resume_cuda_rng' not in obj:raise ValueError('Checkpoint lacks RNG state')
        torch.set_rng_state(obj['_resume_cpu_rng']);torch.cuda.set_rng_state(obj['_resume_cuda_rng'])
        return obj
    native.get_support_validation_check=get_task;native._save_torch=checked_save;native._load_torch=checked_load
    target=4 if smoke else 240
    bank=native._build_meta_bank(root=stage,cfg=None,cache=None,A=u.ARCH,input_dim=25,L=96,device=u.DEVICE,artifact=stage/'hmeta.pt',smoke=smoke)
    if not bank.get('complete') or bank.get('completed_iterations')!=target:raise ValueError('Incomplete H-Meta-NAS source bank')
    save(stage/('SMOKE_PASS.json' if smoke else 'SOURCE_COMPLETE.json'),{'sha256':sha(stage/'hmeta.pt'),'completed_unix':time.time(),'iterations':target,'smoke':smoke})

def main():
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'supervisor.lock').open('a+') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock=OUT/'run_lock.json';doc=lock_doc()
        if lock.exists() and read(lock)!=doc:raise ValueError('H-meta source inputs changed')
        if not lock.exists():save(lock,doc)
        def status(state,phase,reason=''):
            progress=read(OUT/'progress.json') if (OUT/'progress.json').exists() else {}
            save(OUT/'status.json',{'state':state,'phase':phase,'reason':reason,'progress':progress,
                 'worker_slots':12,'active_workers':1 if state=='running' else 0,'updated_unix':time.time(),'disk_free_bytes':shutil.disk_usage(ROOT).free})
        try:
            while True:
                if shutil.disk_usage(ROOT).free<2.5*1024**3:raise RuntimeError('disk guard while waiting')
                other=ROOT/'fresh_search_target/zero_nas/status.json'
                if other.exists() and read(other)['state']=='paused':raise RuntimeError('Prior GPU job paused')
                ready=other.exists() and read(other)['state']=='completed'
                raw=subprocess.check_output(['nvidia-smi','-i',str(GPU),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                if ready and not raw:break
                status('waiting','zero_nas_and_gpu');time.sleep(20)
            if (OUT/'SOURCE_COMPLETE.json').exists():
                if sha(OUT/'hmeta.pt')!=read(OUT/'SOURCE_COMPLETE.json')['sha256']:raise ValueError('Completed bank changed')
                status('completed','source_complete');return
            if not (OUT/'smoke'/'SMOKE_PASS.json').exists():
                with (OUT/'smoke_worker.log').open('ab',buffering=0) as log:
                    proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--smoke-worker'],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(GPU)))
                    while proc.poll() is None:status('running','isolated_smoke');time.sleep(10)
                    if proc.returncode:raise RuntimeError('Smoke worker failed; see smoke_worker.log')
                if not (OUT/'smoke'/'SMOKE_PASS.json').exists():raise RuntimeError('Smoke completion missing')
            if not (OUT/'stage_start.json').exists():save(OUT/'stage_start.json',{'start_unix':time.time(),'lock_sha256':sha(lock)})
            with (OUT/'worker.log').open('ab',buffering=0) as log:
                proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(GPU)))
                while proc.poll() is None:status('running','source');time.sleep(10)
                if proc.returncode:raise RuntimeError('Native source worker failed; see worker.log')
            if not (OUT/'SOURCE_COMPLETE.json').exists():raise ValueError('Missing completion marker')
            status('completed','source_complete')
        except Exception as e:status('paused','error',str(e));print('PAUSED',str(e),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker()
    elif len(sys.argv)>1 and sys.argv[1]=='--smoke-worker':worker(smoke=True)
    else:main()
