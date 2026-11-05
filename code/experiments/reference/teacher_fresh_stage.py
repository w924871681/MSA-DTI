"""Fresh source-only teacher episodes for the Meta+NAS-lite baseline.

Run on the server as `python3 -B teacher_fresh_stage.py`; GPU 0 and 12
processes are intentionally fixed for the Bitbrains run. Each completed
episode is checkpointed separately and identified by all relevant inputs.
"""
import fcntl, hashlib, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORIG = ROOT / 'original'
C = next((ROOT / 'flow').glob('C*'))
OUT = ROOT / 'fresh_search_teacher'
WEIGHTS = ROOT / 'fresh_source_stage' / 'weights'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.pending')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)

def token(cid, h, k): return f's{cid}_h{h}_k{k}'

def identity():
    files = [Path(__file__), ORIG/'src/msa_dti/source_training.py', ORIG/'src/msa_dti/teacher.py',
             ORIG/'src/msa_dti/baseline_methods.py', C/'inputs/source_vms.json',
             ORIG/'data/bitbrains/data/bitbrains_dataset_manifest.json']
    files += [WEIGHTS/f'H{h}'/f'A{i}.pt' for h in (1,4) for i in range(66)]
    files += [C/'inputs/source_tasks'/f'{token(cid,h,k)}.npz' for h in (1,4) for k in (10,20) for cid in range(20)]
    registry=read(C/'inputs/files.json')
    for h in (1,4):
        for i in range(66):
            receipt=read(ROOT/'fresh_source_stage'/'receipts'/f'H{h}_A{i}.json')
            if sha(WEIGHTS/f'H{h}'/f'A{i}.pt')!=receipt['weight_sha256']:raise ValueError('Fresh source weight mismatch')
        for k in (10,20):
            for cid in range(20):
                name=f'inputs/source_tasks/{token(cid,h,k)}.npz'
                if sha(C/name)!=registry[name]:raise ValueError('Source task hash mismatch: '+name)
    srcids = read(ORIG/'data/bitbrains/data/bitbrains_dataset_manifest.json')['source_machine_ids']
    if srcids != read(C/'inputs/source_vms.json'): raise ValueError('Source VM identities differ')
    return {'files': {str(p.relative_to(ROOT)): sha(p) for p in files}, 'workers':12,
            'gpu':0, 'episodes':80, 'architectures_per_episode':24, 'support_steps':50,
            'scope':'Meta+NAS-lite source teacher only; not full method timing'}

def run_episode(cid,h,k,lock_hash):
    os.environ['MSADTI_OUTPUT']=str(ORIG/'fresh_source_run')
    sys.path.insert(0,str(ORIG/'src'))
    import numpy as np
    import torch
    from msa_dti import source_training as src
    from msa_dti import teacher
    with np.load(C/'inputs/source_tasks'/f'{token(cid,h,k)}.npz',allow_pickle=False) as z:
        xs,ys,xq,yq = [torch.from_numpy(z[n].copy()) for n in ('Xs','ys','Xq','yq')]
    s=2904+1009*cid+37*h+53*k
    ids=teacher._balanced_teacher_subset(feasible_indices=list(range(len(src.ARCH))),A=src.ARCH,n_arch=24,seed=s)
    losses=[]; started=time.time()
    for idx in ids:
        obj=torch.load(WEIGHTS/f'H{h}'/f'A{idx}.pt',map_location='cpu',weights_only=True)
        m=src.build(idx,h,obj['state_dict'],s+idx)
        with src.candidate_backend_context(src.ARCH[idx],src.DEVICE,'gru-native'):
            src.update(m,xs,ys,50,s+idx,search=True)
        losses.append(src.metrics(src.pred(m,xq,idx),yq.numpy())['weighted_mse'])
        del m
    torch.cuda.synchronize()
    save(OUT/'episodes'/f'{token(cid,h,k)}.json',dict(center_id=cid,H=h,K=k,candidate_indices=ids,
         validation_mse=losses,lock_sha256=lock_hash,start_unix=started,end_unix=time.time()))

def supervisor():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'supervisor.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock=OUT/'run_lock.json'; proposed=identity()
        if lock.exists():
            if read(lock)!=proposed: raise ValueError('Teacher source identity changed')
        else: save(lock,proposed)
        lock_hash=sha(lock); started=OUT/'stage_start.json'
        if not started.exists():save(started,{'start_unix':time.time(),'lock_sha256':lock_hash})
        queue=[];active={};done=[]; reason=''; phase='teacher_episodes'
        for h in (1,4):
            for k in (10,20):
                for cid in range(20):
                    p=OUT/'episodes'/f'{token(cid,h,k)}.json'
                    if p.exists():
                        d=read(p)
                        if d['lock_sha256']!=lock_hash or len(d['candidate_indices'])!=24 or len(d['validation_mse'])!=24: raise ValueError('Invalid completed episode')
                        done.append(token(cid,h,k))
                    else:queue.append((cid,h,k))
        def status(state):
            save(OUT/'status.json',{'state':state,'phase':phase,'reason':reason,'updated_unix':time.time(),
                 'pending':len(queue),'active':list(active),'completed':done,'disk_free_bytes':shutil.disk_usage(ROOT).free,'workers':12})
        def stop(sig,frame): raise RuntimeError('Supervisor interrupted')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:
            while queue or active:
                if shutil.disk_usage(ROOT).free < 2.5*1024**3: raise RuntimeError('disk guard')
                free=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
                if free<16*1024**3: raise RuntimeError('host memory guard')
                uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
                lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
                own={v['p'].pid for v in active.values()}
                if any(u.strip()==uuid and pid.strip().isdigit() and int(pid.strip()) not in own for u,pid in (line.split(',') for line in lines)):raise RuntimeError('another process using GPU 0')
                for key,entry in list(active.items()):
                    rc=entry['p'].poll()
                    if rc is None:continue
                    entry['log'].close();del active[key]
                    if rc or not (OUT/'episodes'/f'{key}.json').exists():raise RuntimeError('episode failed: '+key)
                    done.append(key)
                while queue and len(active)<12:
                    cid,h,k=queue.pop(0);key=token(cid,h,k);(OUT/'logs').mkdir(exist_ok=True)
                    log=(OUT/'logs'/f'{key}.log').open('ab',buffering=0)
                    p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker',str(cid),str(h),str(k),lock_hash],
                        stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
                    active[key]={'p':p,'log':log}
                status('running');time.sleep(5)
            sys.path.insert(0,str(ORIG/'src'))
            from msa_dti.teacher import _compute_pi50
            eps=[read(OUT/'episodes'/f'{token(cid,h,k)}.json') for h in (1,4) for k in (10,20) for cid in range(20)]
            prior=_compute_pi50(eps,66)
            if min(prior['coverage'])<1:raise ValueError('Teacher did not cover all 66 architectures')
            save(OUT/'pi50.json',dict(**prior,lock_sha256=lock_hash,episodes=80,source_only=True))
            save(OUT/'COMPLETE.json',{'pi50_sha256':sha(OUT/'pi50.json'),'end_unix':time.time(),'episodes':80})
            phase='complete';status('completed')
        except Exception as e:
            reason=str(e);status('paused');print('PAUSED',reason,flush=True)
        finally:
            for v in active.values():
                if v['p'].poll() is None:v['p'].terminate()
            for v in active.values():
                try:v['p'].wait(timeout=15)
                except subprocess.TimeoutExpired:v['p'].kill();v['p'].wait()
                v['log'].close()

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':run_episode(int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),sys.argv[5])
    else:supervisor()
