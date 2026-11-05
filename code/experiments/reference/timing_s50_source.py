"""Real Bitbrains source: original per-center 50-epoch MSE recipe.

Run from the existing server bundle.  The 132 strong checkpoints and two
separate A57-PT checkpoints are stored in a new folder and never overwrite C1.
"""
import fcntl, hashlib, json, os, random, shutil, signal, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ORIGINAL=ROOT/'original'
OUT=ROOT/'timing_s50_source_20261104'
DATA=ORIGINAL/'data/bitbrains/data/bitbrains_dataset.npz'
MANIFEST=ORIGINAL/'data/bitbrains/data/bitbrains_dataset_manifest.json'
MIN_FREE=int(2.5*1024**3)
EXPECTED_INPUT='62f7b942fc584b2fc0e3894367cde6dfe1303d0f1a38ba3f70cc6de76196880c'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['PYTHONDONTWRITEBYTECODE']='1'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.pending')
 tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');os.replace(tmp,p)
def tag(mode,i,h):return f'{mode}_H{h}_A{i}'
def paths(mode,i,h):
 stem=tag(mode,i,h);return OUT/'weights'/f'{stem}.pt',OUT/'receipts'/f'{stem}.json'

def worker(mode,i,h):
 if mode not in ('strong','pt') or (mode=='pt' and i!=57) or h not in (1,4) or i not in range(66):raise ValueError('bad job')
 if sha(DATA)!=EXPECTED_INPUT:raise ValueError('source dataset changed')
 os.environ['MSADTI_OUTPUT']=str(ORIGINAL/'fresh_source_run');sys.path.insert(0,str(ORIGINAL/'src'))
 import numpy as np
 import torch
 from msa_dti import source_training as src
 ids=read(MANIFEST)['source_machine_ids'];assert len(ids)==20 and len(set(ids))==20
 with np.load(DATA,allow_pickle=False) as d:
  source=[(torch.from_numpy(d[f'X_source_{j}_H{h}'].copy()),torch.from_numpy(d[f'y_source_{j}_H{h}'].copy())) for j in range(20)]
 if any(x.shape!=(100,96,25) or y.shape!=(100,h) for x,y in source):raise ValueError('source array shape mismatch')
 seed=2904+101*h+i;start=time.time();model=src.build(i,h,s=seed);steps=visits=0
 with src.candidate_backend_context(src.ARCH[i],src.DEVICE,'gru-native'):
  opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.,betas=(.9,.999),eps=1e-8,amsgrad=False)
  for epoch in range(50):
   if shutil.disk_usage(ROOT).free<MIN_FREE:raise RuntimeError('disk guard')
   order=list(range(20));random.Random(2904+10007*h+epoch if mode=='pt' else seed+epoch).shuffle(order)
   for j in order:
    x,y=source[j];machine_seed=int(hashlib.sha256(ids[j].encode()).hexdigest()[:8],16)
    s=2904+101*h+1009*j+epoch if mode=='pt' else seed+epoch+machine_seed
    g=torch.Generator(device='cpu').manual_seed(s);order100=torch.randperm(100,generator=g)
    for k in range(0,100,64):
     batch=order100[k:k+64];src.step(model,opt,x[batch],y[batch]);steps+=1;visits+=len(batch)
   if (epoch+1)%10==0:print(tag(mode,i,h),'epoch',epoch+1,'updates',steps,flush=True)
 torch.cuda.synchronize(src.DEVICE)
 if (steps,visits)!=(2000,100000):raise ValueError('unexpected training counts')
 weight,receipt=paths(mode,i,h)
 if weight.exists() or receipt.exists():raise FileExistsError('existing output needs inspection')
 src.save_torch(weight,{'state_dict':src.cpu_state(model),'source_only':True,'recipe':'per-center 50 epoch Adam MSE','seed':seed})
 save(receipt,{'status':'COMPLETE','mode':mode,'arch':i,'H':h,'seed':seed,'epochs':50,'updates':steps,'sample_visits':visits,
  'source_vm_ids':ids,'source_input_sha256':sha(DATA),'source_manifest_sha256':sha(MANIFEST),'weight_sha256':sha(weight),
  'driver_sha256':sha(Path(__file__)),'source_code_sha256':sha(ORIGINAL/'src/msa_dti/source_training.py'),
  'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(0),'seconds':time.time()-start})

def supervisor():
 OUT.mkdir(parents=True,exist_ok=True);lk=(OUT/'supervisor.lock').open('a+')
 try:fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('source supervisor already running')
 jobs=[('strong',i,h) for h in (1,4) for i in range(66)]
 priority={(m,i,h):n for n,(m,i,h) in enumerate([('strong',55,1),('strong',56,1),('strong',55,4),('strong',56,4),('pt',57,1),('pt',57,4)])}
 jobs.sort(key=lambda j:(priority.get(j,100),j[2],j[1]))
 done=[];queue=[];active={};started=time.time()
 for mode,i,h in jobs:
  weight,rec=paths(mode,i,h)
  if weight.exists() and rec.exists():
   r=read(rec);assert r['weight_sha256']==sha(weight) and r['driver_sha256']==sha(Path(__file__));done.append(tag(mode,i,h))
  elif weight.exists() or rec.exists():raise RuntimeError('incomplete output '+tag(mode,i,h))
  else:queue.append((mode,i,h))
 def status(state,reason=''):
  save(OUT/'status.json',{'state':state,'reason':reason,'started_unix':started,'updated_unix':time.time(),'total':len(jobs),
   'done':done,'pending':len(queue),'active':{k:{'pid':v['p'].pid,'gpu':v['gpu']} for k,v in active.items()},
   'workers_per_gpu':12,'disk_free_bytes':shutil.disk_usage(ROOT).free,'recipe':'real source, 50 epochs, Adam MSE'})
 def interrupted(sig,frame):raise RuntimeError('supervisor interrupted')
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  if sha(DATA)!=EXPECTED_INPUT:raise ValueError('source dataset changed')
  save(OUT/'run_lock.json',{'source_sha256':sha(DATA),'manifest_sha256':sha(MANIFEST),'script_sha256':sha(Path(__file__)),'jobs':len(jobs),'source_ids':read(MANIFEST)['source_machine_ids']})
  while queue or active:
   if shutil.disk_usage(ROOT).free<MIN_FREE:raise RuntimeError('disk below 2.5 GiB guard')
   mem=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
   for name,item in list(active.items()):
    rc=item['p'].poll()
    if rc is None:continue
    item['log'].close();w,r=paths(*item['job'])
    if rc or not w.is_file() or not r.is_file() or read(r)['weight_sha256']!=sha(w):raise RuntimeError('worker failed '+name+' exit '+str(rc))
    done.append(name);del active[name]
   raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
   gpus={int(q[0]):(q[1],int(q[2]),int(q[3])) for q in ([x.strip() for x in line.split(',')] for line in raw.splitlines())}
   raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
   own={x['p'].pid for x in active.values()};foreign=set()
   for line in raw.splitlines():
    q=[x.strip() for x in line.split(',')]
    if len(q)==2 and q[1].isdigit() and int(q[1]) not in own:foreign.add(q[0])
   for g in range(1):
    uuid,used,total=gpus[g]
    while queue and mem>8*1024**3 and uuid not in foreign and total-used>4096 and sum(x['gpu']==g for x in active.values())<12:
     job=queue.pop(0);name=tag(*job);p=OUT/'logs'/f'{name}.log';p.parent.mkdir(exist_ok=True,parents=True);log=p.open('ab',buffering=0)
     proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker',job[0],str(job[1]),str(job[2])],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g)),stdout=log,stderr=subprocess.STDOUT)
     active[name]={'p':proc,'log':log,'gpu':g,'job':job}
   status('running' if active else 'waiting','resource availability' if not active and queue else '');time.sleep(5)
  status('completed');save(OUT/'SOURCE_COMPLETE.json',{'status':'COMPLETE','jobs':len(done),'lock_sha256':sha(OUT/'run_lock.json'),'completed_unix':time.time()})
 except Exception as e:status('paused',str(e));print('PAUSED',str(e),flush=True)
 finally:
  for item in active.values():
   if item['p'].poll() is None:item['p'].terminate()
  for item in active.values():
   try:item['p'].wait(timeout=15)
   except subprocess.TimeoutExpired:item['p'].kill();item['p'].wait()
   item['log'].close()
  fcntl.flock(lk,fcntl.LOCK_UN);lk.close()
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]))
 else:supervisor()
