"""Fresh Bitbrains C1 source training on one assigned GPU; not end-to-end timing."""
import hashlib,json,os,platform,subprocess,sys,time,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ORIGINAL=ROOT/'original'
OUTPUT=Path(os.environ.get('MSA_SOURCE_OUTPUT',str(ROOT/'fresh_source_stage'))).resolve()
if not OUTPUT.is_relative_to(ROOT) or OUTPUT==ROOT:raise ValueError('Output must be a subdirectory of the bundle')
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE']='1'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.pending')
 tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(tmp,p)

def worker(index,horizon):
 os.environ['MSADTI_OUTPUT']=str(ORIGINAL/'fresh_source_run')
 sys.path.insert(0,str(ORIGINAL/'src'))
 import numpy as np
 import torch
 from msa_dti import source_training as src
 device=src.DEVICE
 with np.load(ORIGINAL/'data/bitbrains/data/bitbrains_dataset.npz',allow_pickle=False) as data:
  x=torch.from_numpy(data[f'X_c1_H{horizon}'].copy());y=torch.from_numpy(data[f'y_c1_H{horizon}'].copy())
 seed=2904+10000*horizon+index;t=time.monotonic()
 model=src.build(index,horizon,s=seed);steps=0
 with src.candidate_backend_context(src.ARCH[index],device,'gru-native'):
  opt=torch.optim.Adam(model.parameters(),lr=.001);g=torch.Generator().manual_seed(seed+100000)
  batches=[]
  for epoch in range(10):
   order=torch.randperm(len(x),generator=g);batches.extend([order[i:i+64] for i in range(0,len(x),64)])
  for ids in batches:src.step(model,opt,x[ids],y[ids],clip=5);steps+=1
  opt=torch.optim.Adam(model.parameters(),lr=.001);g=torch.Generator().manual_seed(seed+200000)
  for it in range(180):
   ids=torch.randint(len(x),((10,20)[it%2],),generator=g);src.step(model,opt,x[ids],y[ids],huber=True,clip=5);steps+=1
 torch.cuda.synchronize(device)
 path=OUTPUT/'weights'/f'H{horizon}'/f'A{index}.pt'
 if path.exists():raise FileExistsError('Fresh output already exists: '+str(path))
 src.save_torch(path,{'state_dict':src.cpu_state(model),'source_only':True,'steps':220,'seed':seed})
 record={'arch':index,'H':horizon,'seed':seed,'actual_updates':steps,'old_recipe_steps_field':220,
  'weight_sha256':sha(path),'seconds':time.monotonic()-t,'torch':str(torch.__version__),'numpy':np.__version__,
  'python':platform.python_version(),'gpu':torch.cuda.get_device_name(device),
  'source_input_sha256':sha(ORIGINAL/'data/bitbrains/data/bitbrains_dataset.npz'),
  'source_code_sha256':sha(ORIGINAL/'src/msa_dti/source_training.py')}
 save(OUTPUT/'receipts'/f'H{horizon}_A{index}.json',record)
 print('COMPLETE',horizon,index,record['seconds'],flush=True)

def gpu():
 raw=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,memory.total',
                              '--format=csv,noheader,nounits'],text=True,timeout=20)
 return [int(x.strip()) for x in raw.strip().split(',')]
def foreign_gpu_processes(own):
 uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=20).strip()
 raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20)
 pairs=([x.strip() for x in row.split(',')] for row in raw.splitlines())
 return [int(pid) for u,pid in pairs if u==uuid and pid.isdigit() and int(pid) not in own]
def memfree():
 return int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024

def supervisor():
 import fcntl,signal
 OUTPUT.mkdir(parents=True,exist_ok=True)
 lock=(OUTPUT/'supervisor.lock').open('a+')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('A source supervisor is already active')
 status=OUTPUT/'status.json'; start=OUTPUT/'stage_start.json'
 provenance={'source_code_sha256':sha(ORIGINAL/'src/msa_dti/source_training.py'),
  'source_input_sha256':sha(ORIGINAL/'data/bitbrains/data/bitbrains_dataset.npz'),
  'source_stage_code_sha256':sha(Path(__file__))}
 if start.exists():
  prior=json.loads(start.read_text());assert all(prior[k]==v for k,v in provenance.items()),'Source inputs or code changed'
  started=prior['started_unix']
 else:
  started=time.time();save(start,{'status':'RUNNING_SOURCE_STAGE_ONLY','started_unix':started,'workers':12,'physical_gpu':0,
   'scope':'Fresh 132 source weights. Downstream end-to-end integration is not yet complete.',**provenance})
 queue=[];active={};done=[];failed=[]
 for h in (1,4):
  for i in range(66):
   tag=f'H{h}_A{i}';receipt=OUTPUT/'receipts'/f'{tag}.json';weight=OUTPUT/'weights'/f'H{h}'/f'A{i}.pt'
   if receipt.exists() and weight.exists():
    r=json.loads(receipt.read_text());assert r['weight_sha256']==sha(weight) and r['arch']==i and r['H']==h
    done.append({'tag':tag,'H':h,'arch':i})
   elif receipt.exists() or weight.exists():raise RuntimeError('Incomplete source output requires review: '+tag)
   else:queue.append((h,i))
 def publish(state,reason=''):
  save(status,{'state':state,'reason':reason,'start_unix':started,'updated_unix':time.time(),'pending':len(queue),
   'active':list(active),'completed':done,'failed':failed,'disk_free_bytes':shutil.disk_usage(ROOT).free,'workers_limit':12})
 publish('starting')
 def interrupted(signum,frame):raise RuntimeError('supervisor interrupted')
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  while queue or active:
   used,total=gpu()
   if foreign_gpu_processes({item['process'].pid for item in active.values()}):raise RuntimeError('another process is using assigned GPU')
   if shutil.disk_usage(ROOT).free<2.5*1024**3 or total-used<4096 or memfree()<16*1024**3:
    publish('pausing','resource guard');raise RuntimeError('resource guard')
   for tag,item in list(active.items()):
    rc=item['process'].poll()
    if rc is None:continue
    item['log'].close()
    h,i=item['H'],item['arch'];receipt=OUTPUT/'receipts'/f'H{h}_A{i}.json';weight=OUTPUT/'weights'/f'H{h}'/f'A{i}.pt'
    if rc==0 and receipt.exists() and weight.exists() and json.loads(receipt.read_text())['weight_sha256']==sha(weight):
     done.append({'tag':tag,'H':h,'arch':i})
    else:
     failed.append({'tag':tag,'exit_code':rc});del active[tag];publish('failed','source worker failed');raise RuntimeError('source worker failed '+tag)
    del active[tag]
   while queue and len(active)<12:
    h,i=queue.pop(0);tag=f'H{h}_A{i}';logdir=OUTPUT/'logs';logdir.mkdir(exist_ok=True)
    log=(logdir/(tag+'.log')).open('ab',buffering=0);env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
    proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(i),str(h)],
                           env=env,stdout=log,stderr=subprocess.STDOUT)
    active[tag]={'process':proc,'log':log,'H':h,'arch':i}
   publish('running');time.sleep(5)
  publish('completed')
 except Exception as e:
  publish('paused',str(e));print('PAUSED',str(e),flush=True)
 finally:
  for item in active.values():
   if item['process'].poll() is None:item['process'].terminate()
  for item in active.values():
   try:item['process'].wait(timeout=15)
   except subprocess.TimeoutExpired:item['process'].kill();item['process'].wait()
   item['log'].close()
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(int(sys.argv[2]),int(sys.argv[3]))
 else:supervisor()
