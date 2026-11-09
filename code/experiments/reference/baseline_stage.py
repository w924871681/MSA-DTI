"""Run fresh C-group backbone selection for one baseline on its assigned GPU.

This is one phase of a future full method run, not an end-to-end timing result.
"""
import fcntl,hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CROOT=next((ROOT/'flow').glob('C*'))
OUTROOT=CROOT/'fresh_baselines'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['PYTHONDONTWRITEBYTECODE']='1'

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.pending')
 tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8');os.replace(tmp,path)
def identity(method):
 obj={'method':method,'initialization_policy':'fresh_source_no_reused_weights','run_lock_sha256':sha(CROOT/'run_lock.json'),
      'baseline_stage_sha256':sha(Path(__file__)),'recipe':'original C-group compute.work'}
 return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def worker(method,i,h):
 spec=importlib.util.spec_from_file_location('frozen_compute',CROOT/'scripts/compute.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 for rel,digest in module.read(CROOT/'run_lock.json')['files'].items():
  if module.sha(CROOT/rel)!=digest:raise ValueError('Frozen dependency changed: '+rel)
 for rel,digest in module.read(CROOT/'inputs/files.json').items():
  if module.sha(CROOT/rel)!=digest:raise ValueError('Frozen input changed: '+rel)
 module.OUT=OUTROOT/method
 module.configure_stage2_runtime(module.DEVICE,'gru-native')
 print(json.dumps({'method':method,'arch':i,'H':h,'fresh_initialization':True,
  'torch':str(module.torch.__version__),'numpy':module.np.__version__,'gpu':module.torch.cuda.get_device_name()}),flush=True)
 module.work(method,i,h,identity(method),module.read(CROOT/'inputs/cases.json'),{})
def gpu(idx):
 out=subprocess.check_output(['nvidia-smi','-i',str(idx),'--query-gpu=memory.used,memory.total',
  '--format=csv,noheader,nounits'],text=True,timeout=20)
 return [int(x.strip()) for x in out.strip().split(',')]
def other_gpu(idx,own):
 uuid=subprocess.check_output(['nvidia-smi','-i',str(idx),'--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=20).strip()
 raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20)
 return [int(pid) for u,pid in ([x.strip() for x in line.split(',')] for line in raw.splitlines()) if u==uuid and pid.isdigit() and int(pid) not in own]
def memfree():return int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
def supervisor(method,device):
 allowed={'pt':1,'medet':2,'scratch':3}
 if allowed.get(method)!=device:raise ValueError('Invalid assigned GPU')
 out=OUTROOT/method;out.mkdir(parents=True,exist_ok=True)
 lock=(out/'supervisor.lock').open('a+')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('Supervisor already active')
 runid=identity(method);start=out/'stage_start.json'
 if start.exists():
  info=read(start)
  if info['identity']!=runid:raise RuntimeError('Baseline run identity changed')
  started=info['start_unix']
 else:
  started=time.time();save(start,{'method':method,'gpu':device,'workers':12,'start_unix':started,'identity':runid,
   'scope':'Fresh source initialization where applicable and C-group selection only; E and full end-to-end timing pending.',
   'run_lock_sha256':sha(CROOT/'run_lock.json'),'driver_sha256':sha(Path(__file__))})
 queue=[];active={};done=[];failed=[]
 for i in range(66):
  for h in (1,4):
   tag=f'{method}_A{i}_H{h}';folder=out/'jobs'/tag;receipt=folder/'scores.json';pred=folder/'predictions.npz'
   if receipt.exists() and pred.exists():
    row=read(receipt)
    if row['signature']!=runid or row['predictions_sha256']!=sha(pred):raise ValueError('Invalid prior result '+tag)
    done.append({'tag':tag,'arch':i,'H':h})
   elif receipt.exists() or pred.exists():raise RuntimeError('Incomplete output requires review '+tag)
   else:queue.append((i,h))
 def publish(state,reason=''):
  save(out/'status.json',{'state':state,'reason':reason,'method':method,'gpu':device,'workers_limit':12,
   'start_unix':started,'updated_unix':time.time(),'pending':len(queue),'active':list(active),'completed':done,
   'failed':failed,'disk_free_bytes':shutil.disk_usage(ROOT).free})
 def interrupted(signum,frame):raise RuntimeError('supervisor interrupted')
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted);publish('starting')
 try:
  while queue or active:
   used,total=gpu(device)
   if other_gpu(device,{x['process'].pid for x in active.values()}):raise RuntimeError('another process is using assigned GPU')
   if shutil.disk_usage(ROOT).free<2.5*1024**3 or total-used<4096 or memfree()<16*1024**3:raise RuntimeError('resource guard')
   for tag,item in list(active.items()):
    rc=item['process'].poll()
    if rc is None:continue
    item['log'].close();folder=out/'jobs'/tag;receipt=folder/'scores.json';pred=folder/'predictions.npz'
    if rc==0 and receipt.exists() and pred.exists() and read(receipt)['predictions_sha256']==sha(pred):
     done.append({'tag':tag,'arch':item['arch'],'H':item['H']})
    else:
     failed.append({'tag':tag,'exit_code':rc});del active[tag];raise RuntimeError('worker failed '+tag)
    del active[tag]
   while queue and len(active)<12:
    i,h=queue.pop(0);tag=f'{method}_A{i}_H{h}';(out/'logs').mkdir(exist_ok=True)
    log=(out/'logs'/(tag+'.log')).open('ab',buffering=0);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(device))
    proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',method,str(i),str(h)],
     stdout=log,stderr=subprocess.STDOUT,env=env)
    active[tag]={'process':proc,'log':log,'arch':i,'H':h}
   publish('running');time.sleep(5)
  publish('completed')
 except Exception as e:publish('paused',str(e));print('PAUSED',str(e),flush=True)
 finally:
  for x in active.values():
   if x['process'].poll() is None:x['process'].terminate()
  for x in active.values():
   try:x['process'].wait(timeout=15)
   except subprocess.TimeoutExpired:x['process'].kill();x['process'].wait()
   x['log'].close()
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]))
 else:supervisor(sys.argv[1],int(sys.argv[2]))
