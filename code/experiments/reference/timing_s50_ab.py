"""Run new A reference selection then new B screening, using fresh source states."""
import csv,fcntl,hashlib,json,math,os,shutil,signal,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
FLOW=ROOT/'flow';A0=next(FLOW.glob('A*'));B0=next(FLOW.glob('B*'))
SOURCE=ROOT/'timing_s50_source_20261104';RUN=ROOT/'timing_s50_ab_20261104';A=RUN/'A_fresh';B=RUN/'B_fresh';BCLONE=RUN/'B_engine'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['PYTHONDONTWRITEBYTECODE']='1'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,value):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.pending')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(tmp,p)
def memfree():return int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
def check(own):
 memory=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
 if memory<16*1024**3 or shutil.disk_usage(ROOT).free<2.5*1024**3:raise RuntimeError('host memory or disk resource guard')
def available_gpus(active):
 own={x['process'].pid if 'process' in x else x['proc'].pid for x in active.values()}
 raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20)
 foreign=set()
 for line in raw.splitlines():
  vals=[x.strip() for x in line.split(',')]
  if len(vals)==2 and vals[1].isdigit() and int(vals[1]) not in own:foreign.add(vals[0])
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=20)
 eligible=[]
 for line in raw.splitlines():
  a=[x.strip() for x in line.split(',')]
  if len(a)==4 and a[0].isdigit():
   i=int(a[0]);used=int(a[2]);total=int(a[3]);count=sum(x['gpu']==i for x in active.values())
   if i==0 and a[1] not in foreign and total-used>=4096 and count<12:eligible.append(i)
 return eligible

def prepare():
 srcstatus=read(SOURCE/'status.json')
 if srcstatus['state']!='completed' or len(srcstatus['done'])!=132:raise RuntimeError('Fresh source stage not complete')
 for h in (1,4):
  for i in range(66):
   p=SOURCE/'weights'/f'strong_H{h}_A{i}.pt';rec=read(SOURCE/'receipts'/f'strong_H{h}_A{i}.json')
   if rec['weight_sha256']!=sha(p):raise ValueError('Source checksum mismatch')
 marker=RUN/'prepare.json'
 if marker.exists():
  m=read(marker)
  current=sha(Path(__file__))
  if m['driver_sha256']!=current:
   revision=read(RUN/'driver_revision.json')
   if revision['previous_driver_sha256']!=m['driver_sha256'] or revision['new_driver_sha256']!=current:
    raise ValueError('A/B driver changed without a matching revision receipt')
  return
 if A.exists():raise RuntimeError('Incomplete A preparation requires inspection')
 def ignore(directory,names):
  d=Path(directory);removed={'__pycache__'}
  if d==A0:removed.add('outputs')
  if d==A0/'inputs':removed.add('validation_mse.csv')
  if d==A0/'native_replay'/'inputs':removed.add('checkpoints')
  return [x for x in names if x in removed]
 shutil.copytree(A0,A,ignore=ignore)
 # Modify private copies only; keep provenance and never relabel originals.
 selector=A/'scripts/select_reference.py';selector.write_text(selector.read_text(encoding='utf8').replace('C1_A','S50_A'),encoding='utf8')
 def ignore_b(directory,names):return [x for x in names if x in {'results','outputs','logs','__pycache__'}]
 shutil.copytree(B0,BCLONE,ignore=ignore_b,copy_function=os.link)
 for rel in ('engine/run_screen.py','selection/screen.py'):
  q=BCLONE/rel;text=q.read_text(encoding='utf8').replace('C1_A','S50_A');tmp=q.with_suffix('.pending');tmp.write_text(text,encoding='utf8');os.replace(tmp,q)
 inputs=A/'native_replay'/'inputs';hashfile=inputs/'sha256.json';hashes=read(hashfile)
 for h in (1,4):
  for i in range(66):
   name=f'checkpoints/H{h}/A{i}.pt';target=inputs/name;target.parent.mkdir(parents=True,exist_ok=True)
   shutil.copyfile(SOURCE/'weights'/f'strong_H{h}_A{i}.pt',target)
   digest=sha(target);assert digest==read(SOURCE/'receipts'/f'strong_H{h}_A{i}.json')['weight_sha256'];hashes[name]=digest
 save(hashfile,hashes)
 save(marker,{'driver_sha256':sha(Path(__file__)),'source_status_sha256':sha(SOURCE/'status.json'),
   'source_states':132,'note':'Real source strong-50 recipe; names S50_A; existing A/B VM splits; old comparison only for diagnostics.'})

def verify_a(path,i):
 o=read(path)
 if o['total_comparisons']!=80 or o['architectures_per_case']!=1 or len(o['results'])!=80:raise ValueError('Incomplete A architecture')
 for r in o['results']:
  if r['architecture']!=i or not math.isfinite(r['validation_mse']) or r['validation_mse']<0:raise ValueError('Invalid A score')
 return o

def select_a():
 template=A0/'inputs/validation_mse.csv';profiles={}
 with template.open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   key=int(r['arch_id']),int(r['horizon']);v=(int(r['params']),int(r['operations']))
   if key in profiles and profiles[key]!=v:raise ValueError('Resource profiles conflict')
   profiles[key]=v
 rows=[]
 for i in range(66):
  for r in verify_a(RUN/'a_jobs'/f'A{i}.json',i)['results']:
   h=r['H'];par,ops=profiles[i,h];weight=SOURCE/'weights'/f'strong_H{h}_A{i}.pt'
   rows.append({'vm':r['VM'],'horizon':h,'support':r['K'],'arch_id':i,'validation_mse':r['validation_mse'],
    'params':par,'operations':ops,'n_validation':r['validation_windows'],'source_sha256':sha(weight),
    'adapted_tensor_sha256':r['adapted_tensor_sha256']})
 if len(rows)!=5280:raise ValueError('A table incomplete')
 p=A/'inputs/validation_mse.csv';tmp=p.with_suffix('.pending')
 with tmp.open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 os.replace(tmp,p)
 subprocess.run([sys.executable,'-B',str(A/'scripts/select_reference.py'),'--output',str(A/'outputs')],check=True)
 print('A_SELECTED',read(A/'outputs/stage_a_reference.json')['source_initialization_token'],flush=True)

def supervisor():
 RUN.mkdir(parents=True,exist_ok=True);lock=(RUN/'supervisor.lock').open('a+')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('A/B supervisor already active')
 def interrupt(signum,frame):raise RuntimeError('supervisor interrupted')
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 active={};phase='prepare';queue=[];completed=[];reason='';started=time.time()
 def publish(state):
  save(RUN/'status.json',{'state':state,'phase':phase,'pending':len(queue),'active':list(active),
   'completed':completed,'reason':reason,'updated_unix':time.time(),'start_unix':started,
   'disk_free_bytes':shutil.disk_usage(ROOT).free,'workers_limit':12})
 def stopchildren():
  for item in active.values():
   if item['process'].poll() is None:item['process'].terminate()
  for item in active.values():
   try:item['process'].wait(timeout=15)
   except subprocess.TimeoutExpired:item['process'].kill();item['process'].wait()
   item['log'].close()
 try:
  publish('preparing');prepare();phase='A';(RUN/'a_jobs').mkdir(exist_ok=True);(RUN/'logs').mkdir(exist_ok=True)
  tasks=read(A/'native_replay/inputs/tasks.json');slugs=[x['slug'] for x in tasks]
  assert len(slugs)==80 and len(set(slugs))==80
  for i in range(66):
   p=RUN/'a_jobs'/f'A{i}.json'
   if p.exists():verify_a(p,i);completed.append(i)
   else:queue.append(i)
  while queue or active:
   check({x['process'].pid for x in active.values()})
   for tag,item in list(active.items()):
    rc=item['process'].poll()
    if rc is None:continue
    item['log'].close();p=RUN/'a_jobs'/f'A{tag}.json'
    if not p.exists():raise RuntimeError(f'A{tag} failed without result; exit={rc}')
    result=verify_a(p,tag)
    if rc not in (0,1) or (rc==1 and result['status']!='MISMATCH'):raise RuntimeError(f'A{tag} failed; exit={rc}')
    completed.append(tag);del active[tag]
   while queue and len(active)<12:
    options=available_gpus(active)
    if not options:break
    i=queue.pop(0);path=RUN/'a_jobs'/f'A{i}.json';log=(RUN/'logs'/f'A{i}.log').open('ab',buffering=0)
    gpu=min(options,key=lambda g:sum(x['gpu']==g for x in active.values()));env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
    cmd=[sys.executable,'-B','-u',str(A/'native_replay/replay_gpu.py'),'--case',*slugs,'--arch',str(i),'--output',str(path)]
    proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);active[i]={'process':proc,'log':log,'gpu':gpu}
   publish('running');time.sleep(5)
  select_a();save(RUN/'A_COMPLETE.json',{'reference_sha256':sha(A/'outputs/stage_a_reference.json'),'completed_unix':time.time()})
  phase='B';completed=[];queue=list(range(12));B.mkdir(exist_ok=True)
  while queue or active:
   check({x['process'].pid for x in active.values()})
   for tag,item in list(active.items()):
    rc=item['process'].poll()
    if rc is None:continue
    item['log'].close()
    if rc!=0:raise RuntimeError(f'B worker {tag} failed; exit={rc}')
    completed.append(tag);del active[tag]
   while queue and len(active)<12:
    options=available_gpus(active)
    if not options:break
    i=queue.pop(0);log=(RUN/'logs'/f'B_worker{i}.log').open('ab',buffering=0);gpu=min(options,key=lambda g:sum(x['gpu']==g for x in active.values()));env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
    cmd=[sys.executable,'-B','-u',str(BCLONE/'engine/run_screen.py'),'--a-root',str(A),'--output',str(B/'results'),
         '--worker-index',str(i),'--worker-count','12']
    proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);active[i]={'process':proc,'log':log,'gpu':gpu}
   publish('running');time.sleep(5)
  subprocess.run([sys.executable,'-B',str(BCLONE/'selection/screen.py'),'--a-root',str(A),'--results',str(B/'results'),
                  '--output',str(B/'outputs')],check=True)
  save(RUN/'B_COMPLETE.json',{'bank_sha256':sha(B/'outputs/retained_bank.json'),'completed_unix':time.time()});phase='B_complete';publish('completed')
 except Exception as e:
  reason=str(e);publish('paused');print('PAUSED',reason,flush=True)
 finally:
  stopchildren();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':supervisor()
