"""Azure MSA-only transfer of frozen 19 strong-50 candidates, after timing."""
import fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time,types,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;C=ROOT/'diagnostic_strong50_flow/C_real'
FIXTURE=os.environ.get('MSADTI_S50_AZURE_FIXTURE')=='1'
AZ=ROOT/('azure_s50_19_fixture_20261104' if FIXTURE else 'azure_s50_19_20261104');E=AZ/'E';SUP=AZ/'supervisor'
for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[n]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['PYTHONDONTWRITEBYTECODE']='1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.pending');q.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf8');os.replace(q,p)
def evaluation():
 path=C/'scripts/evaluation.py';text=path.read_text(encoding='utf8');old='physical_adaptations=120*55';new='physical_adaptations=sum(len(ids) for case in cases for _,ids in budgets(case["H"]))'
 if text.count(old)!=1:raise ValueError('Unexpected evaluator')
 text=text.replace(old,new);sys.path.insert(0,str(C/'scripts'));m=types.ModuleType('s50_azure_eval');m.__file__=str(path);exec(compile(text,str(path),'exec'),m.__dict__)
 m.E=E;m.GLOBAL=E/'outputs/all_actions.json';m.FINAL=E/'outputs/verified_summary.json';m.inputs.cache_clear();return m
def install():
 if E.exists():raise RuntimeError('Azure E already exists; inspect before restart')
 manifest=read(AZ/'selection_manifest.json');payload=AZ/'azure_s50_19_selection.zip'
 if manifest['schema']!='AZURE_S50_19_MSA_ONLY_V1' or sha(payload)!=read(AZ/'upload_receipt.json')['payload_sha256']:raise ValueError('Upload identity mismatch')
 E.mkdir(parents=True)
 with zipfile.ZipFile(payload) as z:
  if set(z.namelist())!=set(manifest['files']):raise ValueError('Unexpected archive membership')
  for name in sorted(z.namelist()):
   p=Path(name)
   if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive path')
   data=z.read(name)
   if hashlib.sha256(data).hexdigest()!=manifest['files'][name]:raise ValueError('Input hash mismatch '+name)
   dest=E/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
 cases=read(E/'inputs/cases.json');policy=read(E/'inputs/frozen_policy.json');proto=read(E/'protocol.json');bank=read(E/'inputs/bank.json')
 if len(cases)!=120 or len({c['VM'] for c in cases})!=30 or len(proto['bank'])!=19 or policy['reference']!=57 or policy['tau']!=.2:raise ValueError('Unexpected dataset or frozen policy')
 for h in ('1','4'):
  for entry in bank[h].values():
   if sha(E/'inputs'/entry['path'])!=entry['sha256']:raise ValueError('Weight changed')
 (E/'inputs/future').mkdir();(E/'outputs').mkdir()
 save(E/'run_lock.json',{'manifest_sha256':sha(AZ/'selection_manifest.json'),'payload_sha256':sha(payload),'driver_sha256':sha(Path(__file__)),'evaluator_sha256':sha(C/'scripts/evaluation.py'),'compute_sha256':sha(C/'scripts/compute.py'),'scope':'same previously observed Azure tasks; frozen S50 19 bank; future unavailable until selection seal'})
def worker(mode,slug):
 lock=read(E/'run_lock.json')
 if sha(Path(__file__))!=lock['driver_sha256'] or sha(C/'scripts/evaluation.py')!=lock['evaluator_sha256'] or sha(C/'scripts/compute.py')!=lock['compute_sha256']:raise ValueError('Dependency changed')
 sys.path.insert(0,str(C/'scripts'));import compute as u;u.configure_stage2_runtime(u.DEVICE,'gru-native')
 case=next(x for x in read(E/'inputs/cases.json') if x['slug']==slug);ev=evaluation()
 if mode=='select':ev.select(case,lambda *_:None)
 elif mode=='score':ev.score(case,lambda *_:None)
 else:raise ValueError('Invalid mode')
def main():
 from dynamic_gpu_controller import Controller
 SUP.mkdir(parents=True,exist_ok=True);lk=(SUP/'supervisor.lock').open('a+')
 try:fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('Azure supervisor active')
 controller=Controller(SUP/'concurrency_events.jsonl');active={};queue=[];done=[];phase='prepare';reason='';started=time.time()
 def status(state):save(SUP/'status.json',{'state':state,'phase':phase,'reason':reason,'started_unix':started,'updated_unix':time.time(),'pending':len(queue),'done':len(done),'active':{k:{'pid':x['proc'].pid,'gpu':x['gpu']} for k,x in active.items()},'disk_free_bytes':shutil.disk_usage(ROOT).free})
 def signal_handler(a,b):raise RuntimeError('supervisor interrupted')
 signal.signal(signal.SIGTERM,signal_handler);signal.signal(signal.SIGINT,signal_handler)
 def runphase(mode,cases):
  nonlocal queue,done,phase
  phase=mode;done=[];queue=[c['slug'] for c in cases]
  while queue or active:
   controller.tick(active,len(done),bool(queue))
   if shutil.disk_usage(ROOT).free<2.5*1024**3:raise RuntimeError('disk guard')
   for tag,x in list(active.items()):
    rc=x['proc'].poll()
    if rc is None:continue
    x['log'].close();path=E/'results'/tag/('selected.json' if mode=='select' else 'scores.json')
    if rc or not path.exists():del active[tag];raise RuntimeError(mode+' worker failed '+tag)
    done.append(tag);del active[tag]
   while queue and len(active)<16:
    options=controller.available(active)
    if not options:break
    gpu=min(options,key=lambda g:sum(x['gpu']==g for x in active.values()));tag=queue.pop(0);(SUP/'logs').mkdir(exist_ok=True)
    log=(SUP/'logs'/(mode+'_'+tag+'.log')).open('ab',buffering=0);p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)), '--worker',mode,tag],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)))
    active[tag]={'proc':p,'log':log,'gpu':gpu}
   status('running' if active else 'waiting');time.sleep(5)
 try:
  install();cases=read(E/'inputs/cases.json');ev=evaluation();runphase('select',cases);ev.seal(cases)
  save(SUP/'SELECTION_COMPLETE.json',{'seal_sha256':sha(E/'outputs/all_actions.json'),'completed_unix':time.time()});phase='waiting_future'
  while True:
   missing=[c['slug'] for c in cases if not (E/'inputs/future'/(c['slug']+'.bin')).is_file()]
   if not missing:break
   if shutil.disk_usage(ROOT).free<2.5*1024**3:raise RuntimeError('disk guard')
   queue=missing;status('waiting_future');time.sleep(15)
  for c in cases:
   if sha(E/'inputs/future'/(c['slug']+'.bin'))!=c['future_hash']:raise ValueError('Future input mismatch')
  runphase('score',cases);ev.summarize(cases)
  save(SUP/'COMPLETE.json',{'summary_sha256':sha(E/'outputs/verified_summary.json'),'seal_sha256':sha(E/'outputs/all_actions.json'),'completed_unix':time.time()});phase='complete';status('completed')
 except Exception as exc:reason=str(exc);status('paused');print('PAUSED',reason,flush=True)
 finally:
  for x in active.values():
   if x['proc'].poll() is None:x['proc'].terminate()
  for x in active.values():
   try:x['proc'].wait(timeout=15)
   except subprocess.TimeoutExpired:x['proc'].kill();x['proc'].wait()
   x['log'].close()
  fcntl.flock(lk,fcntl.LOCK_UN);lk.close()
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(sys.argv[2],sys.argv[3])
 elif len(sys.argv)>1 and sys.argv[1]=='--fixture-one':
  if not FIXTURE:raise RuntimeError('fixture flag required')
  install();case=read(E/'inputs/cases.json')[0];worker('select',case['slug'])
  result=E/'results'/case['slug']/'selected.json'
  if not result.is_file() or any((E/'inputs/future').iterdir()):raise ValueError('Fixture failed or future data leaked')
  save(AZ/'FIXTURE_COMPLETE.json',{'selected_sha256':sha(result),'slug':case['slug'],'future_uploaded':False});print('FIXTURE_PASS',flush=True)
 else:main()
