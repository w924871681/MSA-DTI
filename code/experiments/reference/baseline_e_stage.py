"""Run each fresh C-selected baseline on E; wait for global seal before scoring."""
import fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parent;C=next((ROOT/'flow').glob('C*'));E=C/'fresh_E_stage';SUP=ROOT/'fresh_baseline_e_supervisor';LEGACY=ROOT/'fresh_E_legacy'
for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[n]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['PYTHONDONTWRITEBYTECODE']='1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.pending');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def check(gpu,own):
 s=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=20).strip().split(',');uuid,used,total=[x.strip() for x in s]
 raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20)
 other=[int(p) for u,p in ([x.strip() for x in row.split(',')] for row in raw.splitlines()) if u==uuid and p.isdigit() and int(p) not in own]
 mem=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
 if other or int(total)-int(used)<4096 or mem<16*1024**3 or shutil.disk_usage(ROOT).free<2.5*1024**3:raise RuntimeError('GPU conflict or resource guard')
def prepare_lock():
 guard=(E/'baseline_setup.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX)
 try:
  doc={'E_lock_sha256':sha(E/'run_lock.json'),'baseline_source_sha256':sha(C/'scripts/baseline_e.py'),
       'evaluation_source_sha256':sha(C/'scripts/evaluation.py'),'driver_sha256':sha(Path(__file__)),
       'scope':'Fresh baseline backbones; held-out future carriers read only after global MSA seal.'}
  p=E/'baseline_run_lock.json'
  if p.exists():
   existing=read(p)
   if existing!=doc:
    revision=read(ROOT/'baseline_e_driver_revision.json')
    if revision['previous_driver_sha256']!=existing['driver_sha256'] or revision['new_driver_sha256']!=doc['driver_sha256']:
     raise ValueError('baseline E driver revision mismatch')
    if {k:v for k,v in existing.items() if k!='driver_sha256'}!={k:v for k,v in doc.items() if k!='driver_sha256'}:
     raise ValueError('baseline E dependency changed')
  else:save(p,doc)
 finally:fcntl.flock(guard,fcntl.LOCK_UN);guard.close()
def setup_legacy(cases):
 guard=(E/'future_link_setup.guard').open('a+');fcntl.flock(guard,fcntl.LOCK_EX)
 try:
  for c in cases:
   src=E/'inputs/future'/(c['slug']+'.bin');dst=LEGACY/'data/future/evaluation'/c['slug']/'future.bin'
   if not src.exists() or sha(src)!=c['future_hash']:return False
   dst.parent.mkdir(parents=True,exist_ok=True)
   if dst.exists():
    if sha(dst)!=c['future_hash']:raise ValueError('future link differs')
   else:os.link(src,dst)
  return True
 finally:fcntl.flock(guard,fcntl.LOCK_UN);guard.close()
def modules():
 sys.path.insert(0,str(C/'scripts'));import compute as u;import evaluation as ev;import baseline_e as be
 ev.E=E;ev.GLOBAL=E/'outputs/all_actions.json';ev.FINAL=E/'outputs/verified_summary.json';ev.inputs.cache_clear()
 be.E=E;be.OUT=E/'baseline_results';be.LEGACY=LEGACY
 return u,ev,be
def worker(method,mode,slug):
 u,ev,be=modules();u.configure_stage2_runtime(u.DEVICE,'gru-native');c=next(x for x in read(E/'inputs/cases.json') if x['slug']==slug)
 if mode=='select':be.select(method,c)
 elif mode=='score':be.score(method,c)
 else:raise ValueError('invalid mode')
def main(method,gpu):
 if {'pt':1,'medet':2,'scratch':3}.get(method)!=gpu:raise ValueError('invalid assigned GPU')
 out=SUP/method;out.mkdir(parents=True,exist_ok=True);lock=(out/'supervisor.lock').open('a+')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('baseline E supervisor already running')
 started=time.time();active={};queue=[];done=[];reason='';phase='wait_C_and_D'
 def status(state):save(out/'status.json',{'state':state,'phase':phase,'start_unix':started,'updated_unix':time.time(),'pending':len(queue),'active':list(active),'completed':done,'reason':reason,'disk_free_bytes':shutil.disk_usage(ROOT).free})
 def interrupt(signum,frame):raise RuntimeError('supervisor interrupted')
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 def runphase(mode,cases):
  nonlocal phase,queue,done
  phase='E_'+mode;queue=[];done=[]
  for c in cases:
   slug=c['slug'];f=E/'baseline_results'/method/slug/('selected.json' if mode=='select' else 'scores.json');(done if f.exists() else queue).append(slug)
  while queue or active:
   check(gpu,{x['proc'].pid for x in active.values()})
   for tag,x in list(active.items()):
    rc=x['proc'].poll()
    if rc is None:continue
    x['log'].close();f=E/'baseline_results'/method/tag/('selected.json' if mode=='select' else 'scores.json')
    if rc or not f.exists():del active[tag];raise RuntimeError('baseline E '+mode+' failed '+tag)
    done.append(tag);del active[tag]
   while queue and len(active)<12:
    slug=queue.pop(0);(out/'logs').mkdir(exist_ok=True);log=(out/'logs'/(mode+'_'+slug+'.log')).open('ab',buffering=0)
    proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',method,mode,slug],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
    active[slug]={'proc':proc,'log':log}
   status('running');time.sleep(5)
 try:
  while True:
   report=ROOT/'baseline_reports'/f'{method}_complete.json';err=ROOT/'baseline_reports'/f'{method}_error.json';es=read(ROOT/'fresh_e_supervisor/status.json')
   if err.exists():raise RuntimeError('baseline C report failed')
   if es['state']=='paused':raise RuntimeError('MSA E paused: '+es['reason'])
   if report.exists() and (E/'prepare.json').exists():break
   status('waiting');time.sleep(15)
  prepare_lock();cases=read(E/'inputs/cases.json')
  if len(cases)!=120:raise ValueError('E case count differs')
  select_start=time.time();runphase('select',cases);select_end=time.time();stage=read(C/'fresh_baselines'/method/'stage_start.json')
  marker=out/'SELECTION_COMPLETE.json'
  if marker.exists():
   prior=read(marker)
   if prior.get('method')!=method or prior.get('selected_tasks')!=120:raise ValueError('existing selection timing marker differs')
  else:
   save(marker,{'method':method,'selected_tasks':120,'selection_start_unix':select_start,'selection_end_unix':select_end,
    'gross_from_C_start_seconds':select_end-stage['start_unix'],'scope':'Fresh baseline C source/selection through E model instantiation; includes waiting for upstream E preparation.'})
  phase='wait_seal_and_future';queue=[];done=[]
  while True:
   es=read(ROOT/'fresh_e_supervisor/status.json')
   if es['state']=='paused':raise RuntimeError('MSA E paused: '+es['reason'])
   if (E/'outputs/all_actions.json').exists() and setup_legacy(cases):break
   if shutil.disk_usage(ROOT).free<2.5*1024**3:raise RuntimeError('disk guard while waiting')
   status('waiting');time.sleep(15)
  runphase('score',cases);u,ev,be=modules();be.summarize(method,cases)
  target=E/'outputs'/f'baseline_{method}_verified.json';save(out/'E_COMPLETE.json',{'method':method,'summary_sha256':sha(target),'completed_unix':time.time()})
  phase='E_complete';status('completed')
 except Exception as e:reason=str(e);status('paused');print('PAUSED',reason,flush=True)
 finally:
  for x in active.values():
   if x['proc'].poll() is None:x['proc'].terminate()
  for x in active.values():
   try:x['proc'].wait(timeout=15)
   except subprocess.TimeoutExpired:x['proc'].kill();x['proc'].wait()
   x['log'].close()
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(sys.argv[2],sys.argv[3],sys.argv[4])
 else:main(sys.argv[1],int(sys.argv[2]))
