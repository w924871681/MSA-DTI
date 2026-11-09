"""Fresh Bitbrains E: filter, adapt, seal all choices, then unlock future scoring."""
import csv,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parent;C=next((ROOT/'timing_s50_flow_20261104').glob('C*'));OLD=C/'E_stage';E=C/'fresh_E_stage';SUP=ROOT/'timing_s50_e_supervisor_20261104'
D=C/'fresh_D_stage';DSTATE=ROOT/'timing_s50_d_supervisor_20261104';A=ROOT/'timing_s50_ab_20261104/A_fresh';BANK=ROOT/'timing_s50_ab_20261104/B_fresh/outputs/retained_bank.json'
for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[n]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['PYTHONDONTWRITEBYTECODE']='1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.pending');t.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');os.replace(t,p)
def check(own):
 available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
 if available<16*1024**3 or shutil.disk_usage(ROOT).free<2.5*1024**3:raise RuntimeError('host memory or disk resource guard')
def dynamic_evaluation():
 path=C/'scripts/evaluation.py';src=path.read_text(encoding='utf8');old='physical_adaptations=120*55'
 new='physical_adaptations=sum(len(ids) for case in cases for _,ids in budgets(case["H"]))'
 if src.count(old)!=1:raise ValueError('Unexpected evaluation source')
 src=src.replace(old,new);sys.path.insert(0,str(C/'scripts'));m=types.ModuleType('dynamic_evaluation');m.__file__=str(path)
 exec(compile(src,str(path),'exec'),m.__dict__);m.E=E;m.GLOBAL=E/'outputs/all_actions.json';m.FINAL=E/'outputs/verified_summary.json';m.inputs.cache_clear()
 return m,hashlib.sha256(src.encode()).hexdigest()
def prepare():
 if read(DSTATE/'status.json')['state']!='completed':raise RuntimeError('D calibration incomplete')
 marker=E/'prepare.json'
 if marker.exists():
  m=read(marker)
  if m['driver_sha256']!=sha(Path(__file__)) or m['B_sha256']!=sha(BANK) or m['D_sha256']!=sha(D/'outputs/frozen_policy.json'):raise ValueError('E dependencies changed')
  return
 if E.exists():raise RuntimeError('Partial E preparation requires review')
 E.mkdir();shutil.copytree(OLD/'inputs',E/'inputs',ignore=shutil.ignore_patterns('checkpoints','future','bank.json','frozen_policy.json'))
 (E/'inputs/future').mkdir();bank=read(BANK);ids=bank['retained_including_comparator'];policy=read(D/'outputs/frozen_policy.json');ref=policy['reference']
 profiles={}
 with (A/'inputs/validation_mse.csv').open(encoding='utf8',newline='') as f:
  for row in csv.DictReader(f):profiles[int(row['arch_id']),int(row['horizon'])]=(int(row['params']),int(row['operations']))
 proto=read(OLD/'protocol.json');proto['bank']=ids;proto['reference']=ref;proto['B_receipt']=sha(BANK);proto['C_receipt']=sha(C/'outputs/msa_selected.json');proto['D_policy']=sha(D/'outputs/frozen_policy.json')
 for h in (1,4):
  for cap in proto['budgets'].values():
   if not (profiles[ref,h][0]<=cap[0] and profiles[ref,h][1]<=cap[1]):raise ValueError('Reference is infeasible under E budget')
 save(E/'protocol.json',proto);save(E/'inputs/frozen_policy.json',policy);newbank={}
 reg=read(C/'fresh_msa/dynamic_inputs.json')['registry']
 for h in (1,4):
  newbank[str(h)]={}
  for i in ids:
   ent=reg[f'msa/{h}/{i}'];src=C/ent['path']
   if sha(src)!=ent['sha256']:raise ValueError('Fresh source checkpoint changed')
   rel=f'checkpoints/H{h}/A{i}.pt';dst=E/'inputs'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
   par,ops=profiles[i,h];newbank[str(h)][str(i)]={'path':rel,'sha256':sha(dst),'parameters':par,'operations':ops}
 save(E/'inputs/bank.json',newbank);ev,derived=dynamic_evaluation();files={}
 for p in [E/'protocol.json',BANK,D/'outputs/frozen_policy.json',C/'outputs/msa_selected.json',C/'scripts/evaluation.py',C/'scripts/compute.py',*list((E/'inputs').rglob('*'))]:
  if p.is_file():files[str(p.relative_to(C) if p.is_relative_to(C) else p.relative_to(ROOT))]=sha(p)
 save(E/'run_lock.json',{'files':files,'derived_evaluation_sha256':derived,'scope':'Fresh bank and calibrated D policy; future carriers deliberately excluded until after global seal'})
 save(marker,{'driver_sha256':sha(Path(__file__)),'B_sha256':sha(BANK),'D_sha256':sha(D/'outputs/frozen_policy.json'),
  'derived_evaluation_sha256':derived,'bank_size':len(ids),'reference':ref})
def worker(mode,slug):
 sys.path.insert(0,str(C/'scripts'));import compute as u
 lock=read(E/'run_lock.json')
 for rel,digest in lock['files'].items():
  p=C/rel if (C/rel).exists() else ROOT/rel
  if sha(p)!=digest:raise ValueError('E dependency changed '+rel)
 ev,derived=dynamic_evaluation()
 if derived!=lock['derived_evaluation_sha256']:raise ValueError('Derived evaluation changed')
 u.configure_stage2_runtime(u.DEVICE,'gru-native');cases=read(E/'inputs/cases.json');case=next(x for x in cases if x['slug']==slug)
 if mode=='select':ev.select(case,lambda *_:None)
 elif mode=='score':ev.score(case,lambda *_:None)
 else:raise ValueError('Bad E phase')
def main():
 SUP.mkdir(parents=True,exist_ok=True);lock=(SUP/'supervisor.lock').open('a+')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('E supervisor already running')
 from dynamic_gpu_controller import Controller
 controller=Controller(SUP/'concurrency_events.jsonl');controller.caps={0:12}
 started=time.time();active={};queue=[];done=[];reason='';phase='D'
 def status(state):save(SUP/'status.json',{'state':state,'phase':phase,'start_unix':started,'updated_unix':time.time(),'pending':len(queue),'active':list(active),'completed':done,'reason':reason,'disk_free_bytes':shutil.disk_usage(ROOT).free})
 def interrupt(signum,frame):raise RuntimeError('supervisor interrupted')
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 def runphase(mode,cases):
  nonlocal queue,done,phase
  phase='E_'+mode;queue=[];done=[]
  for c in cases:
   slug=c['slug'];f=E/'results'/slug/('selected.json' if mode=='select' else 'scores.json')
   (done if f.exists() else queue).append(slug)
  while queue or active:
   controller.tick(active,len(done),bool(queue))
   check({x['proc'].pid for x in active.values()})
   for tag,x in list(active.items()):
    rc=x['proc'].poll()
    if rc is None:continue
    x['log'].close();f=E/'results'/tag/('selected.json' if mode=='select' else 'scores.json')
    if rc or not f.exists():del active[tag];raise RuntimeError('E '+mode+' worker failed '+tag)
    done.append(tag);del active[tag]
   while queue and len(active)<12:
    options=controller.available(active)
    if not options:break
    gpu=min(options,key=lambda g:sum(x['gpu']==g for x in active.values()))
    slug=queue.pop(0);(SUP/'logs').mkdir(exist_ok=True);log=(SUP/'logs'/(mode+'_'+slug+'.log')).open('ab',buffering=0)
    proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',mode,slug],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
    active[slug]={'proc':proc,'log':log,'gpu':gpu}
   status('running');time.sleep(5)
 try:
  while True:
   d=read(DSTATE/'status.json')
   if d['state']=='paused':raise RuntimeError('D paused: '+d['reason'])
   if d['state']=='completed' and (DSTATE/'D_COMPLETE.json').exists():break
   status('waiting');time.sleep(15)
  prepare();(E/'outputs').mkdir(exist_ok=True);cases=read(E/'inputs/cases.json')
  if len(cases)!=120:raise ValueError('E cases not 120')
  ev,_=dynamic_evaluation();runphase('select',cases);ev.seal(cases)
  seal=E/'outputs/all_actions.json';save(SUP/'E_SELECTION_COMPLETE.json',{'seal_sha256':sha(seal),'completed_unix':time.time()})
  archived=ROOT/'diagnostic_strong50_flow/C_real/fresh_E_stage/inputs/future'
  for case in cases:
   src=archived/(case['slug']+'.bin');dest=E/'inputs/future'/(case['slug']+'.bin')
   if sha(src)!=case['future_hash']:raise ValueError('Archived E future input differs')
   if dest.exists():
    if sha(dest)!=case['future_hash']:raise ValueError('E future input conflict')
   else:os.link(src,dest)
  phase='E_wait_future';done=[];queue=[]
  while True:
   missing=[]
   for c in cases:
    p=E/'inputs/future'/(c['slug']+'.bin')
    if not p.exists() or sha(p)!=c['future_hash']:missing.append(c['slug'])
   if not missing:break
   if shutil.disk_usage(ROOT).free<2.5*1024**3:raise RuntimeError('disk guard while waiting for future carriers')
   queue=missing;status('waiting_future');time.sleep(15)
  runphase('score',cases);ev.summarize(cases)
  save(SUP/'E_COMPLETE.json',{'summary_sha256':sha(E/'outputs/verified_summary.json'),'completed_unix':time.time()});phase='E_complete';status('completed')
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
 if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(sys.argv[2],sys.argv[3])
 else:main()
