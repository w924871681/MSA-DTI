"""Calibrate the fresh reference and fresh B bank on D; wait for verified C."""
import csv,fcntl,hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parent;C=next((ROOT/'timing_s50_flow_20261104').glob('C*'));OLD=C/'D_stage';D=C/'fresh_D_stage'
CSTATE=ROOT/'timing_s50_c_supervisor_20261104';SUP=ROOT/'timing_s50_d_supervisor_20261104';A=ROOT/'timing_s50_ab_20261104/A_fresh';BANK=ROOT/'timing_s50_ab_20261104/B_fresh/outputs/retained_bank.json'
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

def dynamic_calibration():
 path=C/'scripts/calibration.py';source=path.read_text(encoding='utf8')
 old='assert len(eligible)==20 and len(set(eligible))==20';new='assert len(eligible)>0 and len(set(eligible))==len(eligible)'
 if source.count(old)!=1:raise ValueError('Unexpected calibration source; cannot generalize bank size')
 modified=source.replace(old,new);sys.path.insert(0,str(C/'scripts'));module=types.ModuleType('dynamic_calibration');module.__file__=str(path)
 exec(compile(modified,str(path),'exec'),module.__dict__);module.D=D
 return module,hashlib.sha256(modified.encode()).hexdigest()
def prepare():
 if read(CSTATE/'status.json')['state']!='completed':raise RuntimeError('Fresh C selection incomplete')
 ref=read(C/'outputs/msa_selected.json');bank=read(BANK);ids=bank['retained_including_comparator'];profiles={}
 with (A/'inputs/validation_mse.csv').open(encoding='utf8',newline='') as f:
  for row in csv.DictReader(f):profiles[int(row['arch_id']),int(row['horizon'])]=(int(row['params']),int(row['operations']))
 cap=read(C/'E_stage/protocol.json')['budgets']['medium']
 eligible=[i for i in ids if all(profiles[i,h][0]<=cap[0] and profiles[i,h][1]<=cap[1] for h in (1,4))]
 if ref['selected']['arch_id'] not in eligible:raise ValueError('C reference not feasible under D medium budget')
 marker=D/'prepare.json'
 if marker.exists():
  m=read(marker)
  if m['driver_sha256']!=sha(Path(__file__)) or m['B_sha256']!=sha(BANK) or m['C_sha256']!=sha(C/'outputs/msa_selected.json'):raise ValueError('D dependencies changed')
  return
 if D.exists():raise RuntimeError('Partial D preparation requires review')
 D.mkdir();shutil.copytree(OLD/'inputs',D/'inputs',ignore=shutil.ignore_patterns('checkpoints','weights.json'))
 protocol=read(OLD/'protocol.json');protocol['eligible']=eligible;save(D/'protocol.json',protocol)
 registry=read(C/'fresh_msa/dynamic_inputs.json')['registry'];weights={}
 for i in eligible:
  for h in (1,4):
   k=f'msa/{h}/{i}';ent=registry[k]
   if sha(C/ent['path'])!=ent['sha256']:raise ValueError('Fresh source weight changed')
   weights[k]=ent
 save(D/'inputs/weights.json',weights)
 cal,derived_hash=dynamic_calibration();files={}
 for p in [D/'protocol.json',C/'outputs/msa_selected.json',BANK,C/'scripts/calibration.py',C/'scripts/compute.py',*list((D/'inputs').rglob('*'))]:
  if p.is_file():files[str(p.relative_to(C) if p.is_relative_to(C) else p.relative_to(ROOT))]=sha(p)
 save(D/'run_lock.json',{'files':files,'derived_calibration_sha256':derived_hash,'scope':'Fresh B bank and fresh C reference, dynamic number of medium-feasible configurations'})
 save(marker,{'driver_sha256':sha(Path(__file__)),'B_sha256':sha(BANK),'C_sha256':sha(C/'outputs/msa_selected.json'),
  'eligible':eligible,'derived_calibration_sha256':derived_hash,'source_calibration_sha256':sha(C/'scripts/calibration.py')})
def worker(slug):
 sys.path.insert(0,str(C/'scripts'));import compute as u
 lock=read(D/'run_lock.json')
 for rel,digest in lock['files'].items():
  p=C/rel if (C/rel).exists() else ROOT/rel
  if sha(p)!=digest:raise ValueError('D dependency changed '+rel)
 cal,derived=dynamic_calibration()
 if derived!=lock['derived_calibration_sha256']:raise ValueError('Modified calibration differs from lock')
 u.configure_stage2_runtime(u.DEVICE,'gru-native');cases=read(D/'inputs/cases.json');case=next(x for x in cases if x['slug']==slug)
 cal.run_case(case,read(C/'outputs/msa_selected.json'))
def main():
 SUP.mkdir(parents=True,exist_ok=True);lock=(SUP/'supervisor.lock').open('a+')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('D supervisor already running')
 start=time.time();active={};queue=[];done=[];reason='';phase='C'
 def status(state):save(SUP/'status.json',{'state':state,'phase':phase,'start_unix':start,'updated_unix':time.time(),'pending':len(queue),'active':list(active),'completed':done,'reason':reason,'disk_free_bytes':shutil.disk_usage(ROOT).free})
 def interrupted(signum,frame):raise RuntimeError('supervisor interrupted')
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  while True:
   s=read(CSTATE/'status.json')
   if s['state']=='paused':raise RuntimeError('C paused: '+s['reason'])
   if s['state']=='completed' and (CSTATE/'C_COMPLETE.json').exists():break
   status('waiting');time.sleep(15)
  phase='D';prepare();cases=read(D/'inputs/cases.json')
  if len(cases)!=80:raise ValueError('D cases not 80')
  for c in cases:
   slug=c['slug'];f=D/'results'/slug
   if (f/'scores.json').exists():
    r=read(f/'scores.json')
    if sha(f/'predictions.npz')!=r['panel_sha256'] or sha(f/'validation_seal.json')!=r['seal_sha256']:raise ValueError('Invalid D result '+slug)
    done.append(slug)
   else:queue.append(slug)
  while queue or active:
   check({x['proc'].pid for x in active.values()})
   for tag,x in list(active.items()):
    rc=x['proc'].poll()
    if rc is None:continue
    x['log'].close();folder=D/'results'/tag
    if rc or not (folder/'scores.json').exists():del active[tag];raise RuntimeError('D worker failed '+tag)
    row=read(folder/'scores.json')
    if sha(folder/'predictions.npz')!=row['panel_sha256'] or sha(folder/'validation_seal.json')!=row['seal_sha256']:raise ValueError('Invalid D panel')
    done.append(tag);del active[tag]
   while queue and len(active)<12:
    options=available_gpus(active)
    if not options:break
    tag=queue.pop(0);(SUP/'logs').mkdir(exist_ok=True);log=(SUP/'logs'/(tag+'.log')).open('ab',buffering=0)
    gpu=min(options,key=lambda g:sum(x['gpu']==g for x in active.values()));p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',tag],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
    active[tag]={'proc':p,'log':log,'gpu':gpu}
   status('running');time.sleep(5)
  cal,_=dynamic_calibration();cal.summarize();policy=D/'outputs/frozen_policy.json'
  save(SUP/'D_COMPLETE.json',{'policy_sha256':sha(policy),'completed_unix':time.time(),'eligible':len(read(D/'protocol.json')['eligible'])});status('completed')
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
 if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(sys.argv[2])
 else:main()
