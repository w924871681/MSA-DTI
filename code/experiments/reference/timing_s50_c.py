"""Wait for fresh B bank, evaluate jointly deployable members on C, then select reference."""
import csv,fcntl,hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
C=next((ROOT/'timing_s50_flow_20261104').glob('C*'));A=ROOT/'timing_s50_ab_20261104/A_fresh';BANK=ROOT/'timing_s50_ab_20261104/B_fresh/outputs/retained_bank.json'
OUT=C/'fresh_msa';SUP=ROOT/'timing_s50_c_supervisor_20261104'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['PYTHONDONTWRITEBYTECODE']='1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.pending')
 t.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n');os.replace(t,p)
def gpu():
 out=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=20)
 uuid,used,total=[x.strip() for x in out.strip().split(',')];return uuid,int(used),int(total)
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
 bank=read(BANK);members=bank['members'];ids=bank['retained_including_comparator']
 if sorted(x['arch_id'] for x in members)!=sorted(ids):raise ValueError('B bank membership mismatch')
 profile={}
 with (A/'inputs/validation_mse.csv').open(encoding='utf8',newline='') as f:
  for row in csv.DictReader(f):
   key=(int(row['arch_id']),int(row['horizon']));value=(int(row['params']),int(row['operations']))
   if key in profile and profile[key]!=value:raise ValueError('Inconsistent A resource profile')
   profile[key]=value
 budgets=read(C/'E_stage/protocol.json')['budgets']
 eligible=[i for i in ids if all(profile[i,h][0]<=cap[0] and profile[i,h][1]<=cap[1]
                                 for h in (1,4) for cap in budgets.values())]
 ref=int(bank['screening_comparator'].split('A')[-1])
 if ref not in eligible:raise ValueError('B comparator is not jointly deployable on C')
 registry={};OUT.mkdir(parents=True,exist_ok=True)
 for member in members:
  i=member['arch_id']
  for h in (1,4):
   ent=member['source_checkpoints'][f'H{h}'];src=A/ent['path_relative_to_A'];expected=ent['sha256']
   if sha(src)!=expected:raise ValueError('Changed fresh source checkpoint')
   target=OUT/'weights'/f'H{h}'/f'A{i}.pt';target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists():
    if sha(target)!=expected:raise ValueError('Existing fresh source differs')
   else:shutil.copyfile(src,target)
   registry[f'msa/{h}/{i}']={'path':target.relative_to(C).as_posix(),'sha256':expected}
 state={'bank_sha256':sha(BANK),'source_C_lock_sha256':sha(C/'run_lock.json'),
        'driver_sha256':sha(Path(__file__)),'retained':ids,'eligible':eligible,'reference_for_screen':ref,'registry':registry}
 identity=hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode()).hexdigest();state['identity']=identity
 p=OUT/'dynamic_inputs.json'
 if p.exists():
  if read(p)!=state:raise ValueError('Dynamic C inputs changed on resume')
 else:save(p,state)
 return state
def worker(i,h):
 spec=importlib.util.spec_from_file_location('frozen_compute',C/'scripts/compute.py');u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
 for rel,d in u.read(C/'run_lock.json')['files'].items():
  if u.sha(C/rel)!=d:raise ValueError('C frozen dependency changed: '+rel)
 for rel,d in u.read(C/'inputs/files.json').items():
  if u.sha(C/rel)!=d:raise ValueError('C frozen input changed: '+rel)
 inp=read(OUT/'dynamic_inputs.json')
 if i not in inp['eligible'] or h not in (1,4):raise ValueError('Ineligible C task')
 u.OUT=OUT;u.configure_stage2_runtime(u.DEVICE,'gru-native')
 u.work('msa',i,h,inp['identity'],u.read(C/'inputs/cases.json'),inp['registry'])
def report(inp):
 for i in inp['eligible']:
  for h in (1,4):
   tag=f'msa_A{i}_H{h}';a=OUT/'jobs'/tag;b=C/'results/jobs'/tag;b.mkdir(parents=True,exist_ok=True)
   score=read(a/'scores.json')
   if score['signature']!=inp['identity'] or sha(a/'predictions.npz')!=score['predictions_sha256']:raise ValueError('Invalid C result')
   for n in ('scores.json','predictions.npz'):
    src=a/n;dst=b/n
    if dst.exists():
     if sha(dst)!=sha(src):raise ValueError('Conflicting C report data')
    else:os.link(src,dst)
 spec=importlib.util.spec_from_file_location('original_report',C/'scripts/report.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
 origsha,origread=r.sha,r.read
 r.sha=lambda p:inp['identity'] if Path(p)==C/'run_lock.json' else origsha(p)
 def read_dynamic(p):
  val=origread(p)
  if Path(p)==C/'protocol.json':val['eligible']['msa']=inp['eligible']
  return val
 r.read=read_dynamic
 target=C/'outputs/msa_selected.json'
 if target.exists():raise ValueError('MSA reference already exists; inspect before reuse')
 r.main('msa')
 save(SUP/'C_COMPLETE.json',{'selected_sha256':sha(target),'selected_arch':read(target)['selected']['arch_id'],
  'fresh_bank_sha256':inp['bank_sha256'],'identity':inp['identity'],'completed_unix':time.time(),
  'note':'Original report rechecks C prediction MSE; eligibility and identity are from the fresh B bank.'})
def main():
 SUP.mkdir(parents=True,exist_ok=True);lock=(SUP/'supervisor.lock').open('a+')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise RuntimeError('C supervisor already active')
 start=time.time();active={};queue=[];done=[];reason=''
 def status(state,phase):save(SUP/'status.json',{'state':state,'phase':phase,'started_unix':start,'updated_unix':time.time(),
  'pending':len(queue),'active':list(active),'completed':done,'reason':reason,'disk_free_bytes':shutil.disk_usage(ROOT).free})
 def interrupted(signum,frame):raise RuntimeError('supervisor interrupted')
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  while True:
   ab=read(ROOT/'timing_s50_ab_20261104/status.json')
   if ab['state']=='paused':raise RuntimeError('A/B paused: '+ab['reason'])
   if ab['state']=='completed' and ab['phase']=='B_complete' and BANK.exists():break
   status('waiting','B');time.sleep(15)
  inp=prepare()
  for i in inp['eligible']:
   for h in (1,4):
    tag=f'msa_A{i}_H{h}';folder=OUT/'jobs'/tag
    if (folder/'scores.json').exists() and (folder/'predictions.npz').exists():
     r=read(folder/'scores.json')
     if r['signature']!=inp['identity'] or sha(folder/'predictions.npz')!=r['predictions_sha256']:raise ValueError('Bad completed task '+tag)
     done.append(tag)
    elif (folder/'scores.json').exists() or (folder/'predictions.npz').exists():raise ValueError('Partial task requires review '+tag)
    else:queue.append((i,h))
  while queue or active:
   check({x['process'].pid for x in active.values()})
   for tag,x in list(active.items()):
    rc=x['proc'].poll()
    if rc is None:continue
    x['log'].close();folder=OUT/'jobs'/tag
    if rc or not (folder/'scores.json').exists() or sha(folder/'predictions.npz')!=read(folder/'scores.json')['predictions_sha256']:
     del active[tag];raise RuntimeError('C worker failed '+tag)
    done.append(tag);del active[tag]
   while queue and len(active)<12:
    options=available_gpus(active)
    if not options:break
    i,h=queue.pop(0);tag=f'msa_A{i}_H{h}';(SUP/'logs').mkdir(exist_ok=True)
    log=(SUP/'logs'/(tag+'.log')).open('ab',buffering=0);gpu=min(options,key=lambda g:sum(x['gpu']==g for x in active.values()));env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
    p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(i),str(h)],env=env,stdout=log,stderr=subprocess.STDOUT)
    active[tag]={'proc':p,'process':p,'log':log,'gpu':gpu}
   status('running','C');time.sleep(5)
  report(inp);status('completed','C')
 except Exception as e:
  reason=str(e);status('paused','C');print('PAUSED',reason,flush=True)
 finally:
  for x in active.values():
   if x['proc'].poll() is None:x['proc'].terminate()
  for x in active.values():
   try:x['proc'].wait(timeout=15)
   except subprocess.TimeoutExpired:x['proc'].kill();x['proc'].wait()
   x['log'].close()
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(int(sys.argv[2]),int(sys.argv[3]))
 else:main()
