"""Fresh, independently sealed target search for three native baselines.

Run one supervisor per method: `python3 -B fresh_search_stage.py METHOD GPU`.
Selection reads only target prefixes; scoring starts after all 120 decisions
for that method have been sealed. CUDA workers are capped at 12 per GPU.
"""
import fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ORIG=ROOT/'original'
C=next((ROOT/'flow').glob('C*'))
E=C/'fresh_E_stage'
WEIGHTS=ROOT/'fresh_source_stage/weights'
TEACHER=ROOT/'fresh_search_teacher'
SEARCH_ROOT=ROOT/('fresh_search_smoke' if os.environ.get('MSADTI_SEARCH_SMOKE')=='1' else 'fresh_search_target')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['PYTHONDONTWRITEBYTECODE']='1'
METHOD_GPUS={'meta_nas_lite':0,'zero_nas':1,'zero_nas_ft':3}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.pending')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)

def savept(path,obj,torch):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.pending');torch.save(obj,tmp);os.replace(tmp,path)

def npz(path,arrays,np):
    tmp=path.with_name(path.name+f'.{os.getpid()}.pending')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    os.replace(tmp,path)

class PrefixFence:
    def __init__(self):self.active=False;self.denied=[]
    def hook(self,event,args):
        if not self.active or event!='open' or isinstance(args[0],int):return
        p=Path(os.fsdecode(args[0])).resolve()
        if any(p.is_relative_to(x) for x in (E/'inputs/future',ROOT/'fresh_E_legacy/data/future',C/'inputs/assessment',ORIG/'results')):
            self.denied.append(str(p));raise PermissionError('Future or old result access during selection')
    def __enter__(self):sys.addaudithook(self.hook);self.active=True;return self
    def __exit__(self,*args):self.active=False

def import_runtime():
    os.environ['MSADTI_OUTPUT']=str(ORIG/'fresh_source_run');sys.path.insert(0,str(ORIG/'src'))
    import numpy as np
    import torch
    from msa_dti import runtime as u
    from msa_dti.prefix import prefix_selection,DTYPE
    from msa_dti.features import causal_features
    return np,torch,u,prefix_selection,DTYPE,causal_features

def lock_doc(method,gpu):
    cases=read(E/'inputs/cases.json')
    if len(cases)!=120 or len({c['slug'] for c in cases})!=120:raise ValueError('Wrong E task list')
    files=[Path(__file__),ORIG/'src/msa_dti/baseline_methods.py',ORIG/'src/msa_dti/runtime.py',
           E/'run_lock.json',E/'inputs/cases.json',E/'inputs/normalization.json']
    files += [WEIGHTS/f'H{h}'/f'A{i}.pt' for h in (1,4) for i in range(66)]
    for c in cases:
        p=E/'inputs/prefix'/f"{c['slug']}.bin"
        if sha(p)!=c['prefix_hash']:raise ValueError('Prefix hash mismatch')
        files.append(p)
    for h in (1,4):
        for i in range(66):
            p=WEIGHTS/f'H{h}'/f'A{i}.pt'; receipt=read(ROOT/'fresh_source_stage/receipts'/f'H{h}_A{i}.json')
            if sha(p)!=receipt['weight_sha256']:raise ValueError('Source weight mismatch')
    if method=='meta_nas_lite':
        receipt=read(TEACHER/'COMPLETE.json')
        if sha(TEACHER/'pi50.json')!=receipt['pi50_sha256']:raise ValueError('Teacher checksum mismatch')
        files += [TEACHER/'pi50.json',TEACHER/'COMPLETE.json']
    return {'method':method,'gpu':gpu,'workers':12,'tasks':120,'tier':'loose',
            'selection':'Native search over all 66; other budgets checked after selection',
            'files':{str(p.relative_to(ROOT)):sha(p) for p in files}}

def select(method,slug):
    np,torch,u,prefix_selection,_,_=import_runtime()
    cases=read(E/'inputs/cases.json'); c=next(x for x in cases if x['slug']==slug)
    out=SEARCH_ROOT/method/'results'/slug;out.mkdir(parents=True,exist_ok=True)
    lock=SEARCH_ROOT/method/'run_lock.json';lockhash=sha(lock)
    prefix=E/'inputs/prefix'/f'{slug}.bin'
    if sha(prefix)!=c['prefix_hash']:raise ValueError('Prefix changed')
    h=c['H'];s=c['seed'];fence=PrefixFence();start=time.time()
    with fence:
        data=prefix_selection(c,prefix,read(E/'inputs/normalization.json'))
        xs,ys,xv,yv=[torch.from_numpy(data[x]) for x in ('Xs','ys','Xv','yv')]
        bank={}
        for i in range(66):
            obj=torch.load(WEIGHTS/f'H{h}'/f'A{i}.pt',map_location='cpu',weights_only=True)
            key=u.make_bank_key_shared(H=h,arch_key=u.ARCH[i].arch_key,input_dim=25,L=96)
            bank[key]=obj['state_dict']
        pi=read(TEACHER/'pi50.json')['pi50'] if method=='meta_nas_lite' else None
        feasible=u.native.feasible_indices(u.CFG,u.ARCH,'loose',96,25,h)
        if set(feasible)!=set(range(66)):raise ValueError('Loose tier does not include all 66 architectures')
        model,spec,row,details,count,adapted=u.native._run_search_case(method,{'bank':bank,'pi50':pi},u.CFG,u.ARCH,u.DEVICE,'gru-native',xs,ys,xv,yv,'loose',h,25,96,s)
        idx=int(row['arch_idx']);model.eval()
        with u.candidate_backend_context(u.ARCH[idx],u.DEVICE,'gru-native'),torch.no_grad():prediction=model(xv.to(u.DEVICE).contiguous()).detach().cpu().numpy()
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        savept(out/'model.pt',state,torch);npz(out/'validation.npz',{'prediction':prediction},np)
        par,ops=u.profile_arch(u.ARCH[idx],96,25,h)
        save(out/'selected.json',{'method':method,'slug':slug,'VM':c['VM'],'H':h,'K':c['K'],'arch':idx,
             'params':par,'operations':ops,'selection':details,'validation':row['validation'],
             'shortlist_count':count,'adapted_count':adapted,'model_sha256':sha(out/'model.pt'),
             'validation_sha256':sha(out/'validation.npz'),'prefix_sha256':c['prefix_hash'],
             'run_lock_sha256':lockhash,'future_opened':False,'start_unix':start,'end_unix':time.time()})
    if fence.denied:raise ValueError('Forbidden target data read')

def metrics(pred,y,np):
    if pred.shape!=y.shape or not len(y):raise ValueError('Invalid prediction shape')
    err=pred.astype(np.float64)-y.astype(np.float64);loss=np.square(err).mean(1)
    if not np.isfinite(loss).all():raise ValueError('Invalid future metric')
    return {'MSE':float(loss.mean()),'MAE':float(np.abs(err).mean()),'Worst10':float(np.sort(loss)[-max(1,round(.1*len(loss))):].mean())}

def score(method,slug):
    np,torch,u,_,DTYPE,causal_features=import_runtime()
    c=next(x for x in read(E/'inputs/cases.json') if x['slug']==slug)
    root=SEARCH_ROOT/method;out=root/'results'/slug;seal=read(root/'SELECTION_SEAL.json')
    if seal['selected'][slug]!=sha(out/'selected.json'):raise ValueError('Selection seal mismatch')
    selected=read(out/'selected.json')
    if sha(out/'model.pt')!=selected['model_sha256'] or sha(out/'validation.npz')!=selected['validation_sha256']:raise ValueError('Model or validation changed')
    prefix=E/'inputs/prefix'/f'{slug}.bin';future=E/'inputs/future'/f'{slug}.bin'
    if sha(future)!=c['future_hash']:raise ValueError('Future hash mismatch')
    raw=np.fromfile(future,dtype=DTYPE);t,h=c['T'],c['H']
    if len(raw)!=t+c['B']+h or raw[:t+1].tobytes()!=prefix.read_bytes():raise ValueError('Future prefix mismatch')
    origins=np.arange(t,t+c['B']);cpu=raw['raw'][origins[:,None]+np.arange(1,h+1),3]
    valid=np.all(np.isfinite(cpu)&(cpu>=0),axis=1);y=(cpu[valid]/100).astype(np.float32)
    features=causal_features(raw['timestamp'],raw['raw'],read(E/'inputs/normalization.json'))
    x=torch.from_numpy(np.stack([features[o-95:o+1] for o in origins]).astype(np.float32)).to(u.DEVICE)
    model=u.build(selected['arch'],h,torch.load(out/'model.pt',map_location='cpu',weights_only=True),c['seed']);model.eval()
    with u.candidate_backend_context(u.ARCH[selected['arch']],u.DEVICE,'gru-native'),torch.no_grad():prediction=model(x).detach().cpu().numpy()[valid]
    npz(out/'forecast.npz',{'prediction':prediction},np)
    budgets=read(E/'protocol.json')['budgets'];feasible={tier:selected['params']<=p and selected['operations']<=op for tier,(p,op) in budgets.items()}
    save(out/'scores.json',{'method':method,'slug':slug,'VM':c['VM'],'H':h,'K':c['K'],'arch':selected['arch'],
         'feasible':feasible,'future_sha256':c['future_hash'],'selected_sha256':sha(out/'selected.json'),
         'forecast_sha256':sha(out/'forecast.npz'),'seal_sha256':sha(root/'SELECTION_SEAL.json'),
         'n_valid':len(y),**metrics(prediction,y,np)})

def main(method,gpu):
    if METHOD_GPUS.get(method)!=gpu:raise ValueError('Invalid assigned GPU')
    out=SEARCH_ROOT/method;out.mkdir(parents=True,exist_ok=True)
    with (out/'supervisor.lock').open('a+') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock=out/'run_lock.json';doc=lock_doc(method,gpu)
        if lock.exists():
            if read(lock)!=doc:raise ValueError('Search inputs changed')
        else:save(lock,doc)
        if not (out/'stage_start.json').exists():save(out/'stage_start.json',{'start_unix':time.time(),'run_lock_sha256':sha(lock),'scope':'Target search only; common source and teacher recorded separately'})
        active={};queue=[];done=[];phase='start';reason=''
        def status(state):save(out/'status.json',{'state':state,'phase':phase,'reason':reason,'updated_unix':time.time(),'pending':len(queue),'active':list(active),'completed':done,'disk_free_bytes':shutil.disk_usage(ROOT).free})
        def stop(sig,frame):raise RuntimeError('Supervisor interrupted')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        def runphase(mode,cases):
            nonlocal phase,queue,done
            phase=mode;done=[];queue=[]
            for c in cases:
                slug=c['slug'];path=out/'results'/slug/('selected.json' if mode=='select' else 'scores.json')
                (done if path.exists() else queue).append(slug)
            while queue or active:
                if shutil.disk_usage(ROOT).free<2.5*1024**3:raise RuntimeError('disk guard')
                mem=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
                if mem<16*1024**3:raise RuntimeError('host memory guard')
                uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
                lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines(); own={x['p'].pid for x in active.values()}
                if any(u.strip()==uuid and p.strip().isdigit() and int(p.strip()) not in own for u,p in (line.split(',') for line in lines)):raise RuntimeError('GPU conflict')
                for slug,x in list(active.items()):
                    rc=x['p'].poll()
                    if rc is None:continue
                    x['log'].close();del active[slug]
                    if rc:raise RuntimeError('Worker failed '+mode+' '+slug)
                    done.append(slug)
                while queue and len(active)<12:
                    slug=queue.pop(0);(out/'logs').mkdir(exist_ok=True);log=(out/'logs'/f'{mode}_{slug}.log').open('ab',buffering=0)
                    proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker',method,mode,slug],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)))
                    active[slug]={'p':proc,'log':log}
                status('running');time.sleep(5)
        try:
            cases=read(E/'inputs/cases.json');runphase('select',cases)
            selected={}
            for c in cases:
                p=out/'results'/c['slug']; a=read(p/'selected.json')
                if a['run_lock_sha256']!=sha(lock) or a['model_sha256']!=sha(p/'model.pt') or a['validation_sha256']!=sha(p/'validation.npz'):raise ValueError('Selection artifact mismatch')
                selected[c['slug']]=sha(p/'selected.json')
            seal=out/'SELECTION_SEAL.json'
            if seal.exists():
                if read(seal)['selected']!=selected:raise ValueError('Changed prior selection seal')
            else:save(seal,{'selected':selected,'run_lock_sha256':sha(lock),'selected_before_future_scoring':True,'end_unix':time.time()})
            runphase('score',cases)
            rows=[]
            for c in cases:
                p=out/'results'/c['slug'];a=read(p/'selected.json');v=read(p/'scores.json')
                if v['selected_sha256']!=sha(p/'selected.json') or v['seal_sha256']!=sha(seal) or v['forecast_sha256']!=sha(p/'forecast.npz') or v['future_sha256']!=c['future_hash']:raise ValueError('Score integrity mismatch')
                rows.append(v)
            save(out/'all_scores.json',{'tasks':120,'rows':rows,'seal_sha256':sha(seal),'status':'HASH_VERIFIED'})
            save(out/'COMPLETE.json',{'tasks':120,'scores_sha256':sha(out/'all_scores.json'),'end_unix':time.time(),'requires_independent_numerical_recheck':True})
            phase='complete';status('completed')
        except Exception as e:reason=str(e);status('paused');print('PAUSED',reason,flush=True)
        finally:
            for x in active.values():
                if x['p'].poll() is None:x['p'].terminate()
            for x in active.values():
                try:x['p'].wait(timeout=15)
                except subprocess.TimeoutExpired:x['p'].kill();x['p'].wait()
                x['log'].close()

def smoke(method,gpu):
    if os.environ.get('MSADTI_SEARCH_SMOKE')!='1' or METHOD_GPUS.get(method)!=gpu:raise ValueError('Invalid isolated smoke setup')
    out=SEARCH_ROOT/method;out.mkdir(parents=True,exist_ok=True)
    lock=out/'run_lock.json';doc=lock_doc(method,gpu)
    if lock.exists() and read(lock)!=doc:raise ValueError('Smoke identity changed')
    if not lock.exists():save(lock,doc)
    c=read(E/'inputs/cases.json')[0];select(method,c['slug']);folder=out/'results'/c['slug'];result=read(folder/'selected.json')
    if result['model_sha256']!=sha(folder/'model.pt') or result['validation_sha256']!=sha(folder/'validation.npz'):raise ValueError('Smoke hash mismatch')
    save(out/'SMOKE_PASS.json',{'method':method,'slug':c['slug'],'selected_sha256':sha(folder/'selected.json'),'scope':'Smoke only; not formal run'})

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        if sys.argv[3]=='select':select(sys.argv[2],sys.argv[4])
        elif sys.argv[3]=='score':score(sys.argv[2],sys.argv[4])
        else:raise ValueError('Bad worker mode')
    elif len(sys.argv)>1 and sys.argv[1]=='--smoke':smoke(sys.argv[2],int(sys.argv[3]))
    else:main(sys.argv[1],int(sys.argv[2]))
