"""Sealed H-Meta-NAS target search after its original source recipe completes."""
import fcntl, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path
import fresh_search_stage as fs

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'fresh_hmeta_target'; SOURCE=ROOT/'fresh_hmeta_source';E=fs.E;GPU=1
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['PYTHONDONTWRITEBYTECODE']='1'

def source_lock():
    source=fs.read(SOURCE/'SOURCE_COMPLETE.json');bank=SOURCE/'hmeta.pt'
    if fs.sha(bank)!=source['sha256'] or source['iterations']!=240:raise ValueError('Invalid H-meta source bank')
    files=[Path(__file__),ROOT/'fresh_search_stage.py',SOURCE/'SOURCE_COMPLETE.json',bank,
           fs.ORIG/'src/msa_dti/baseline_runtime.py',fs.ORIG/'src/msa_dti/hmeta.py',E/'inputs/cases.json',E/'inputs/normalization.json',E/'protocol.json']
    cases=fs.read(E/'inputs/cases.json')
    if len(cases)!=120:raise ValueError('Wrong E task count')
    for c in cases:
        p=E/'inputs/prefix'/f"{c['slug']}.bin"
        if fs.sha(p)!=c['prefix_hash']:raise ValueError('Prefix hash mismatch')
        files.append(p)
    return {'method':'h_meta_nas','gpu':GPU,'workers':12,'scope':'Native budget-aware search on each tier',
            'files':{str(p.relative_to(ROOT)):fs.sha(p) for p in files}}

def select(slug,smoke=False):
    np,torch,u,prefix_selection,_,_=fs.import_runtime()
    from msa_dti import baseline_runtime as br
    c=next(x for x in fs.read(E/'inputs/cases.json') if x['slug']==slug)
    root=OUT/'smoke' if smoke else OUT;folder=root/'results'/slug;folder.mkdir(parents=True,exist_ok=True)
    prefix=E/'inputs/prefix'/f'{slug}.bin'
    if fs.sha(prefix)!=c['prefix_hash']:raise ValueError('Prefix changed')
    br.HM=torch.load(SOURCE/'hmeta.pt',map_location='cpu',weights_only=False);br.HMHASH=fs.sha(SOURCE/'hmeta.pt')
    if not br.HM.get('complete') or br.HM.get('completed_iterations')!=240:raise ValueError('Source bank incomplete')
    br.CURRENT=dict(case=c,prefix_sha=fs.sha(prefix),native_hash=fs.sha(Path(br.__file__)),cache=root/'cache'/slug,
                    logical_adaptations=0,actual_adaptations=0,cache_hits=0,events=[],proxies={})
    fence=fs.PrefixFence();actions={};start=time.time()
    with fence:
        data=prefix_selection(c,prefix,fs.read(E/'inputs/normalization.json'))
        xs,ys,xv,yv=[torch.from_numpy(data[k]) for k in ('Xs','ys','Xv','yv')]
        for tier in ('medium','tight','loose'):
            state,idx,details=br.hmeta(c['H'],tier,c['seed'],xs,ys,xv,yv)
            if idx not in u.native.feasible_indices(u.CFG,u.ARCH,tier,96,25,c['H']):raise ValueError('Search selected infeasible architecture')
            model=u.build(idx,c['H'],state,c['seed']);model.eval()
            with u.candidate_backend_context(u.ARCH[idx],u.DEVICE,'gru-native'),torch.no_grad():prediction=model(xv.to(u.DEVICE).contiguous()).cpu().numpy()
            path=folder/tier;path.mkdir(exist_ok=True);fs.savept(path/'model.pt',state,torch);fs.npz(path/'validation.npz',{'prediction':prediction},np)
            par,ops=u.profile_arch(u.ARCH[idx],96,25,c['H']);actual=details['evaluated_candidates']
            chosen=next(r for r in actual if r['arch_idx']==idx)
            fs.save(path/'action.json',{'tier':tier,'arch':idx,'params':par,'operations':ops,'validation':chosen['validation'],
                'details':details,'model_sha256':fs.sha(path/'model.pt'),'validation_sha256':fs.sha(path/'validation.npz')})
            actions[tier]=fs.sha(path/'action.json')
        fs.save(folder/'selected.json',{'slug':slug,'VM':c['VM'],'H':c['H'],'K':c['K'],'method':'h_meta_nas',
            'actions':actions,'prefix_sha256':c['prefix_hash'],'run_lock_sha256':fs.sha(OUT/'run_lock.json'),
            'future_opened':False,'start_unix':start,'end_unix':time.time(),'actual_adaptations':br.CURRENT['actual_adaptations'],
            'cache_hits':br.CURRENT['cache_hits']})
    if fence.denied:raise ValueError('Forbidden target data read')

def score(slug):
    np,torch,u,_,DTYPE,causal_features=fs.import_runtime()
    c=next(x for x in fs.read(E/'inputs/cases.json') if x['slug']==slug);folder=OUT/'results'/slug
    seal=fs.read(OUT/'SELECTION_SEAL.json')
    if seal['selected'][slug]!=fs.sha(folder/'selected.json'):raise ValueError('Seal mismatch')
    chosen=fs.read(folder/'selected.json');prefix=E/'inputs/prefix'/f'{slug}.bin';future=E/'inputs/future'/f'{slug}.bin'
    if fs.sha(future)!=c['future_hash']:raise ValueError('Future hash mismatch')
    raw=np.fromfile(future,dtype=DTYPE);t,h=c['T'],c['H']
    if len(raw)!=t+c['B']+h or raw[:t+1].tobytes()!=prefix.read_bytes():raise ValueError('Prefix mismatch')
    origins=np.arange(t,t+c['B']);cpu=raw['raw'][origins[:,None]+np.arange(1,h+1),3]
    valid=np.all(np.isfinite(cpu)&(cpu>=0),axis=1);y=(cpu[valid]/100).astype(np.float32)
    features=causal_features(raw['timestamp'],raw['raw'],fs.read(E/'inputs/normalization.json'))
    x=torch.from_numpy(np.stack([features[o-95:o+1] for o in origins]).astype(np.float32)).to(u.DEVICE);rows=[];panels={}
    for tier in ('medium','tight','loose'):
        path=folder/tier;action=fs.read(path/'action.json')
        if fs.sha(path/'action.json')!=chosen['actions'][tier] or fs.sha(path/'model.pt')!=action['model_sha256'] or fs.sha(path/'validation.npz')!=action['validation_sha256']:raise ValueError('Action hash mismatch')
        model=u.build(action['arch'],h,torch.load(path/'model.pt',map_location='cpu',weights_only=True),c['seed']);model.eval()
        with u.candidate_backend_context(u.ARCH[action['arch']],u.DEVICE,'gru-native'),torch.no_grad():pred=model(x).cpu().numpy()[valid]
        panels[tier]=pred;rows.append({'tier':tier,'arch':action['arch'],'n_valid':len(y),**fs.metrics(pred,y,np)})
    fs.npz(folder/'forecasts.npz',panels,np)
    fs.save(folder/'scores.json',{'slug':slug,'VM':c['VM'],'H':h,'K':c['K'],'method':'h_meta_nas','rows':rows,
         'future_sha256':c['future_hash'],'selected_sha256':fs.sha(folder/'selected.json'),
         'forecast_sha256':fs.sha(folder/'forecasts.npz'),'seal_sha256':fs.sha(OUT/'SELECTION_SEAL.json')})

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'supervisor.lock').open('a+') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);phase='wait_source';queue=[];active={};done=[];reason=''
        def status(state):fs.save(OUT/'status.json',{'state':state,'phase':phase,'reason':reason,'completed':done,'active':list(active),'pending':len(queue),'updated_unix':time.time(),'disk_free_bytes':shutil.disk_usage(ROOT).free})
        def stop(sig,frame):raise RuntimeError('Supervisor interrupted')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        def runphase(mode,cases):
            nonlocal phase,queue,done
            phase=mode;queue=[];done=[]
            for c in cases:
                slug=c['slug'];p=OUT/'results'/slug/('selected.json' if mode=='select' else 'scores.json')
                (done if p.exists() else queue).append(slug)
            while queue or active:
                if shutil.disk_usage(ROOT).free<2.5*1024**3:raise RuntimeError('disk guard')
                mem=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
                if mem<16*1024**3:raise RuntimeError('memory guard')
                uuid=subprocess.check_output(['nvidia-smi','-i',str(GPU),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
                lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines();own={x['p'].pid for x in active.values()}
                if any(u.strip()==uuid and p.strip().isdigit() and int(p.strip()) not in own for u,p in (line.split(',') for line in lines)):raise RuntimeError('GPU conflict')
                for slug,x in list(active.items()):
                    rc=x['p'].poll()
                    if rc is None:continue
                    x['log'].close();del active[slug]
                    if rc:raise RuntimeError('Worker failed '+mode+' '+slug)
                    done.append(slug)
                while queue and len(active)<12:
                    slug=queue.pop(0);(OUT/'logs').mkdir(exist_ok=True);log=(OUT/'logs'/f'{mode}_{slug}.log').open('ab',buffering=0)
                    proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker',mode,slug],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(GPU)));active[slug]={'p':proc,'log':log}
                status('running');time.sleep(5)
        try:
            while True:
                if shutil.disk_usage(ROOT).free<2.5*1024**3:raise RuntimeError('disk guard')
                source=SOURCE/'status.json'
                if source.exists() and fs.read(source)['state']=='paused':raise RuntimeError('H-meta source paused')
                ready=source.exists() and fs.read(source)['state']=='completed'
                busy=subprocess.check_output(['nvidia-smi','-i',str(GPU),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                if ready and not busy:break
                status('waiting');time.sleep(20)
            lock=OUT/'run_lock.json';doc=source_lock()
            if lock.exists() and fs.read(lock)!=doc:raise ValueError('H-meta target identity changed')
            if not lock.exists():fs.save(lock,doc)
            cases=fs.read(E/'inputs/cases.json')
            if not (OUT/'smoke/SMOKE_PASS.json').exists():
                phase='smoke';(OUT/'smoke').mkdir(exist_ok=True)
                with (OUT/'smoke_worker.log').open('ab',buffering=0) as log:
                    p=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--smoke-worker',cases[0]['slug']],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(GPU)))
                    while p.poll() is None:status('running');time.sleep(10)
                    if p.returncode:raise RuntimeError('Target smoke failed')
            if not (OUT/'stage_start.json').exists():fs.save(OUT/'stage_start.json',{'start_unix':time.time(),'run_lock_sha256':fs.sha(lock)})
            runphase('select',cases);selected={}
            for c in cases:
                folder=OUT/'results'/c['slug'];a=fs.read(folder/'selected.json')
                if a['run_lock_sha256']!=fs.sha(lock):raise ValueError('Selection lock mismatch')
                for tier in ('medium','tight','loose'):
                    p=folder/tier;v=fs.read(p/'action.json')
                    if a['actions'][tier]!=fs.sha(p/'action.json') or v['model_sha256']!=fs.sha(p/'model.pt') or v['validation_sha256']!=fs.sha(p/'validation.npz'):raise ValueError('Selection output mismatch')
                selected[c['slug']]=fs.sha(folder/'selected.json')
            seal=OUT/'SELECTION_SEAL.json'
            if seal.exists() and fs.read(seal)['selected']!=selected:raise ValueError('Seal changed')
            if not seal.exists():fs.save(seal,{'selected':selected,'run_lock_sha256':fs.sha(lock),'selected_before_future_scoring':True,'decisions':360,'end_unix':time.time()})
            runphase('score',cases);rows=[]
            for c in cases:
                f=OUT/'results'/c['slug'];r=fs.read(f/'scores.json')
                if r['selected_sha256']!=fs.sha(f/'selected.json') or r['forecast_sha256']!=fs.sha(f/'forecasts.npz') or r['seal_sha256']!=fs.sha(seal):raise ValueError('Score hash mismatch')
                rows.append(r)
            fs.save(OUT/'all_scores.json',{'status':'HASH_VERIFIED','tasks':120,'rows':rows,'seal_sha256':fs.sha(seal)})
            fs.save(OUT/'COMPLETE.json',{'tasks':120,'scores_sha256':fs.sha(OUT/'all_scores.json'),'end_unix':time.time(),'requires_independent_numerical_recheck':True})
            phase='complete';status('completed')
        except Exception as e:reason=str(e);status('paused');print('PAUSED',reason,flush=True)
        finally:
            for x in active.values():
                if x['p'].poll() is None:x['p'].terminate()
            for x in active.values():
                try:x['p'].wait(timeout=15)
                except subprocess.TimeoutExpired:x['p'].kill();x['p'].wait()
                x['log'].close()

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        if sys.argv[2]=='select':select(sys.argv[3])
        elif sys.argv[2]=='score':score(sys.argv[3])
        else:raise ValueError('Bad worker mode')
    elif len(sys.argv)>1 and sys.argv[1]=='--smoke-worker':
        select(sys.argv[2],True);f=OUT/'smoke/results'/sys.argv[2];fs.save(OUT/'smoke/SMOKE_PASS.json',{'slug':sys.argv[2],'selected_sha256':fs.sha(f/'selected.json'),'scope':'Smoke only'})
    else:main()
