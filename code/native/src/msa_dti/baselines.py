"""Fresh training and selection for all seven comparison methods."""
import gc,time
from .runtime import *
from .target import cases,job_cases,PrefixFence,source
from .prefix import prefix_selection
from . import baseline_runtime as br

METHODS=['pt_ft','medet_style','scratch50','meta_nas_lite','zero_nas','zero_nas_ft','h_meta_nas']
BUDGETS=['medium','tight','loose']

def adapt(j):
    br.C1={};br.C1HASH={};med={};pt={}
    for h in [1,4]:
        br.C1[h]={};br.C1HASH[h]={}
        for i in range(66):
            p=RUN/'source_assets/c1'/f'H{h}'/f'A{i}.pt'
            br.C1[h][i]=br.loadpt(p)['state_dict'];br.C1HASH[h][i]=sha(p)
        med[h]=br.loadpt(RUN/'source_assets/medet'/f'H{h}.pt')['state_dict']
        pt[h]=source('source20_seed2904','PT_A57_ref',h)[0]
    br.PI=read(RUN/'source_assets/pi50.json')['pi50'];hp=RUN/'source_assets/hmeta.pt'
    br.HM=br.loadpt(hp);br.HMHASH=sha(hp)
    norm=read(DATA/'bitbrains/data/source_normalization.json')
    tasks=job_cases(j)
    if j.get('limit'):tasks=tasks[:j['limit']]
    for ci,c in enumerate(tasks):
        folder=RUN/'baselines'/c['slug'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'receipt.json').exists():
            r=read(folder/'receipt.json');assert all(sha(folder/n)==h for n,h in r['files'].items());continue
        prefix=DATA/'prefix'/c['slug']/'prefix.bin'
        br.CURRENT=dict(case=c,prefix_sha=sha(prefix),native_hash=sha(Path(br.__file__)),cache=RUN/'cache/baseline_candidates'/c['slug'],logical_adaptations=0,actual_adaptations=0,cache_hits=0,events=[],proxies={})
        actions=[];fence=PrefixFence()
        with fence:
            d=prefix_selection(c,prefix,norm);xs,ys,xv,yv=[torch.from_numpy(d[n]) for n in ['Xs','ys','Xv','yv']]
            for method in METHODS:
                for tier in BUDGETS:
                    before=br.CURRENT['actual_adaptations'];h=c['H'];s=c['seed'];detail={}
                    if method in ['pt_ft','medet_style','scratch50']:
                        st=pt[h] if method=='pt_ft' else med[h] if method=='medet_style' else None
                        m,val,rec=br.cached_adapt(57,st,method,s,'sgd_mse_50',xs,ys,xv,yv)
                        selected=br.state(m);idx=57;detail={'validation':val,'adapt_identity':rec['identity']};del m
                    elif method=='h_meta_nas':selected,idx,detail=br.hmeta(h,tier,s,xs,ys,xv,yv)
                    else:selected,idx,detail=br.search(method,h,tier,s,xs,ys,xv,yv)
                    assert idx in native.feasible_indices(CFG,ARCH,tier,96,25,h)
                    p=folder/f'{method}_{tier}.pt';br.savept(p,selected);pa,op=profile_arch(ARCH[idx],96,25,h)
                    actions.append(dict(case=c,method=method,budget=tier,selected_arch=idx,spec=ARCH[idx].to_dict(),params=pa,operations=op,details=detail,model_file=p.name,tensor_sha256=br.statehash(selected),model_sha256=sha(p),new_adaptations=br.CURRENT['actual_adaptations']-before,future_arrays_loaded=False))
            write(folder/'actions.json',actions)
        expected=read(ROOT/'expected/baseline_actions.json');checks=[]
        for a in actions:
            e=expected[f"{c['slug']}/{a['method']}/{a['budget']}"]
            checks.append(dict(method=a['method'],budget=a['budget'],weights_exact=a['tensor_sha256']==e['tensor'],selection_exact=a['selected_arch']==e['arch']))
        assert all(r['weights_exact'] and r['selection_exact'] for r in checks), (c['slug'],checks)
        write(folder/'receipt.json',dict(status='PASS',checks=checks,files={p.name:sha(p) for p in folder.iterdir() if p.is_file()},actual_adaptations=br.CURRENT['actual_adaptations'],access=dict(python_reads=fence.reads,denied=fence.denied)))
        progress(j,ci+1,len(tasks),c['slug']);gc.collect()
    finish(j,[RUN/'baselines'/c['slug']/'receipt.json' for c in tasks])

def evaluate(j):
    from .forecast import future
    norm=read(DATA/'bitbrains/data/source_normalization.json')
    tasks=cases(j['role'])[j.get('worker',0)::j.get('workers',1)]
    if j.get('limit'):tasks=tasks[:j['limit']]
    for ci,c in enumerate(tasks):
        folder=RUN/'baselines'/c['slug'];audit=read(folder/'receipt.json');assert audit['status']=='PASS'
        assert all(sha(folder/n)==h for n,h in audit['files'].items())
        out=folder/'future';out.mkdir(exist_ok=True)
        if (out/'metrics.json').exists():continue
        x,y,mask,origins=future(c,norm);cache={};records=[]
        for a in read(folder/'actions.json'):
            key=a['tensor_sha256']
            if key not in cache:
                st=br.loadpt(folder/a['model_file']);assert br.statehash(st)==key
                m=build(a['selected_arch'],c['H'],st);p=pred(m,torch.from_numpy(x),a['selected_arch']).astype(np.float64)
                assert br.statehash(br.state(m))==key
                err=p[mask]-y[mask].astype(np.float64);loss=np.square(err).mean(1)
                cache[key]=dict(MSE=float(loss.mean()),MAE=float(np.abs(err).mean()),Worst10=float(np.sort(loss)[-max(1,round(.1*len(loss))):].mean()),N_valid=int(mask.sum()))
                np.savez_compressed(out/(key+'.npz'),predictions=p,targets=y,mask=mask,origins=origins);del m
            records.append(dict(method=a['method'],budget=a['budget'],arch=a['selected_arch'],**cache[key]))
        write(out/'metrics.json',dict(case=c,records=records,gradient_updates=0));progress(j,ci+1,len(tasks),c['slug'])
    finish(j,[RUN/'baselines'/c['slug']/'future/metrics.json' for c in tasks])
