"""Fresh candidate adaptation using only observations available at selection."""
import gc,contextlib,time,shutil
from .runtime import *
from .prefix import prefix_selection
from .raw_data import array_sha256
from .source_training import save_torch,cpu_state

Q7=['A1','A13','A55','A56','A6','C1_A57_pair','PT_A57_ref']
ARCH_INDEX={'A1':1,'A13':13,'A55':55,'A56':56,'A6':6,'C1_A57_pair':57,'PT_A57_ref':57}

def tensorhash(st):return digest({k:array_sha256(v.detach().cpu().numpy()) for k,v in st.items()})

class PrefixFence:
    """Python-level guard against future, raw-archive and expected-result reads."""
    def __init__(self):self.active=False;self.denied=[];self.reads=0
    def hook(self,event,args):
        if not self.active or event!='open' or isinstance(args[0],int):return
        path=Path(os.fsdecode(args[0])).resolve()
        if any(path.is_relative_to(p) for p in [DATA/'future',DATA/'raw',ROOT/'expected']):
            self.denied.append(str(path.relative_to(ROOT)));raise PermissionError('Prediction-prefix access violation: '+str(path))
        self.reads+=1
    def __enter__(self):sys.addaudithook(self.hook);self.active=True;return self
    def __exit__(self,*args):self.active=False

def cases(role):
    return [c for c in read(ROOT/('protocol/development_prefix_tasks.json' if role=='development' else 'protocol/target_tasks.json'))['cases'] if c['role']==role]

def source(variant,q,h):
    if (RUN/'bank_definitions'/(variant+'.json')).exists():p=ROOT/read(RUN/'bank_definitions'/(variant+'.json'))['checkpoints'][q][str(h)]
    elif variant=='alternative_source' and q!='A55':p=RUN/'source_assets/c1'/f'H{h}'/f'A{ARCH_INDEX[q]}.pt'
    else:p=RUN/'source_assets/variants'/('source20_seed2904' if variant=='alternative_source' else variant)/f'{q}_H{h}.pt'
    obj=torch.load(p,map_location='cpu',weights_only=True)
    return obj.get('state_dict',obj),p

def job_cases(j):
    tasklist=cases(j['role'])
    if 'case_slugs' in j:
        requested=j['case_slugs']
        assert len(requested)==len(set(requested)), 'Duplicate benchmark tasks'
        known={c['slug']:c for c in tasklist}
        assert all(s in known for s in requested), 'Task must belong to the declared data role'
        tasklist=[known[s] for s in requested]
    return tasklist[j.get('worker',0)::j.get('workers',1)]

def adapt(j):
    variant=j['variant'];tasklist=job_cases(j)
    if j.get('limit'):tasklist=tasklist[:j['limit']]
    definition=read(RUN/'bank_definitions'/(variant+'.json')) if (RUN/'bank_definitions'/(variant+'.json')).exists() else None
    qaxis=definition['qaxis'] if definition else Q7
    indices=definition['arch_indices'] if definition else ARCH_INDEX
    norm=read(DATA/'bitbrains/data/source_normalization.json');states={(q,h):source(variant,q,h) for q in qaxis for h in [1,4]}
    for ci,c in enumerate(tasklist):
        folder=RUN/'banks'/variant/c['slug'];folder.mkdir(parents=True,exist_ok=True);receipt=folder/'adaptation.json'
        if receipt.exists():
            old=read(receipt)
            assert all(sha(folder/k)==v for k,v in old['files'].items()), 'Adaptation output changed'
            continue
        prefix=DATA/'prefix'/c['slug']/'prefix.bin';rows=[];fence=PrefixFence();reused=0
        with fence:
            d=prefix_selection(c,prefix,norm)
            xs=torch.from_numpy(d['Xs']).to(DEVICE).contiguous();ys=torch.from_numpy(d['ys']).to(DEVICE).contiguous()
            for q in qaxis:
                idx=indices[q];st,sp=states[q,c['H']]
                steps=[0,1,5,10,20,50] if variant=='source20_seed2904' and c['role']!='development' else [50]
                cache_key=digest(dict(prefix=sha(prefix),source=tensorhash(st),spec=ARCH[idx].to_dict(),seed=c['seed'],code=sha(Path(__file__)),steps=50))
                cache=RUN/'cache/adaptation_catalog'/c['slug']/(cache_key+'.json')
                if steps==[50] and cache.exists():
                    hit=read(cache);origin=ROOT/hit['folder'];assert all(sha(origin/k)==v for k,v in hit['files'].items())
                    row=dict(hit['row'],token=q,source_sha256=sha(sp),state_file=q+'.step50.pt')
                    for suffix in ['.pt','.validation.npz']:
                        src=origin/(hit['row']['token']+'.step50'+suffix);dst=folder/(q+'.step50'+suffix)
                        if src.resolve()!=dst.resolve():shutil.copyfile(src,dst)
                    rows.append(row);reused+=1;continue
                model=build(idx,c['H'],st,c['seed'])
                native.seed_all(c['seed'],DEVICE);opt=torch.optim.SGD(model.parameters(),lr=.01)
                pa,op=profile_arch(ARCH[idx],96,25,c['H'])
                with torch.backends.cudnn.flags(enabled=False):
                    for it in range(51):
                        if it in steps:
                            pv=pred(model,torch.from_numpy(d['Xv']),idx).astype(np.float64)
                            loss=np.square(pv-d['yv'].astype(np.float64)).mean(1)
                            assert np.isfinite(loss).all() and loss.mean()>0
                            state=cpu_state(model);name=f'{q}.step{it}'
                            save_torch(folder/(name+'.pt'),state)
                            np.savez_compressed(folder/(name+'.validation.npz'),predictions=pv,targets=d['yv'],origins=d['origins'],losses=loss)
                            rows.append(dict(token=q,arch_index=idx,spec=ARCH[idx].to_dict(),params=pa,operations=op,steps=it,
                                             tensor_sha256=tensorhash(state),source_sha256=sha(sp),state_file=name+'.pt',
                                             validation_mean=float(loss.mean()),n_validation=len(loss)))
                        if it<50:
                            model.train();opt.zero_grad(set_to_none=True);loss=((model(xs)-ys)**2).mean()
                            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                del model,opt;gc.collect()
                row=next(r for r in rows if r['token']==q and r['steps']==50)
                write(cache,dict(folder=folder.relative_to(ROOT).as_posix(),row=row,files={q+'.step50'+s:sha(folder/(q+'.step50'+s)) for s in ['.pt','.validation.npz']}))
            write(folder/'candidates.json',dict(case=c,variant=variant,qaxis=qaxis,rows=rows,prefix_sha256=sha(prefix),future_access=False))
        expected=read(ROOT/'expected/target_adaptations.json').get(variant+'/'+c['slug'])
        checks=[]
        if expected:
            for row in [r for r in rows if r['steps']==50]:
                e=expected[row['token']]
                checks.append(dict(token=row['token'],weights_exact=row['tensor_sha256']==e['tensor'],validation_exact=row['validation_mean']==e['validation']))
            assert all(r['weights_exact'] and r['validation_exact'] for r in checks), (variant,c['slug'],checks)
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.suffix in ['.pt','.npz','.json']}
        write(receipt,dict(status='PASS',case_id=c['case_id'],files=files,checks=checks,expected_record_available=bool(expected),gradient_updates=50*len(qaxis),actual_gradient_updates=50*(len(qaxis)-reused),identical_fresh_adaptations_reused=reused,source_checkpoints_retrained=True,access=dict(python_reads=fence.reads,denied=fence.denied),utc=utc()))
        progress(j,ci+1,len(tasklist),c['slug'])
    finish(j,[RUN/'banks'/variant/c['slug']/'adaptation.json' for c in tasklist])

