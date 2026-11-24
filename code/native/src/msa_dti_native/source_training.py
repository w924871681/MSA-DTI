from .runtime import *
import copy,gc,time,importlib.util

def save_torch(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,p)

def load_state(p):
    v=torch.load(p,map_location='cpu',weights_only=False);return v.get('state_dict',v)

def cpu_state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}

def train_c1(j):
    H=j['H'];bank={};out=RUN/'source_assets/c1'/f'H{H}';out.mkdir(parents=True,exist_ok=True)
    with np.load(REAL/'bitbrains/data/bitbrains_dataset.npz') as d:
        x=torch.from_numpy(d[f'X_c1_H{H}'].copy());y=torch.from_numpy(d[f'y_c1_H{H}'].copy())
    for idx in range(len(ARCH)):
        p=out/f'A{idx}.pt';s=2904+10000*H+idx
        if not p.exists():
            m=build(idx,H,s=s)
            with candidate_backend_context(ARCH[idx],DEVICE,'gru-native'):
                opt=torch.optim.Adam(m.parameters(),lr=.001);g=torch.Generator().manual_seed(s+100000)
                batches=[]
                for ep in range(10):
                    order=torch.randperm(len(x),generator=g)
                    batches.extend([order[i:i+64] for i in range(0,len(x),64)])
                for ii,ids in enumerate(batches):step(m,opt,x[ids],y[ids],clip=5)
                opt=torch.optim.Adam(m.parameters(),lr=.001);g=torch.Generator().manual_seed(s+200000)
                for it in range(180):
                    ids=torch.randint(len(x),((10,20)[it%2],),generator=g);step(m,opt,x[ids],y[ids],huber=True,clip=5)
            save_torch(p,{'state_dict':cpu_state(m),'source_only':True,'steps':220,'seed':s});del m,opt
        bank[str(idx)]={'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)}
        progress(j,idx+1,len(ARCH),f'H{H} A{idx}');print('C1',H,idx+1,flush=True)
    p=RUN/'source_assets'/f'c1_H{H}.json';write(p,dict(states=bank,source_only=True,source_input_sha256=sha(RUN/'BITBRAINS_INPUT_LOCK.json')));finish(j,[p])

def step(m,opt,x,y,huber=False,clip=None):
    m.train();opt.zero_grad(set_to_none=True);p=m(x.to(DEVICE).contiguous());yy=y.to(DEVICE).contiguous()
    loss=torch.nn.functional.huber_loss(p,yy,delta=.2) if huber else ((p-yy)**2).mean()
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite source loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),float('inf') if clip is None else clip,error_if_nonfinite=True);opt.step()

def teacher_functions():
    # Frozen native pure functions, without loading old cfg/data/runtime paths.
    from . import teacher as bb_teacher_native
    return bb_teacher_native

def teacher(j):
    t=teacher_functions();episodes=[];banks={H:read(RUN/'source_assets'/f'c1_H{H}.json')['states'] for H in [1,4]}
    for H in [1,4]:
        for K in [10,20]:
            for cid in range(20):
                p=RUN/'source_assets/teacher'/f's{cid}_h{H}_k{K}.json'
                if p.exists():episodes.append(read(p));continue
                xs,ys,xq,yq=source_task(cid,H,K);s=2904+1009*cid+37*H+53*K
                ids=t._balanced_teacher_subset(feasible_indices=list(range(len(ARCH))),A=ARCH,n_arch=24,seed=s);losses=[]
                for idx in ids:
                    a=banks[H][str(idx)];assert sha(ROOT/a['path'])==a['sha256'];m=build(idx,H,load_state(ROOT/a['path']),s+idx)
                    with candidate_backend_context(ARCH[idx],DEVICE,'gru-native'):update(m,xs,ys,50,s+idx,search=True)
                    losses.append(metrics(pred(m,xq,idx),yq.numpy())['weighted_mse']);del m
                ep=dict(center_id=cid,H=H,K=K,candidate_indices=ids,validation_mse=losses);write(p,ep);episodes.append(ep)
                progress(j,len(episodes),80,f'Source {cid} H{H} K{K}');print('Teacher',len(episodes),flush=True)
    info=t._compute_pi50(episodes,len(ARCH));assert min(info['coverage'])>0
    p=RUN/'source_assets/pi50.json';write(p,dict(episodes=episodes,**info,source_only=True,source_tiers='none; family-balanced 24 of all 66; target hard constraints applied at deployment'));finish(j,[p])

def medet(j):
    H=j['H'];idx=57;s=2904+211*H+idx;seed(s)
    m=build(idx,H,s=s);opt=torch.optim.Adam(m.parameters(),lr=.001);rng=random.Random(s)
    for it in range(400):
        tasks=rng.sample(range(20),8);opt.zero_grad(set_to_none=True)
        grads=[torch.zeros_like(v) for v in m.parameters()]
        with candidate_backend_context(ARCH[idx],DEVICE,'gru-native'):
            for cid in tasks:
                xs,ys,xq,yq=source_task(cid,H,(10,20)[(it+cid)%2]);seed(2904+1009*cid+37*H+it);fast=copy.deepcopy(m);inner=torch.optim.SGD(fast.parameters(),lr=.01)
                for k in range(5):step(fast,inner,xs,ys,clip=1)
                loss=((fast(xq.to(DEVICE))-yq.to(DEVICE))**2).mean()
                if not torch.isfinite(loss):raise RuntimeError('nonfinite meta query')
                gg=torch.autograd.grad(loss,tuple(fast.parameters()))
                for a,b in zip(grads,gg):a.add_(b.detach()/8)
                del fast,inner
        for a,b in zip(m.parameters(),grads):a.grad=b
        torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
        if (it+1)%10==0:
            save_torch(RUN/'source_assets/medet'/f'H{H}_partial.pt',{'state_dict':cpu_state(m),'iterations':it+1,'optimizer':opt.state_dict()});progress(j,it+1,400,f'H{H} A{idx}');print('MeDeT',H,it+1,flush=True)
    p=RUN/'source_assets/medet'/f'H{H}.pt';save_torch(p,{'state_dict':cpu_state(m),'arch_index':idx,'iterations':400,'source_only':True});finish(j,[p])

def hmeta(j):
    from . import meta_training as h
    # Source-only task loader is defined in meta_training.
    p=RUN/'source_assets/hmeta.pt';p.parent.mkdir(parents=True,exist_ok=True)
    h._build_meta_bank(root=RUN,cfg=None,cache=None,A=ARCH,input_dim=25,L=96,device=DEVICE,artifact=p,smoke=False)
    finish(j,[p])

def strong_variant(j):
    H=j['H'];count=j['source_count'];base=j['source_seed'];manifest=[]
    with np.load(REAL/'bitbrains/data/bitbrains_dataset.npz') as d:
        x=[torch.from_numpy(d[f'X_source_{o}_H{H}'].copy()) for o in range(count)]
        y=[torch.from_numpy(d[f'y_source_{o}_H{H}'].copy()) for o in range(count)]
    for a in [a for a in ASSETS if a['H']==H]:
        idx=a['arch_index'];token=a['token'];s=a['initialization_seed']+(base-2904);p=RUN/'source_assets/variants'/j['variant']/f'{token}_H{H}.pt'
        if not p.exists():
            m=build(idx,H,s=s)
            with candidate_backend_context(ARCH[idx],DEVICE,'gru-native'):
                if token=='C1_A57_pair':
                    with np.load(REAL/'bitbrains/data/bitbrains_dataset.npz') as d:
                        xp=torch.from_numpy(d[f'X_c1_H{H}'][:count*10].copy());yp=torch.from_numpy(d[f'y_c1_H{H}'][:count*10].copy())
                    opt=torch.optim.Adam(m.parameters(),lr=.001);g=torch.Generator().manual_seed(s+100000)
                    # Equal 40 pretrain updates; balanced repeated epoch traversal.
                    done=0
                    while done<40:
                        order=torch.randperm(len(xp),generator=g)
                        for left in range(0,len(xp),64):
                            if done==40:break
                            ids=order[left:left+64];step(m,opt,xp[ids],yp[ids],clip=5);done+=1
                    opt=torch.optim.Adam(m.parameters(),lr=.001);g=torch.Generator().manual_seed(s+200000)
                    for it in range(180):
                        ids=torch.randint(len(xp),((10,20)[it%2],),generator=g);step(m,opt,xp[ids],yp[ids],huber=True,clip=5)
                else:
                    opt=torch.optim.Adam(m.parameters(),lr=.001);done=0;epoch=0
                    while done<2000:
                        order=list(range(count));random.Random(base+10007*H+epoch if token=='PT_A57_ref' else s+epoch).shuffle(order)
                        for cid in order:
                            shuffle_seed=base+101*H+1009*cid+epoch if token=='PT_A57_ref' else s+epoch+int(hashlib.sha256(DS['source_machine_ids'][cid].encode()).hexdigest()[:8],16)
                            g=torch.Generator().manual_seed(shuffle_seed);ix=torch.randperm(100,generator=g)
                            for left in [0,64]:
                                if done==2000:break
                                ids=ix[left:left+64];step(m,opt,x[cid][ids],y[cid][ids]);done+=1
                        epoch+=1
            save_torch(p,{'state_dict':cpu_state(m),'source_count':count,'source_seed':base});del m,opt
        manifest.append(dict(a,path=p.relative_to(ROOT).as_posix(),sha256=sha(p),variant=j['variant'],training_recipe='pooled_mse_40_then_huber_180' if token=='C1_A57_pair' else 'source_mse_2000_updates'))
        progress(j,len(manifest),7,token)
    p=RUN/'source_assets/variants'/j['variant']/f'H{H}.json';write(p,{'checkpoints':manifest,'source_count':count,'source_seed':base,'budget':'2000 strong updates / C1 40+180'});finish(j,[p])

if __name__=='__main__':
    j=read(sys.argv[1]);{'c1':train_c1,'teacher':teacher,'medet':medet,'hmeta':hmeta,'strong':strong_variant}[j['action']](j)
