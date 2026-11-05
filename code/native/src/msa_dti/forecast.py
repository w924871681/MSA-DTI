"""Post-selection rolling-origin forecast evaluation without parameter updates."""
from .runtime import *
from .features import causal_features
from .prefix import DTYPE
from .target import cases,tensorhash

def future(c,norm):
    rows=np.frombuffer((DATA/'future'/c['role']/c['slug']/'future.bin').read_bytes(),dtype=DTYPE)
    assert rows[:c['T']+1].tobytes()==(DATA/'prefix'/c['slug']/'prefix.bin').read_bytes()
    f=causal_features(rows['timestamp'],rows['raw'],norm);origins=np.arange(c['T'],c['T']+c['B'])
    x=np.stack([f[o-95:o+1] for o in origins]).astype(np.float32)
    ix=origins[:,None]+np.arange(1,c['H']+1)[None,:];cpu=rows['raw'][ix,3]
    mask=np.all(np.isfinite(cpu)&(cpu>=0),axis=1);y=(cpu/100).astype(np.float32)
    assert mask.any()
    return x,y,mask,origins

def evaluate(j):
    variant=j['variant'];tasklist=cases(j['role'])[j.get('worker',0)::j.get('workers',1)]
    if j.get('limit'):tasklist=tasklist[:j['limit']]
    if j['role']=='evaluation':
        assert (RUN/'policies'/f'{variant}.json').is_file(), 'Calibrate before evaluation'
        assert (RUN/'policies'/f'{variant}.actions.json').is_file(), 'Seal all evaluation actions first'
    norm=read(DATA/'bitbrains/data/source_normalization.json')
    for ci,c in enumerate(tasklist):
        folder=RUN/'banks'/variant/c['slug'];out=folder/'future';out.mkdir(parents=True,exist_ok=True)
        if (out/'metrics.json').exists():continue
        audit=read(folder/'adaptation.json');assert audit['status']=='PASS'
        assert all(sha(folder/n)==h for n,h in audit['files'].items())
        snapshot=read(folder/'candidates.json');x,y,mask,origins=future(c,norm);rows=[]
        for row in snapshot['rows']:
            st=torch.load(folder/row['state_file'],map_location='cpu',weights_only=True)
            assert tensorhash(st)==row['tensor_sha256']
            model=build(row['arch_index'],c['H'],st)
            predictions=pred(model,torch.from_numpy(x),row['arch_index']).astype(np.float64)
            assert tensorhash(model.state_dict())==row['tensor_sha256']
            err=predictions[mask]-y[mask].astype(np.float64);loss=np.square(err).mean(1)
            met=dict(MSE=float(loss.mean()),MAE=float(np.abs(err).mean()),Worst10=float(np.sort(loss)[-max(1,round(.1*len(loss))):].mean()),N_valid=int(mask.sum()))
            name=f"{row['token']}.step{row['steps']}"
            np.savez_compressed(out/(name+'.npz'),predictions=predictions,targets=y,label_mask=mask,origins=origins,valid_losses=loss)
            rows.append(dict(token=row['token'],steps=row['steps'],**met));del model
        write(out/'metrics.json',dict(case=c,rows=rows,adaptations=0,utc=utc()))
        progress(j,ci+1,len(tasklist),c['slug'])
    finish(j,[RUN/'banks'/variant/c['slug']/'future/metrics.json' for c in tasklist])
