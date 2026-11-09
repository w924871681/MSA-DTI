"""Regenerate every scientific input from the original raw workload archive."""
import hashlib,zipfile
import numpy as np
from .io import *
from . import raw_data as d
from .features import raw_role_bounds
from .prefix import DTYPE,prefix_selection

def input_lock(archive):
    tensor_receipt=RUN/'verification/raw_data.json'
    tasks_receipt=RUN/'SOURCE_TASKS_LOCK.json'
    return dict(raw_archive='data/raw/fastStorage.zip',archive_sha256=sha(archive),
                prepared_source='data/bitbrains/data/bitbrains_dataset.npz',
                source_tensor_receipt=tensor_receipt.relative_to(ROOT).as_posix(),
                source_tensor_receipt_sha256=sha(tensor_receipt),
                source_tasks_receipt=tasks_receipt.relative_to(ROOT).as_posix(),
                source_tasks_sha256=sha(tasks_receipt))

def parse_until(archive,case,last):
    rows=[]
    with archive.open(case['raw_member']) as stream:
        assert len(stream.readline().decode('utf-8-sig').strip().split(';'))==11
        for i in range(case['native_start']+last+1):
            line=stream.readline()
            if not line:raise ValueError('Early raw-file EOF')
            if i<case['native_start']:continue
            fields=line.decode('utf-8').strip().split(';');assert len(fields)==11
            timestamp=float(fields[0]);assert np.isfinite(timestamp) and timestamp==int(timestamp)
            values=[]
            for value in fields[1:]:
                try:values.append(float(value))
                except ValueError:values.append(float('nan'))
            rows.append((int(timestamp),values))
    array=np.array(rows,dtype=DTYPE);assert len(array)==last+1
    return array.tobytes()

def development_payloads(case,series_path):
    # Development inputs historically used the pandas-parsed target series.
    # Re-parsing decimal strings with Python float changes some binary64 bits.
    count=case['T']+case['B']+case['H']
    with np.load(series_path,allow_pickle=False) as series:
        assert len(series['timestamps'])>=count
        rows=np.empty(count,dtype=DTYPE)
        rows['timestamp']=series['timestamps'][:count]
        rows['raw']=series['raw_metrics'][:count]
    payload=rows.tobytes()
    return payload[:(case['T']+1)*DTYPE.itemsize],payload


def main():
    contract=read(ROOT/'protocol/source_development.json')
    archive=DATA/'raw/fastStorage.zip'
    assert sha(archive)==contract['archive_sha256']
    source=[m for m in contract['machines'] if m['group']=='source']
    prepared=DATA/'bitbrains/data';prepared.mkdir(parents=True,exist_ok=True)
    assert not (RUN/'verification/raw_data.json').exists(), 'Raw preparation already completed; use a fresh output directory for a new run.'
    with zipfile.ZipFile(archive) as z:
        series=[]
        for m in source:
            t,raw=d.read_member(z,m);keep=t<t[0]+86400;series.append((t[keep],raw[keep]))
        norm=d.fit_source_normalization(series)
        expected=read(ROOT/'expected/source_normalization.json')
        assert all(norm[k]==expected[k] for k in norm), 'Source normalization mismatch'
        norm.update(source_machine_ids=[m['vm_id'] for m in source],archive_sha256=contract['archive_sha256'])
        write(prepared/'source_normalization.json',norm)
        arrays={};pools={h:([],[]) for h in [1,4]};source_tasks=[]
        taskfolder=RUN/'source_tasks';taskfolder.mkdir(parents=True,exist_ok=True)
        for cid,(m,(t,raw)) in enumerate(zip(source,series)):
            for h in [1,4]:
                eligible=d.complete_origins(raw,h);chosen=eligible[np.linspace(0,len(eligible)-1,100).astype(int)]
                x,y=d.windows(t,raw,norm,h,chosen)
                arrays[f'X_source_{cid}_H{h}']=x;arrays[f'y_source_{cid}_H{h}']=y
                x,y=d.windows(t,raw,norm,h,eligible[:10]);pools[h][0].append(x);pools[h][1].append(y)
                for k in [10,20]:
                    cut=96+h+k-1;sup=d.complete_origins(raw[:cut],h);query=d.complete_origins(raw[cut:],h)
                    assert len(sup)>=k and len(query)>0
                    xs,ys=d.windows(t[:cut],raw[:cut],norm,h,sup[:k]);xq,yq=d.windows(t[cut:],raw[cut:],norm,h,query)
                    p=taskfolder/f's{cid}_h{h}_k{k}.npz';np.savez_compressed(p,Xs=xs,ys=ys,Xq=xq,yq=yq)
                    source_tasks.append(dict(ordinal=cid,VM=m['vm_id'],H=h,K=k,path=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
            print('Raw source VM',cid+1,'/20',flush=True)
        for h,(xs,ys) in pools.items():arrays[f'X_c1_H{h}']=np.concatenate(xs);arrays[f'y_c1_H{h}']=np.concatenate(ys)
        actual={key:d.array_sha256(a) for key,a in arrays.items()}
        assert actual==read(ROOT/'expected/source_data.json')['tensor_hashes'], 'Source tensor mismatch'
        np.savez_compressed(prepared/'bitbrains_dataset.npz',**arrays)
        write(RUN/'SOURCE_TASKS_LOCK.json',dict(records=source_tasks))
        dev=read(ROOT/'protocol/development_tasks.json')['cases'];manifest=[];series_paths={}
        for m in contract['machines'][20:]:
            t,raw=d.read_member(z,m);start=int(np.searchsorted(t,t[0]+86400,side='left'))
            assert start==m['target_start_record_index']
            p=prepared/'target_series'/(m['vm_id_sha256']+'.npz');p.parent.mkdir(exist_ok=True)
            np.savez_compressed(p,timestamps=t[start:start+3000],raw_metrics=raw[start:start+3000])
            series_paths[m['vm_id']]=p
            for c in [c for c in dev if c['machine_id']==m['vm_id']]:
                c=dict(c,target_series_sha256=sha(p));manifest.append(c)
        write(prepared/'bitbrains_dataset_manifest.json',dict(source_machine_ids=[m['vm_id'] for m in source],cases=manifest,normalization_path='bitbrains/data/source_normalization.json',normalization_sha256=sha(prepared/'source_normalization.json')))
        expected_payloads=read(ROOT/'expected/target_binary_hashes.json');receipts=[]
        for c in read(ROOT/'protocol/target_tasks.json')['cases']+read(ROOT/'protocol/development_prefix_tasks.json')['cases']:
            assert hashlib.sha256(z.read(c['raw_member'])).hexdigest()==c['member_sha256']
            if c['role']=='development':
                prefix,future=development_payloads(c,series_paths[c['VM']])
            else:
                prefix=parse_until(z,c,c['T']);future=parse_until(z,c,c['T']+c['B']+c['H']-1)
            assert future[:len(prefix)]==prefix
            p=DATA/'prefix'/c['slug']/'prefix.bin';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(prefix)
            f=DATA/'future'/c['role']/c['slug']/'future.bin';f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(future)
            assert sha(p)==expected_payloads[c['slug']]['prefix'] and sha(f)==expected_payloads[c['slug']]['future'], c['slug']
            x=prefix_selection(c,p,norm)
            receipts.append(dict(case_id=c['case_id'],prefix=sha(p),future=sha(f),support_windows=len(x['ys']),validation_windows=len(x['yv'])))
        write(RUN/'verification/raw_data.json',dict(status='PASS',source_tensors=actual,source_vms=20,development_tasks=len(manifest),source_tasks=len(source_tasks),target_tasks=len(receipts),target_binary_payloads=2*len(receipts),cases=receipts,utc=utc()))
        write(RUN/'BITBRAINS_INPUT_LOCK.json',input_lock(archive))
    print('RAW_DATA_PASS: 84 source tensors, 80 source tasks, 80 development tasks, 200 calibration/evaluation tasks.',flush=True)

if __name__=='__main__':main()
