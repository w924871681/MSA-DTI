"""Mean validation loss, feasibility filtering, and relative-margin calibration."""
import csv,math
import numpy as np
from .io import *

Q7=['A1','A13','A55','A56','A6','C1_A57_pair','PT_A57_ref']
TAUS=(0.,.05,.075,.10,.125,.15,.20)
BUDGETS={'tight':(1500000,30000),'medium':(5000000,100000),'loose':(20000000,500000)}
NESTED=['A55','A1','A6','A13','A56','C1_A57_pair','PT_A57_ref']

def cases(role):return [c for c in read(ROOT/('protocol/development_prefix_tasks.json' if role=='development' else 'protocol/target_tasks.json'))['cases'] if c['role']==role]
def balanced(rows,key):
    means=[]
    for vm in sorted({r['VM'] for r in rows}):
        rr=sorted([r for r in rows if r['VM']==vm],key=lambda r:(r['H'],r['K']))
        assert len(rr)==4 and {(r['H'],r['K']) for r in rr}=={(1,10),(1,20),(4,10),(4,20)}
        a=np.asarray([r[key] for r in rr],dtype=np.float64);assert np.isfinite(a).all()
        means.append(float(np.mean(a)))
    assert means
    return float(np.mean(np.asarray(means,dtype=np.float64)))

def choose(rows,ref,tau):
    by={r['token']:r for r in rows};assert ref in by
    admissible=[r for r in rows if r['token']==ref or r['validation_mean']<=(1-tau)*by[ref]['validation_mean']]
    return min(admissible,key=lambda r:(r['validation_mean'],r['params'],r['operations'],r['arch_index'],r['token']))['token']

def snapshots(variant,role,steps=50,tokens=None):
    result=[]
    for c in cases(role):
        p=RUN/'banks'/variant/c['slug'];r=read(p/'candidates.json')
        rr=[x for x in r['rows'] if x['steps']==steps and (tokens is None or x['token'] in tokens)]
        assert len(rr)==len(tokens or Q7)
        result.append((c,rr))
    return result

def future_metrics(variant,c,steps):
    return {r['token']:r for r in read(RUN/'banks'/variant/c['slug']/'future/metrics.json')['rows'] if r['steps']==steps}

def reference(variant):
    rows=[];ss=snapshots(variant,'development')
    for q in Q7:
        risks=[];pmax=0;omax=0;legal=True
        for c,rr in ss:
            r=next(r for r in rr if r['token']==q);m=future_metrics(variant,c,50)
            risks.append(dict(VM=c['VM'],H=c['H'],K=c['K'],loss=m[q]['MSE']))
            pmax=max(pmax,r['params']);omax=max(omax,r['operations'])
            legal &= all(r['params']<=pa and r['operations']<=op for op,pa in BUDGETS.values())
        rows.append(dict(token=q,risk=balanced(risks,'loss'),legal=bool(legal),max_params=pmax,max_operations=omax))
    q=min([r for r in rows if r['legal']],key=lambda r:(r['risk'],r['max_params'],r['max_operations'],r['token']))['token']
    write(RUN/'policies'/f'{variant}.reference.json',dict(reference=q,rows=rows))
    return q

def calibrate(variant,ref,steps=50,tokens=None):
    curves=[];ref_risk=None;ss=snapshots(variant,'calibration',steps,tokens)
    for tau in TAUS:
        rr=[]
        for c,rows in ss:
            rows=[r for r in rows if r['params']<=BUDGETS['medium'][1] and r['operations']<=BUDGETS['medium'][0]]
            q=choose(rows,ref,tau);m=future_metrics(variant,c,steps)
            rr.append(dict(VM=c['VM'],H=c['H'],K=c['K'],risk=m[q]['MSE'],ref=m[ref]['MSE'],harm=int(q!=ref and m[q]['MSE']>m[ref]['MSE']),switch=int(q!=ref)))
        ref_risk=balanced(rr,'ref')
        curves.append(dict(tau=tau,risk=balanced(rr,'risk'),harm=balanced(rr,'harm'),switch=balanced(rr,'switch')))
    eligible=[r for r in curves if r['harm']<=.05]
    best=min(eligible,key=lambda r:(r['risk'],-r['tau'])) if eligible else None
    policy=dict(policy='pure_reference',tau=None) if best is None or best['risk']>=ref_risk else dict(policy='selector',tau=best['tau'],risk=best['risk'])
    return dict(**policy,reference=ref,reference_risk=ref_risk,curves=curves,steps=steps,tokens=tokens or Q7)

def seal(variant,policies):
    actions=[]
    for name,p in policies.items():
        for c,rows in snapshots(variant,'evaluation',p['steps'],p['tokens']):
            for tier,(op,pa) in BUDGETS.items():
                rr=[r for r in rows if r['params']<=pa and r['operations']<=op]
                q=p['reference'] if p['policy']=='pure_reference' else choose(rr,p['reference'],p['tau'])
                actions.append(dict(case_id=c['case_id'],slug=c['slug'],VM=c['VM'],H=c['H'],K=c['K'],budget=tier,condition=name,selected=q,reference=p['reference'],steps=p['steps'],tau=p['tau'],policy=p['policy'],feasible=[r['token'] for r in rr]))
    write(RUN/'policies'/f'{variant}.actions.json',dict(actions=actions,evaluation_labels_loaded=False))

def fit(j):
    v=j['variant'];ref='A55' if v=='alternative_source' else reference(v)
    policies={'main':calibrate(v,ref)}
    if v=='source20_seed2904':
        policies['without_margin']=dict(policy='selector',tau=0.,reference=ref,steps=50,tokens=Q7)
        policies['reference_only']=dict(policy='pure_reference',tau=None,reference=ref,steps=50,tokens=Q7)
        policies['fixed_A57']=calibrate(v,'PT_A57_ref')
        policies['A57_only']=dict(policy='pure_reference',tau=None,reference='PT_A57_ref',steps=50,tokens=Q7)
        policies['two_A57']=calibrate(v,'PT_A57_ref',tokens=['C1_A57_pair','PT_A57_ref'])
        for size in range(1,8):policies[f'bank{size}']=calibrate(v,ref,tokens=NESTED[:size])
        for n in [0,1,5,10,20,50]:policies[f'updates{n}']=calibrate(v,ref,steps=n)
    write(RUN/'policies'/f'{v}.json',dict(variant=v,policies=policies,calibration_VMs=20,evaluation_labels_loaded=False))
    seal(v,policies)
    print('POLICY_SEALED',v,'reference',ref,'tau',policies['main']['tau'],flush=True)

def summarize(j):
    v=j['variant'];actions=read(RUN/'policies'/f'{v}.actions.json')['actions'];records=[]
    lookup={c['case_id']:c for c in cases('evaluation')}
    for a in actions:
        m=future_metrics(v,lookup[a['case_id']],a['steps']);q=a['selected'];ref=a['reference']
        records.append(dict(**a,**{k:m[q][k] for k in ['MSE','MAE','Worst10','N_valid']},E_reference=m[ref]['MSE'],harm=int(q!=ref and m[q]['MSE']>m[ref]['MSE']),benefit=int(q!=ref and m[q]['MSE']<m[ref]['MSE']),switch=int(q!=ref)))
    summaries=[]
    for condition in sorted({r['condition'] for r in records}):
        for tier in BUDGETS:
            rr=[r for r in records if r['condition']==condition and r['budget']==tier]
            assert len(rr)==120
            summaries.append(dict(condition=condition,budget=tier,**{key:balanced(rr,key) for key in ['MSE','MAE','Worst10','harm','benefit','switch']},CVaR90=float(np.sort([r['MSE'] for r in rr])[-12:].mean())))
    write(RUN/'statistics'/f'{v}.json',dict(records=records,summaries=summaries))
    print('SUMMARIZED',v,len(records),'task/budget/condition records',flush=True)
