"""Regenerate Figures 5--7 from unchanged plot-ready data."""
import argparse
from pathlib import Path
from collections import Counter
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
from unified_style import *

def overall(out):
    r=read('overall_new.csv')
    expected=['PT-FT','MeDeT-style','Scratch50','Meta+NAS-lite','Zero-shot NAS',
      'Zero-shot NAS+FT','H-Meta-NAS-based','MSA-DTI']
    assert list(r.Method)==expected
    names=['PT-FT','MeDeT-style','Scratch50','Meta+NAS-lite','Zero-shot NAS',
      'Zero-shot NAS+FT','H-Meta-NAS-based','MSA-DTI (ours)']
    colors=['#6e8fb3','#8db8b0','#c6c2bd','#d9a25f','#c77979','#b39bbc','#8f99a6','#4f7f5c']
    fig=canvas(316); panels=[]
    specs=[('MAE','MAE',None,None),('MSE','MSE',None,None),
      ('Worst10','Worst-10% error',None,None),('CVaR90','CVaR90',None,None)]
    sem={}
    for i,(metric,title,xmax,ticks) in enumerate(specs):
        ax=axis(fig,94 if i%2==0 else 325,197 if i<2 else 47,133,108)
        vals=r[metric].to_numpy()*100;sem[metric]=vals.tolist();xmax=float(vals.max()*1.22)
        bars=ax.barh(np.arange(8),vals,height=.64,color=colors,edgecolor='#555555',lw=.5)
        bars[-1].set_hatch('//');ax.set_yticks(np.arange(8),names)
        ax.invert_yaxis();ax.set_xlim(0,xmax)
        for t in ax.get_yticklabels():
            if t.get_text()=='MSA-DTI (ours)':t.set_fontweight('bold')
        for j,v in enumerate(vals):
            ax.text(v+.025*xmax,j,f'{v:.3f}',va='center',ha='left',
              weight='bold' if j==7 else 'normal')
        ax.set_xlabel(r'Error ($\times10^{-2}$)',labelpad=3)
        clean(ax,'x'); panels.append((ax, f'({chr(97+i)}) {title}'))
    panel_row(fig, panels[:2]); panel_row(fig, panels[2:])
    save(fig,out,'fig5',['overall_new.csv'],sem)

def paired(out):
    r=read('paired_new.csv');assert len(r)==120
    cats=[('beneficial alternative','Beneficial alternative',BLUE,'o'),
      ('reference retained','Reference retained',GREY,'s'),
      ('harmful alternative','Harmful alternative',ORANGE,'X')]
    counts=Counter(r.selection_category);assert sum(counts.values())==120
    fig=canvas(225);axes=[];panels=[]
    for i,(xcol,ycol,title) in enumerate([('pt_ft_mse','proposed_mse','MSE'),
      ('pt_ft_worst10','proposed_worst10','Worst-10% error')]):
        ax=axis(fig,52 if i==0 else 288,48,151,151);axes.append(ax)
        values=r[[xcol,ycol]].to_numpy()
        if np.any(values<=0):raise ValueError('Log comparison requires positive errors')
        lower=10**np.floor(np.log10(values.min()));upper=10**np.ceil(np.log10(values.max()))
        ax.plot([lower,upper],[lower,upper],color='#222222',lw=.7,ls='--',zorder=1)
        for key,name,col,mark in cats:
            a=r[r.selection_category==key]
            ax.scatter(a[xcol],a[ycol],s=19 if mark!='X' else 26,
              marker=mark,facecolors='white' if mark!='X' else col,
              edgecolors=col,linewidths=.8,label=f'{name} ({counts[key]})',zorder=2)
        ax.set_xscale('log');ax.set_yscale('log')
        ax.set(xlim=(lower,upper),ylim=(lower,upper),aspect='equal',
          xlabel='PT-FT (log scale)',ylabel='MSA-DTI (log scale)')
        clean(ax,'both');panels.append((ax,f'({chr(97+i)}) {title}'))
    panel_row(fig,panels)
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,1.005),ncol=3,
      frameon=False,handlelength=1,columnspacing=1.0,handletextpad=.4)
    save(fig,out,'fig6',['paired_new.csv'],
      {'cases':120,'categories':dict(counts),'columns':list(r.columns),'scale':'logarithmic, raw error'})

def forest(out):
    r=read('subgroups.csv');fig=canvas(186);panels=[]
    for i,(panel,col,mark) in enumerate([('H/K',BLUE,'o'),('Deployment budget',GREEN,'s')]):
        a=r[r.panel==panel];y=np.arange(len(a))[::-1]
        m=a.gain_percent.to_numpy();lo=a.ci_low_percent.to_numpy();hi=a.ci_high_percent.to_numpy()
        assert np.all(lo>=0) and np.all(hi<=70)
        ax=axis(fig,92 if i==0 else 328,47,143 if i==0 else 129,130)
        ax.errorbar(m,y,xerr=np.array([m-lo,hi-m]),fmt=mark,color=col,capsize=3,markersize=4.5)
        for yy,v in zip(y,m):ax.annotate(f'{v:.1f}',(v,yy),xytext=(3,4),textcoords='offset points',color=col)
        labels=[v+f' (n={30 if i==0 else 120})' for v in a.group]
        ax.set_yticks(y,labels);ax.set_ylim(-.22,len(a)-.57);ax.set_xlim(0,70)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.set_xlabel('MSE reduction relative to PT-FT (%)',labelpad=4)
        clean(ax,'x');panels.append((ax,'(a) Horizon and support size' if i==0 else '(b) Deployment budget'))
    panel_row(fig,panels)
    save(fig,out,'fig7',['subgroups.csv'],r.to_dict(orient='records'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'figures');a=p.parse_args()
    for fn in [overall,paired,forest]:fn(a.output)
    print('Generated Figures 5--7 from unchanged CSV data:',a.output)
if __name__=='__main__':main()
