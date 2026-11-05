"""One typography contract for main and supplementary experimental figures.

All figures have their final printed width (164 mm). Include the PDF at its
natural size in LaTeX. No statistics, predictions or bootstrap draws are changed.
"""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DATA=ROOT/'data'
WIDTH=164/25.4*72
FONT_SIZE=9.0
BLUE='#0072B2'; GREEN='#009E73'; ORANGE='#D55E00'; GREY='#666666'

def style():
    for f in (HERE/'fonts').glob('*.ttf'):font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({'font.family':'Tinos','font.size':FONT_SIZE,
      'axes.labelsize':FONT_SIZE,'axes.titlesize':FONT_SIZE,
      'xtick.labelsize':FONT_SIZE,'ytick.labelsize':FONT_SIZE,
      'legend.fontsize':FONT_SIZE,'figure.titlesize':FONT_SIZE,
      'mathtext.fontset':'custom','mathtext.rm':'Tinos',
      'mathtext.it':'Tinos:italic','mathtext.bf':'Tinos:bold',
      'mathtext.fallback':'stix','axes.linewidth':.7,'lines.linewidth':1.1,
      'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
      'svg.hashsalt':'MSA-DTI-20261101','axes.unicode_minus':False,
      'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})

def canvas(height):
    style();return plt.figure(figsize=(WIDTH/72,height/72))

def axis(fig,x,y,w,h):
    fw,fh=fig.get_size_inches()*72
    return fig.add_axes([x/fw,y/fh,w/fw,h/fh])

PANEL_GAP_PT = 5

def panel_row(fig, panels):
    """Place a row of regular 9-point panel labels below its lowest axis decoration."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    rows = [(ax, title, ax.get_tightbbox(renderer)) for ax, title in panels]
    bottoms = [box.y0 for _, _, box in rows]
    for ax, _, _ in rows:
        for dimension in (ax.xaxis, ax.yaxis, getattr(ax, "zaxis", None)):
            if dimension is not None and dimension.label.get_text():
                bottoms.append(dimension.label.get_window_extent(renderer).y0)
    write_panel_titles(fig, [((ax.get_position().x0 + ax.get_position().x1) / 2, title)
                             for ax, title, _ in rows], min(bottoms))


def write_panel_titles(fig, centers, lowest):
    renderer = fig.canvas.get_renderer()
    font = font_manager.FontProperties(family="Tinos", size=FONT_SIZE)
    ascent = []
    for _, title in centers:
        _, height, descent = renderer.get_text_width_height_descent(title, font, False)
        ascent.append(height - descent)
    baseline = (lowest - PANEL_GAP_PT * fig.dpi / 72 - max(ascent)) / fig.bbox.height
    for center, title in centers:
        fig.text(center, baseline, title, ha="center", va="baseline",
                 fontproperties=font)


def clean(ax,grid='y'):
    ax.spines[['top','right']].set_visible(False)
    ax.set_axisbelow(True);ax.grid(axis=grid,color='#d9d9d9',lw=.55,ls=':')
    ax.tick_params(length=3,width=.7,pad=3)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(name):return pd.read_csv(DATA/name)

def save(fig,out,stem,sources,semantics):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();issues=[];texts=[]
    ignore=set()
    for ax in fig.axes:
        for axobj,lim in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            lo,hi=sorted(lim)
            for t in axobj.get_major_ticks()+axobj.get_minor_ticks():
                if not lo<=t.get_loc()<=hi:ignore.update([t.label1,t.label2])
    for t in fig.findobj(match=plt.Text):
        if t in ignore or not t.get_visible() or not t.get_text().strip():continue
        b=t.get_window_extent(renderer)
        if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.x1+.5 or b.y1>fig.bbox.y1+.5:
            issues.append({'text':t.get_text(),'bbox':list(b.bounds)})
        texts.append({'text':t.get_text(),'font_pt':t.get_fontsize(),'family':t.get_fontfamily()})
    receipt={'figure':stem,'width_mm':164,'height_mm':fig.get_figheight()*25.4,
      'font_family':'Tinos','font_pt':FONT_SIZE,'natural_size_required':True,
      'data_sha256':{s:digest(DATA/s) for s in sources},
      'numeric_semantics':semantics,'outside_canvas':issues,
      'all_text_uses_common_size':all(t['font_pt']==FONT_SIZE for t in texts),
      'texts':texts,'training_or_resampling':False}
    (out/(stem+'.audit.json')).write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    if issues:raise ValueError(f'{stem}: text outside canvas: {issues}')
    fig.savefig(out/(stem+'.png'),dpi=300)
    fig.savefig(out/(stem+'.pdf'),metadata={'Title':stem,'Creator':'MSA-DTI unified reporting',
        'CreationDate':None,'ModDate':None})
    fig.savefig(out/(stem+'.svg'))
    plt.close(fig)
