"""Package-relative paths and atomic experiment records."""
import hashlib,json,os,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'
RUN=Path(os.environ.get('MSADTI_OUTPUT',str(ROOT/'results'))).resolve()
if not RUN.is_relative_to(ROOT) or RUN==ROOT:
    raise ValueError('MSADTI_OUTPUT must be a separate directory inside the extracted MSA-DTI package')
REAL=DATA
CODE=ROOT/'src'

def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def canon(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canon(x).encode()).hexdigest()
def require(v,m):
    if not v:raise RuntimeError(m)
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    # Windows readers may briefly hold the destination without delete sharing.
    # Keep the previous complete record until atomic replacement succeeds.
    deadline=time.monotonic()+5.;delay=.02
    while True:
        try:
            os.replace(t,p)
            return
        except OSError as error:
            remaining=deadline-time.monotonic()
            if os.name!='nt' or getattr(error,'winerror',None) not in (5,32,33) or remaining<=0:
                raise
            time.sleep(min(delay,remaining));delay=min(.5,delay*2)
def write_progress(p,obj):write(p,obj)
def event(kind,**kw):
    p=RUN/'events.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as f:f.write(canon(dict(utc=utc(),kind=kind,**kw))+'\n')

