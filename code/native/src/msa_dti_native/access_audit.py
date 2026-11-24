"""Reject experiment inputs outside the extracted package and Python runtime."""
import os,sys
from pathlib import Path
from .io import ROOT,RUN,write,utc

class PackageAccessAudit:
    def __init__(self,job):
        self.job=job;self.active=False;self.denied=[];self.package_reads=set()
        self.runtime_roots={Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve(),Path(sys.exec_prefix).resolve()}
        self.formats={'.pt','.pth','.npz','.npy','.bin','.csv','.zip','.pkl','.pickle','.json','.jsonl','.yaml','.yml'}
    def hook(self,event,args):
        if not self.active or event!='open' or isinstance(args[0],int):return
        path=Path(os.fsdecode(args[0])).resolve()
        mode=args[1]
        if isinstance(mode,str) and any(x in mode for x in ['w','a','x']):return
        if path.is_relative_to(ROOT):
            if path.suffix.lower() in self.formats:self.package_reads.add(path.relative_to(ROOT).as_posix())
            return
        if path.suffix.lower() not in self.formats:return
        if any(path.is_relative_to(p) for p in self.runtime_roots):return
        self.denied.append(str(path));raise PermissionError('External experimental input is not part of this package: '+str(path))
    def __enter__(self):
        sys.addaudithook(self.hook);self.active=True;return self
    def __exit__(self,typ,value,tb):
        self.active=False
        write(RUN/'access'/(self.job+'.json'),dict(job_id=self.job,utc=utc(),status='PASS' if not self.denied else 'FAIL',stage_completed=typ is None,external_experiment_reads_denied=self.denied,package_experiment_reads=sorted(self.package_reads),scope='Python open audit for experiment tensor, archive, and table inputs; not an operating-system sandbox'))
