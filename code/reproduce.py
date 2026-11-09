"""Reviewer entry points. All generated files stay in work/."""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile, urllib.request, zipfile
from pathlib import Path

CODE=Path(__file__).resolve().parent
ROOT=CODE.parent
WORK=ROOT/"work"
def checksum(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def py(path,*args,cwd=None):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",MPLBACKEND="Agg",PYTHONUTF8="1")
    subprocess.run([sys.executable,"-B",str(path),*map(str,args)],cwd=cwd,env=env,check=True)
def same_pdf(current,reference):
    """Different font/PDF encoders may represent identical drawn content differently."""
    import pymupdf
    import numpy as np
    with pymupdf.open(stream=current,filetype="pdf") as actual,pymupdf.open(stream=reference,filetype="pdf") as expected:
        if len(actual)!=len(expected):return False
        for page,ref in zip(actual,expected):
            if tuple(page.rect)!=tuple(ref.rect):return False
            if "".join(page.get_text().split())!="".join(ref.get_text().split()):return False
            panels=[p.get_pixmap(matrix=pymupdf.Matrix(2,2),alpha=False) for p in (page,ref)]
            pixels=[np.frombuffer(pm.samples,dtype=np.uint8).reshape(pm.height,pm.width,pm.n) for pm in panels]
            delta=np.abs(pixels[0].astype(np.int16)-pixels[1].astype(np.int16))
            if delta.mean()>0.35 or np.mean(delta.max(axis=2)>48)>0.0015:return False
    return True
def fetch(groups):
    manifest=json.loads((CODE/"assets.json").read_bytes())
    dest=WORK/"assets";dest.mkdir(parents=True,exist_ok=True)
    for name,item in manifest.items():
        if item["group"] not in groups:continue
        out=dest/name
        if not out.exists():
            with tempfile.NamedTemporaryFile(dir=dest,delete=False) as tmp:
                tmp_path=Path(tmp.name)
            try:
                with urllib.request.urlopen(item["url"],timeout=120) as source,tmp_path.open("wb") as fp:shutil.copyfileobj(source,fp)
                if checksum(tmp_path)!=item["sha256"]:raise ValueError("Download mismatch: "+name)
                os.replace(tmp_path,out)
            finally:tmp_path.unlink(missing_ok=True)
        if checksum(out)!=item["sha256"]:raise ValueError("Asset checksum mismatch: "+name)
        print("Verified",name,flush=True)
        if item["group"]=="weights":
            with zipfile.ZipFile(out) as z:z.extractall(WORK)
def build(from_verified):
    out=WORK/"paper";out.mkdir(parents=True,exist_ok=True)
    # Start with no pre-existing figure or table output, so a silent skip is detectable.
    with tempfile.TemporaryDirectory(dir=WORK,prefix="plot-") as temp:
        stage=Path(temp)
        shutil.copytree(CODE/"plotting",stage/"plotting");shutil.copytree(CODE/"data",stage/"data")
        if from_verified:
            from report_inputs import populate
            populate(stage/"data")
        (stage/"figures").mkdir();(stage/"tables").mkdir()
        scripts=["build_results","plot_figures","plot_budget_figure","plot_msa_vs_fixed_pt","plot_azure_budget_paths","build_supplement"]
        for script in scripts:py(stage/"plotting"/(script+".py"),cwd=stage)
        with zipfile.ZipFile(CODE/"experimental-reference.zip") as canonical:
            expected={s for s in canonical.namelist() if s.startswith("tables/") and s not in ("tables/baselines.tex","tables/configuration.tex")}
            # These nine plots have experimental inputs and retained verification references.
            plots = ("fig5", "fig6", "fig7", "budget_feasibility", "fig9_deployment",
                     "fig10_azure_budget", "figS1", "figS2", "figS3")
            expected.update(f"figures/{stem}.pdf" for stem in plots)
            rendered=[]
            for name in sorted(expected):
                item=stage/name
                # Text newline representation depends on the operating system.
                a=item.read_bytes();b=canonical.read(name)
                if name.endswith(".tex"):a,b=a.replace(b"\r\n",b"\n"),b.replace(b"\r\n",b"\n")
                if a!=b:
                    if not name.endswith(".pdf") or not same_pdf(a,b):raise ValueError("Output differs from experimental reference: "+name)
                    rendered.append(name)
            canonical.extractall(out)
            for name in expected:shutil.copyfile(stage/name,out/name)
        print("Verified",len(expected),"figure/table files;",len(rendered),"PDFs matched by page size, text and 144-dpi rendering. Output:",out)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="cmd",required=True)
    figures = sub.add_parser("figures")
    sources = figures.add_mutually_exclusive_group()
    sources.add_argument("--from-verified", action="store_true", help="Use the two published-result audits")
    sources.add_argument("--from-run", type=Path, help="Report the target CSVs in this run directory")
    figures.add_argument("--partial", action="store_true", help="Allow a matched subset of tasks or methods with --from-run")
    f=sub.add_parser("fetch");f.add_argument("groups",nargs="+",choices=["scores","weights","supplement"])
    for cmd in ("scores","supplement"):
        a=sub.add_parser(cmd);a.add_argument("--bitbrains",type=Path,required=True)
    args=parser.parse_args();WORK.mkdir(exist_ok=True)
    if args.cmd=="figures":
        if args.from_run is not None:
            from run_report import report
            report(args.from_run, partial=args.partial)
        elif args.partial:
            parser.error("--partial requires --from-run")
        else:
            build(args.from_verified)
    elif args.cmd=="fetch":fetch(args.groups)
    elif args.cmd=="scores":
        fetch(["scores"]);py(CODE/"verify_results.py","--bitbrains",args.bitbrains.resolve())
    else:
        fetch(["supplement"]);py(CODE/"verification/check_supplement.py",args.bitbrains.resolve())
if __name__=="__main__":main()

