"""Unified entry point for reproducing MSA-DTI figures, tables, and frozen audits."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
PAPER_FIGURES_DIR = REPO_ROOT / "paper_figures"
DEFAULT_FIG_OUT = REPO_ROOT / "work" / "paper" / "figures"


def run_figures(fig_out: Path) -> int:
    # Ensure the output directory contains strictly the 6 finalized main experimental figures (Figs. 5-10)
    if fig_out.exists():
        shutil.rmtree(fig_out)
    fig_out.mkdir(parents=True, exist_ok=True)

    rc = subprocess.call(
        [sys.executable, str(PAPER_FIGURES_DIR / "render_figures.py"), str(fig_out)]
    )
    if rc != 0:
        return rc

    return subprocess.call(
        [sys.executable, str(CODE_DIR / "plotting" / "generate_paper_assets.py"), "--tables-only"]
    )


def run_verify_figures(fig_out: Path, strict_final_three: bool = False) -> int:
    rc = subprocess.call(
        [sys.executable, str(PAPER_FIGURES_DIR / "verify_inputs.py"), str(REPO_ROOT)]
    )
    if rc != 0:
        return rc
    cmd = [sys.executable, str(PAPER_FIGURES_DIR / "verify_figures.py"), str(fig_out)]
    if strict_final_three:
        cmd.append("--strict-final-three")
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce MSA-DTI finalized experimental figures (Figs. 5-10), LaTeX tables, and numerical audits."
    )
    parser.add_argument(
        "command",
        choices=["figures", "verify", "verify-figures", "all"],
        help=(
            "figures: regenerate finalized Figs. 5-10 (6 PDFs) and LaTeX tables; "
            "verify: run three-tier numerical + live RGMS/SPA verification; "
            "verify-figures: check figure inputs and compare rendered Figs. 5-10 against manuscript PDFs; "
            "all: run verify, figures, and verify-figures."
        ),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIG_OUT,
        help="Destination directory for rendered Figs. 5-10 (default: work/paper/figures).",
    )
    parser.add_argument(
        "--strict-final-three",
        action="store_true",
        help="Pass --strict-final-three to paper_figures/verify_figures.py.",
    )
    args = parser.parse_args()
    fig_out = args.figures_dir.resolve()

    if args.command in ("verify", "all"):
        rc = subprocess.call([sys.executable, str(CODE_DIR / "verification" / "verify_results.py")])
        if rc != 0:
            return rc
    if args.command in ("figures", "all"):
        rc = run_figures(fig_out)
        if rc != 0:
            return rc
    if args.command in ("verify-figures", "all"):
        rc = run_verify_figures(fig_out, strict_final_three=args.strict_final_three)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
