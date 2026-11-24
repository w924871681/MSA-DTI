"""Unified entry point for reproducing MSA-DTI figures, tables, and frozen audits."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce MSA-DTI experimental figures, LaTeX tables, and numerical audits."
    )
    parser.add_argument(
        "command",
        choices=["figures", "verify", "all"],
        help="figures: regenerate Figs. 5-11 and LaTeX tables; verify: check frozen dataset integrity; all: run both.",
    )
    args = parser.parse_args()

    if args.command in ("verify", "all"):
        rc = subprocess.call([sys.executable, str(CODE_DIR / "verification" / "verify_results.py")])
        if rc != 0:
            return rc
    if args.command in ("figures", "all"):
        rc = subprocess.call([sys.executable, str(CODE_DIR / "plotting" / "generate_paper_assets.py")])
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
