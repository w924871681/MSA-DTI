"""Render the six main experimental figures of MSA-DTI v1.1."""
import argparse
from pathlib import Path
import runpy
import subprocess
import sys

from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path("build/figures"))
    folder = parser.parse_args().output.resolve()
    if folder.is_relative_to(ROOT / "reference"):
        parser.error("Choose an output directory outside the manuscript reference directory.")
    folder.mkdir(parents=True, exist_ok=True)

    # Keep independent Matplotlib configurations of the two source scripts.
    subprocess.run([sys.executable, str(ROOT / "src/bitbrains/plot_bitbrains.py"), str(folder)], check=True)
    for path in sorted((ROOT / "fonts/liberation").glob("*.ttf")):
        font_manager.fontManager.addfont(path)
    source = ROOT / "src/manuscript/plot_figures.py"
    sys.path.insert(0, str(source.parent))
    sys.argv = [str(source), str(folder)]
    runpy.run_path(str(source), run_name="__main__")
    print(f"Main experimental figures 5-10 written to {folder}")


if __name__ == "__main__":
    main()
