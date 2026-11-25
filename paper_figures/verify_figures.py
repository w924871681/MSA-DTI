"""Compare regenerated main figures 5–10 with the manuscript PDFs.

Usage: python verify_figures.py OUTPUT_DIR [--strict-final-three]
Requires PyMuPDF and NumPy. Run both renders on the same machine.
"""

import argparse
import hashlib
import math
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np


FIGURES = {
    5: "fig5_overall_comparison.pdf",
    6: "fig6_paired_reliability.pdf",
    7: "fig7_robustness_forest.pdf",
    8: "fig8_budget_feasibility.pdf",
    9: "fig9_ablation_coupling.pdf",
    10: "fig10_external_adaptation.pdf",
}
REFERENCE = Path(__file__).resolve().parent / "reference" / "figures"
# Original Matplotlib 3.10.0 vs 3.10.8: observed shift at most 0.46 pt.
POSITION_TOLERANCE_PT = 0.65


def close_coordinates(left, right, tolerance):
    return len(left) == len(right) and all(
        abs(a - b) <= tolerance for a, b in zip(left, right)
    )


def vector_coordinates(path):
    """Flatten Matplotlib's point and rectangle operands in paint order."""
    for item in path["items"]:
        for operand in item[1:]:
            if isinstance(operand, (float, int)):
                yield operand
            else:
                yield from operand


def check_layout(original, generated):
    if not close_coordinates(original.rect, generated.rect, 0.01):
        raise ValueError("page bounds differ")
    first = original.get_text("words")
    second = generated.get_text("words")
    if not first or [word[4] for word in first] != [word[4] for word in second]:
        raise ValueError("labels, displayed values or text order differ")
    for a, b in zip(first, second):
        if not close_coordinates(a[:4], b[:4], POSITION_TOLERANCE_PT):
            raise ValueError(f"position of text {a[4]!r} differs")

    expected_paths = original.get_drawings()
    actual_paths = generated.get_drawings()
    if len(expected_paths) != len(actual_paths):
        raise ValueError("vector path counts differ")
    for index, (a, b) in enumerate(zip(expected_paths, actual_paths), start=1):
        if a["type"] != b["type"] or [i[0] for i in a["items"]] != [i[0] for i in b["items"]]:
            raise ValueError(f"vector operations differ at path {index}")
        if not close_coordinates(
            tuple(vector_coordinates(a)), tuple(vector_coordinates(b)), POSITION_TOLERANCE_PT
        ):
            raise ValueError(f"vector geometry differs at path {index}")

    # Fig. 8 also has embedded raster artwork. Decoded pixel digests ignore PDF compression.
    first_images = original.get_image_info(hashes=True)
    second_images = generated.get_image_info(hashes=True)
    if [im["digest"] for im in first_images] != [im["digest"] for im in second_images]:
        raise ValueError("embedded raster content differs")


def rendered_pixels(page):
    pixmap = page.get_pixmap(matrix=fitz.Matrix(1, 1), colorspace=fitz.csRGB, alpha=False)
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)


def compare(reference, regenerated):
    same_bytes = hashlib.sha256(reference.read_bytes()).digest() == hashlib.sha256(regenerated.read_bytes()).digest()
    with fitz.open(reference) as a, fitz.open(regenerated) as b:
        if len(a) != 1 or len(b) != 1:
            raise ValueError("one page per figure is required")
        check_layout(a[0], b[0])
        original_pixels = rendered_pixels(a[0])
        new_pixels = rendered_pixels(b[0])
        if original_pixels.shape != new_pixels.shape:
            raise ValueError("render dimensions differ")
        delta = np.abs(original_pixels.astype(np.int16) - new_pixels.astype(np.int16))
        mean_delta = float(delta.mean())
        changed = float((delta.max(axis=2) > 32).mean())
        # Across the three reference charts from 3.10.0 the maxima observed in 3.10.8
        # were 0.85/255 mean difference and 0.94% of pixels above the channel tolerance.
        if not math.isfinite(mean_delta) or mean_delta > 1.5 or changed > 0.015:
            raise ValueError(f"render differs (mean {mean_delta:.2f}/255, changed pixels {changed:.2%})")
    return same_bytes, mean_delta, changed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_dir", type=Path, help="directory containing the six regenerated PDFs")
    ap.add_argument("--reference-dir", type=Path, default=REFERENCE)
    ap.add_argument("--strict-final-three", action="store_true",
                    help="require byte-identical output for Figs. 5, 9 and 10 in the pinned rendering environment")
    args = ap.parse_args()
    failures = 0
    for number, filename in FIGURES.items():
        try:
            exact, mean_delta, changed = compare(args.reference_dir / filename, args.output_dir / filename)
            if args.strict_final_three and number in (5, 9, 10) and not exact:
                raise ValueError("PDF bytes differ from the final reference (--strict-final-three)")
            result = "identical bytes" if exact else "text, geometry and render agree within tolerance"
            print(f"Fig. {number:>2}: PASS — {result} (mean pixel delta {mean_delta:.3f}/255, changed {changed:.2%})")
        except (OSError, RuntimeError, ValueError) as error:
            failures += 1
            print(f"Fig. {number:>2}: FAIL — {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
