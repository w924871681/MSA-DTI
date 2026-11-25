"""Rebuild manuscript Figures 6, 7, and 8 from the frozen Bitbrains inputs.

The figure builders, display settings and bootstrap calculation are taken from
w924871681/MSA-DTI, v1.1 (1438a1d4), code/plotting/generate_paper_assets.py.
Only the command line, input paths and PDF export wrapper are adapted here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from math import erfc, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FONT_DIR = HERE / "fonts"
DATA_DIR = HERE / "data"
SRC_EVENTS_720 = DATA_DIR / "deployment_events_720.csv.gz"
SRC_TASK_SCORES = DATA_DIR / "primary_task_scores.csv"
SRC_BUDGET_PROF = DATA_DIR / "budget_resource_profiles.json"
SRC_SEL_DEC = DATA_DIR / "selection_decisions.csv.gz"

WIDTH_PT = 164.0 / 25.4 * 72.0
FONT_SIZE = 8.5
BLUE = "#0072B2"
GREEN = "#009E73"
FOREST = "#3E7B58"


def apply_style():
    for f in FONT_DIR.glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({
        "font.family": "Tinos",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 0.5,
        "ytick.labelsize": FONT_SIZE - 0.5,
        "legend.fontsize": FONT_SIZE - 0.5,
        "figure.titlesize": FONT_SIZE + 0.5,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Tinos",
        "mathtext.it": "Tinos:italic",
        "mathtext.bf": "Tinos:bold",
        "mathtext.fallback": "stix",
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def new_canvas(height_pt: float) -> plt.Figure:
    apply_style()
    return plt.figure(figsize=(WIDTH_PT / 72.0, height_pt / 72.0))


def add_ax(fig: plt.Figure, x: float, y: float, w: float, h: float) -> plt.Axes:
    fw, fh = fig.get_size_inches() * 72.0
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh])


def clean_ax(ax: plt.Axes, grid: str = "y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color="#DCDCDC", lw=0.55, ls=":")
    ax.tick_params(length=2.8, width=0.7, pad=2.5)


def place_panel_labels(fig: plt.Figure, panels: list[tuple[plt.Axes, str]], gap_pt: float = 5.0):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    rows = [(ax, title, ax.get_tightbbox(renderer)) for ax, title in panels]
    bottoms = [box.y0 for _, _, box in rows]
    for ax, _, _ in rows:
        for dim in (ax.xaxis, ax.yaxis, getattr(ax, "zaxis", None)):
            if dim is not None and dim.label.get_text():
                bottoms.append(dim.label.get_window_extent(renderer).y0)
    lowest = min(bottoms)
    font = font_manager.FontProperties(family="Tinos", size=FONT_SIZE)
    ascent = []
    for _, title, _ in rows:
        _, h, d = renderer.get_text_width_height_descent(title, font, False)
        ascent.append(h - d)
    baseline = (lowest - gap_pt * fig.dpi / 72.0 - max(ascent)) / fig.bbox.height
    for ax, title, _ in rows:
        cx = (ax.get_position().x0 + ax.get_position().x1) / 2.0
        fig.text(cx, baseline, title, ha="center", va="baseline", fontproperties=font)


def _luminance(rgb):
    rgb = np.asarray(rgb[:3], dtype=float)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(np.dot(lin, [0.2126, 0.7152, 0.0722]))


def contrast_text_color(rgba):
    bg = _luminance(rgba)
    c_dark = (max(bg, 0.015) + 0.05) / (min(bg, 0.015) + 0.05)
    c_white = (1.0 + 0.05) / (bg + 0.05)
    return "#1F2328" if c_dark >= c_white else "white"


def save_pdf(fig: plt.Figure, output_dir: Path, stem: str, *, pdf_dpi: int = 300) -> None:
    fig.canvas.draw()
    fig.savefig(output_dir / f"{stem}.pdf", dpi=pdf_dpi, metadata={
        "Title": stem, "Creator": "Paper1 Final Assets", "CreationDate": None, "ModDate": None,
    })
    plt.close(fig)


def boot_rel_gain(df: pd.DataFrame, col_new: str, col_ref: str, n_boot: int = 4000, seed: int = 20261123):
    g = df.groupby("VM")[[col_new, col_ref]].mean()
    a = g[col_new].to_numpy()
    b = g[col_ref].to_numpy()
    pt = 100.0 * (1.0 - a.mean() / b.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    draws = 100.0 * (1.0 - a[idx].mean(axis=1) / b[idx].mean(axis=1))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(pt), float(lo), float(hi)


def build_fig6(output_dir: Path):
    ev = pd.read_csv(SRC_EVENTS_720)
    task_scores = pd.read_csv(SRC_TASK_SCORES)
    ptft = task_scores[task_scores["method"] == "PT-FT"][["stage", "VM", "slug", "H", "K", "budget", "MSE"]].rename(
        columns={"MSE": "MSE_PTFT"}
    )
    merged = ev.merge(ptft, on=["stage", "VM", "slug", "H", "K", "budget"], how="left")
    med = merged[(merged["stage"] == "E") & (merged["budget"] == "medium")].copy()

    fig = new_canvas(222)
    ax_a = add_ax(fig, 52, 46, 151, 151)
    ax_b = add_ax(fig, 288, 46, 151, 151)

    for ax, xcol, xlabel in [
        (ax_a, "MSE_PTFT", "PT-FT task MSE (log scale)"),
        (ax_b, "MSE_FullVal__StandardFT", "Full-Val task MSE (log scale)"),
    ]:
        ycol = "MSE_RGMS__SPA"
        vals = med[[xcol, ycol]].dropna().to_numpy()
        pos = vals[vals > 0]
        lo = max(1e-8, 10.0 ** np.floor(np.log10(pos.min())))
        hi = 10.0 ** np.ceil(np.log10(pos.max()))
        ax.plot([lo, hi], [lo, hi], color="#222222", lw=0.75, ls="--", zorder=1)
        tot_wins = int((med[ycol] < med[xcol]).sum())
        for h_val, label, col, mk in [(1, "H = 1", BLUE, "o"), (4, "H = 4", GREEN, "s")]:
            sub = med[med["H"] == h_val]
            wins = int((sub[ycol] < sub[xcol]).sum())
            xv = np.maximum(sub[xcol].to_numpy(), lo * 1.05)
            yv = np.maximum(sub[ycol].to_numpy(), lo * 1.05)
            ax.scatter(xv, yv, s=20, marker=mk, facecolors="white",
                       edgecolors=col, linewidths=0.9, alpha=0.9,
                       label=f"{label} ({wins}/{len(sub)} wins)", zorder=2)
        ax.text(0.96, 0.06, f"Total: {tot_wins}/{len(med)} wins", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, fontweight="bold", color="#1D3E2B",
                bbox=dict(boxstyle="round,pad=0.2", fc="#EEF6F0", ec=FOREST, lw=0.5))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
        ax.set_xlabel(xlabel, labelpad=2.5)
        ax.set_ylabel("MSA-DTI task MSE (log scale)", labelpad=2.5)
        clean_ax(ax, "both")
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#DDDDDD",
                  fontsize=7.3, handletextpad=0.3, borderpad=0.3)

    place_panel_labels(fig, [(ax_a, "(a) Paired MSE: MSA-DTI vs. PT-FT"),
                             (ax_b, "(b) Paired MSE: MSA-DTI vs. Full-Val")], gap_pt=4.5)
    save_pdf(fig, output_dir, "fig6_paired_reliability")


def build_fig7_forest(output_dir: Path):
    ev = pd.read_csv(SRC_EVENTS_720)
    task_scores = pd.read_csv(SRC_TASK_SCORES)
    ptft = task_scores[task_scores["method"] == "PT-FT"][["stage", "VM", "slug", "H", "K", "budget", "MSE"]].rename(
        columns={"MSE": "MSE_PTFT"}
    )
    merged = ev.merge(ptft, on=["stage", "VM", "slug", "H", "K", "budget"], how="left")
    e_all = merged[merged["stage"] == "E"].copy()
    e_med = e_all[e_all["budget"] == "medium"].copy()

    fig = new_canvas(184)
    ax_a = add_ax(fig, 78, 46, 146, 118)
    ax_b = add_ax(fig, 306, 46, 146, 118)

    hk_specs = [(1, 10, "H=1, K=10"), (1, 20, "H=1, K=20"), (4, 10, "H=4, K=10"), (4, 20, "H=4, K=20")]
    y_hk = np.arange(len(hk_specs))[::-1]
    pts, los, his = [], [], []
    for H, K, _ in hk_specs:
        sub = e_med[(e_med["H"] == H) & (e_med["K"] == K)]
        p, l, h = boot_rel_gain(sub, "MSE_RGMS__SPA", "MSE_PTFT")
        pts.append(p); los.append(l); his.append(h)
    pts, los, his = np.array(pts), np.array(los), np.array(his)
    ax_a.errorbar(pts, y_hk, xerr=np.vstack([pts - los, his - pts]),
                  fmt="o", color=BLUE, capsize=3.0, markersize=4.6, lw=1.15)
    for yy, v, lo_v, hi_v in zip(y_hk, pts, los, his):
        ax_a.annotate(f"{v:.1f}% [{lo_v:.1f}, {hi_v:.1f}]", (v, yy), xytext=(0, 6.0),
                      textcoords="offset points", ha="center", fontsize=6.9, color=BLUE)
    ax_a.axvline(0, color="#333333", lw=0.65, ls=":")
    ax_a.set_yticks(y_hk, [s[2] + " (n=30)" for s in hk_specs])
    ax_a.set_ylim(-0.55, 3.65)
    ax_a.set_xlim(-2, 56)
    ax_a.xaxis.set_major_locator(MultipleLocator(10))
    ax_a.set_xlabel("MSE reduction vs. PT-FT (%)", labelpad=3.0)
    clean_ax(ax_a, "x")

    b_specs = [("tight", "Tight (10 cand.)"), ("medium", "Medium (17 cand.)"), ("loose", "Loose (19 cand.)")]
    y_b = np.arange(len(b_specs))[::-1]
    pts, los, his = [], [], []
    for b, _ in b_specs:
        sub = e_all[e_all["budget"] == b]
        p, l, h = boot_rel_gain(sub, "MSE_RGMS__SPA", "MSE_FullVal__StandardFT")
        pts.append(p); los.append(l); his.append(h)
    pts, los, his = np.array(pts), np.array(los), np.array(his)
    ax_b.errorbar(pts, y_b, xerr=np.vstack([pts - los, his - pts]),
                  fmt="s", color=FOREST, capsize=3.0, markersize=4.6, lw=1.15)
    for yy, v, lo_v, hi_v in zip(y_b, pts, los, his):
        ax_b.annotate(f"{v:.1f}% [{lo_v:.1f}, {hi_v:.1f}]", (v, yy), xytext=(0, 6.0),
                      textcoords="offset points", ha="center", fontsize=6.9, color="#1D3E2B")
    ax_b.axvline(0, color="#333333", lw=0.65, ls=":")
    ax_b.set_yticks(y_b, [s[1] for s in b_specs])
    ax_b.set_ylim(-0.55, 2.65)
    ax_b.set_xlim(0, 32)
    ax_b.xaxis.set_major_locator(MultipleLocator(5))
    ax_b.set_xlabel("MSE reduction vs. Full-Val (%)", labelpad=3.0)
    clean_ax(ax_b, "x")

    place_panel_labels(fig, [(ax_a, "(a) Horizon and support size (medium)"),
                             (ax_b, "(b) Deployment budget (Tight / Medium / Loose)")], gap_pt=4.5)
    save_pdf(fig, output_dir, "fig7_robustness_forest")


_ERFC = np.vectorize(erfc, otypes=[np.float64])
_PARAM_RANGE = (2_081.0, 100_000.0)
_OP_RANGE = (155_712.0, 1_250_000.0)
_SIGMA_FRAC = 0.10


def _passing_fraction(centers, reqs, limits, sigma=_SIGMA_FRAC):
    lo, hi = limits
    centers = np.asarray(centers, dtype=float)[..., None]
    reqs = np.asarray(reqs, dtype=float)
    scale = sqrt(2.0) * sigma * (hi - lo)
    upper_tail = _ERFC((hi - centers) / scale)
    mass = _ERFC((lo - centers) / scale) - upper_tail
    passing = _ERFC((np.clip(reqs, lo, hi) - centers) / scale) - upper_tail
    return passing / mass


def _weighted_feasible_mean(p, f, costs):
    wp = _passing_fraction(p, costs[:, 0], _PARAM_RANGE)
    wf = _passing_fraction(f, costs[:, 1], _OP_RANGE)
    return np.sum(wp * wf, axis=-1)


def build_fig8_budget_feasibility(output_dir: Path):
    prof = json.loads(SRC_BUDGET_PROF.read_text(encoding="utf-8"))
    costs = np.array([(x["parameters"], x["operations"]) for x in prof["1"].values()], dtype=float)
    assert len(costs) == 19

    sel_dec = pd.read_csv(SRC_SEL_DEC)
    sub_e = sel_dec[(sel_dec["scope"] == "full") & (sel_dec["arm"] == "Match") & (sel_dec["stage"] == "E")].copy()
    assert len(sub_e) == 360

    fig = new_canvas(214)

    # Panel (a): Exact Original Gaussian-Weighted 3D Feasibility Surface (H = 1)
    p = np.linspace(*_PARAM_RANGE, 193)
    f = np.linspace(*_OP_RANGE, 193)
    pp, ff = np.meshgrid(p, f)
    wp = _passing_fraction(p, costs[:, 0], _PARAM_RANGE)
    wf = _passing_fraction(f, costs[:, 1], _OP_RANGE)
    z = wf @ wp.T
    ax_s = fig.add_axes([0.025, 0.181, 0.471, 0.777], projection="3d", computed_zorder=False)
    surf_cmap = LinearSegmentedColormap.from_list(
        "feasibility", ["#e9f1e9", "#b9d8ce", "#78b9af", "#37868a", "#185665", "#173d53"], N=2048
    )
    ax_s.plot_surface(
        pp / 1000.0, ff / 1e6, z, cmap=surf_cmap, norm=Normalize(0, len(costs)),
        rcount=len(f), ccount=len(p), linewidth=0, antialiased=False,
        shade=False, rasterized=True, zorder=1
    )
    for center in np.linspace(*_PARAM_RANGE, 9):
        ax_s.plot(np.full_like(f, center) / 1000.0, f / 1e6, _weighted_feasible_mean(center, f, costs),
                  color="#224955", alpha=0.52, lw=0.32, zorder=3)
    for center in np.linspace(*_OP_RANGE, 9):
        ax_s.plot(p / 1000.0, np.full_like(p, center) / 1e6, _weighted_feasible_mean(p, center, costs),
                  color="#224955", alpha=0.52, lw=0.32, zorder=3)
    ax_s.set(xlim=(p[-1] / 1000.0, p[0] / 1000.0), ylim=(f[-1] / 1e6, f[0] / 1e6),
             zlim=(0, len(costs)), zticks=[0, 5, 10, 15, 19])
    ax_s.set_xticks([25, 50, 75, 100])
    ax_s.set_yticks([0.25, 0.50, 0.75, 1.00, 1.25], ["0.25", "0.50", "0.75", "1.00", "1.25"])
    ax_s.set_xlabel("Parameter cap (thousands)", labelpad=1.0)
    ax_s.set_ylabel("Compute cap (M FLOPs)", labelpad=1.0)
    ax_s.set_zlabel("Mean feasible candidates", labelpad=1.0)
    ax_s.tick_params(labelsize=8.2, pad=0)
    ax_s.view_init(elev=30, azim=42)
    ax_s.set_proj_type("ortho")
    ax_s.set_box_aspect((1.25, 1.06, 0.90), zoom=0.96)
    for axis in (ax_s.xaxis, ax_s.yaxis, ax_s.zaxis):
        axis.set_pane_color((0.975, 0.981, 0.979, 1.0))
        axis._axinfo["grid"].update(color="#d6e0dd", linewidth=0.45)

    # Panel (b): Final Model Selection Heatmap (Exact original GnBu palette, NEW MSA-DTI decisions!)
    ax_h = fig.add_axes([0.640, 0.177, 0.253, 0.699])
    ax_cb = fig.add_axes([0.913, 0.177, 0.013, 0.699])
    top6 = [int(q) for q, _ in Counter(sub_e["pick"]).most_common(6)]
    rates = np.array([
        [100.0 * ((sub_e["bi"] == bi) & (sub_e["pick"] == q)).sum() / 120.0 for bi in (0, 1, 2)]
        for q in top6
    ])
    rates = np.vstack([rates, 100.0 - rates.sum(axis=0)])
    assert rates.shape == (7, 3) and np.allclose(rates.sum(axis=0), 100.0)

    cmap_gnbu = plt.get_cmap("GnBu")
    norm_sel = Normalize(0, 60)
    im = ax_h.imshow(rates, cmap=cmap_gnbu, norm=norm_sel, aspect="auto", interpolation="nearest")
    ax_h.set_xticks([0, 1, 2], ["Tight", "Medium", "Loose"])
    ax_h.set_yticks(range(7), [f"A{x}" for x in top6] + ["Other"])
    ax_h.tick_params(length=3, width=0.7, pad=3)
    for i in range(7):
        for j in range(3):
            rgba = cmap_gnbu(norm_sel(rates[i, j]))
            ax_h.text(j, i, f"{rates[i, j]:.1f}%", ha="center", va="center",
                      fontsize=8.0, color=contrast_text_color(rgba))
    cb = fig.colorbar(im, cax=ax_cb, ticks=range(0, 61, 15))
    cb.set_label("Selection rate (%)", labelpad=4)

    place_panel_labels(fig, [(ax_s, "(a) Weighted feasible count (H = 1)"),
                             (ax_h, "(b) Final model selection")], gap_pt=4.5)
    save_pdf(fig, output_dir, "fig8_budget_feasibility", pdf_dpi=600)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory for the three figure PDFs")
    output_dir = parser.parse_args().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    build_fig6(output_dir)
    build_fig7_forest(output_dir)
    build_fig8_budget_feasibility(output_dir)


if __name__ == "__main__":
    main()
