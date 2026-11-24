"""Generate publication-ready experimental figures (Fig. 5--10) and exact LaTeX tables
(Main Tables 1--6 + Supplementary Tables S8--S11) for MSA-DTI.

Layout contract (strictly aligned with the revised manuscript):
- Sections 5.2--5.5 (Fig. 5--9, Tables 3--5): Bitbrains E primary evaluation ONLY.
- Section 5.6 (Fig. 10, Table 6): Cross-platform External Evaluation (Azure Public Dataset V2 + UFCG 2024--2025).
- Supplementary Material: Full 3-budget tables (Bitbrains E & Azure), Azure ablation, and Bitbrains Rnd6 prospective cohort.
- Zero coined acronyms (no BC-ARS, no AP-NCI, no (M1)/(M2)), zero semicolons in captions or cells.
- Reads strictly frozen CSV/JSON artifacts in code/data/.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from math import erfc, sqrt
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.ticker import MultipleLocator

# ---------- Paths (repository-relative, portable across platforms) ----------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "code" / "data"
FONT_DIR = REPO_ROOT / "code" / "plotting" / "fonts"
OUT_ROOT = REPO_ROOT / "work" / "paper"
FIG_DIRS = [OUT_ROOT / "figures"]
TAB_DIRS = [OUT_ROOT / "tables"]
OUT_DATA = OUT_ROOT / "audit"

# Frozen audited datasets
SRC_PRIMARY_BASELINES = DATA_DIR / "primary_baselines.csv"
SRC_SAME_BANK = DATA_DIR / "same_bank_control.csv"
SRC_TASK_SCORES = DATA_DIR / "primary_task_scores.csv"
SRC_UNIFIED_T1 = DATA_DIR / "unified_benchmark_summary.csv"
SRC_COUPLING_MAT = DATA_DIR / "coupling_matrix_risk.csv"
SRC_PAIRWISE_CMP = DATA_DIR / "pairwise_comparisons.csv"
SRC_EVENTS_720 = DATA_DIR / "deployment_events_720.csv.gz"
SRC_TIMING_SUM = DATA_DIR / "timing_summary.json"
SRC_EXT_T1 = DATA_DIR / "external_prospective_summary.csv"
SRC_EXT_DEP = DATA_DIR / "external_deployments.csv.gz"
SRC_SEL_DEC = DATA_DIR / "selection_decisions.csv.gz"
SRC_BUDGET_PROF = DATA_DIR / "budget_resource_profiles.json"

# ---------- Exact unified_style.py visual contract ----------
WIDTH_PT = 164.0 / 25.4 * 72.0   # 464.88 pt (Elsevier double-column width)
FONT_SIZE = 8.5
BLUE = "#0072B2"
GREEN = "#009E73"
FOREST = "#3E7B58"
ORANGE = "#D55E00"
GREY = "#666666"
SLATE = "#6E8FB3"

AUDIT_RECORDS: dict[str, dict] = {}


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


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _luminance(rgb):
    rgb = np.asarray(rgb[:3], dtype=float)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(np.dot(lin, [0.2126, 0.7152, 0.0722]))


def contrast_text_color(rgba):
    bg = _luminance(rgba)
    c_dark = (max(bg, 0.015) + 0.05) / (min(bg, 0.015) + 0.05)
    c_white = (1.0 + 0.05) / (bg + 0.05)
    return "#1F2328" if c_dark >= c_white else "white"


def save_figure(fig: plt.Figure, stem: str, sources: list[Path], pdf_dpi: int = 300):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ignore = set()
    for ax in fig.axes:
        if getattr(ax, "name", "") == "3d":
            continue
        for axobj, lim in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            lo, hi = sorted(lim)
            for t in axobj.get_major_ticks() + axobj.get_minor_ticks():
                if not (lo - 1e-12 <= t.get_loc() <= hi + 1e-12):
                    ignore.update([t.label1, t.label2])
    issues = []
    for t in fig.findobj(match=plt.Text):
        if t in ignore or not t.get_visible() or not t.get_text().strip():
            continue
        b = t.get_window_extent(renderer)
        if b.x0 < -1.5 or b.y0 < -1.5 or b.x1 > fig.bbox.x1 + 1.5 or b.y1 > fig.bbox.y1 + 1.5:
            issues.append({"text": t.get_text(), "bbox": [round(v, 2) for v in b.bounds],
                           "canvas": [round(fig.bbox.x1, 2), round(fig.bbox.y1, 2)]})
    for d in FIG_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}.png", dpi=300)
        fig.savefig(d / f"{stem}.pdf", dpi=pdf_dpi, metadata={"Title": stem, "Creator": "Paper1 Final Assets",
                                                              "CreationDate": None, "ModDate": None})
        fig.savefig(d / f"{stem}.svg")
    plt.close(fig)
    AUDIT_RECORDS[stem] = {
        "width_mm": 164.0,
        "height_mm": round(fig.get_figheight() * 25.4, 2),
        "outside_canvas_count": len(issues),
        "outside_canvas": issues,
        "sources": {str(s.name): sha256_file(s) for s in sources},
    }
    if issues:
        print(f"WARNING [{stem}] outside canvas texts: {issues}")
    else:
        print(f"OK [{stem}] 164mm x {AUDIT_RECORDS[stem]['height_mm']}mm, 0 clipping issues.")


# ---------- Data Helpers ----------
def load_curated_medium_table():
    sb = pd.read_csv(SRC_SAME_BANK)
    nat = pd.read_csv(SRC_PRIMARY_BASELINES)
    t1 = pd.read_csv(SRC_UNIFIED_T1)
    ev = pd.read_csv(SRC_EVENTS_720)

    def get_sb(cn_name: str, budget_cn: str = "中"):
        row = sb[(sb["method"] == cn_name) & (sb["budget"] == budget_cn)].iloc[0]
        return {
            "E_MSE": float(row["bitbrains_mse_x1000"]),
            "E_MAE": float(row["bitbrains_mae_x1000"]),
            "E_W10": float(row["bitbrains_worst10_x1000"]),
            "E_CVaR": float(row["bitbrains_cvar10_x1000"]),
            "A_MSE": float(row["azure_mse_x1000"]),
            "A_MAE": float(row["azure_mae_x1000"]),
            "A_W10": float(row["azure_worst10_x1000"]),
            "A_CVaR": float(row["azure_cvar10_x1000"]),
        }

    def get_nat(cn_name: str, budget_cn: str = "中"):
        row = nat[(nat["method"] == cn_name) & (nat["budget"] == budget_cn)].iloc[0]
        return {
            "E_MSE": float(row["bitbrains_mse_x1000"]),
            "E_MAE": float(row["bitbrains_mae_x1000"]),
            "E_W10": float(row["bitbrains_worst10_x1000"]),
            "E_CVaR": float(row["bitbrains_cvar10_x1000"]),
            "A_MSE": float(row["azure_mse_x1000"]),
            "A_MAE": float(row["azure_mae_x1000"]),
            "A_W10": float(row["azure_worst10_x1000"]),
            "A_CVaR": float(row["azure_cvar10_x1000"]),
        }

    def get_unified(method: str, budget: str = "medium"):
        rE = t1[(t1["stage"] == "E") & (t1["budget"] == budget) & (t1["method"] == method)].iloc[0]
        rA = t1[(t1["stage"] == "azure") & (t1["budget"] == budget) & (t1["method"] == method)].iloc[0]
        return {
            "E_MSE": float(rE["MSE"]) * 1000.0,
            "E_MAE": float(rE["MAE"]) * 1000.0,
            "E_W10": float(rE["Worst10"]) * 1000.0,
            "E_CVaR": float(rE["CVaR10"]) * 1000.0,
            "A_MSE": float(rA["MSE"]) * 1000.0,
            "A_MAE": float(rA["MAE"]) * 1000.0,
            "A_W10": float(rA["Worst10"]) * 1000.0,
            "A_CVaR": float(rA["CVaR10"]) * 1000.0,
        }

    scpu = {}
    for st, pref in [("E", "E"), ("azure", "A")]:
        sub = ev[(ev["stage"] == st) & (ev["budget"] == "medium")]
        scpu[f"{pref}_MSE"] = float(sub.groupby("VM")["MSE_standalone_cpu"].mean().mean()) * 1000.0
        scpu[f"{pref}_CVaR"] = float(sub["MSE_standalone_cpu"].nlargest(12).mean()) * 1000.0

    rows = [
        ("Reuse-Aware", get_sb("Reuse-Aware-style")),
        ("LEAD (Full-NTK)", get_sb("LEAD-style（完整NTK）")),
        ("H-Meta-NAS", get_nat("H-Meta-NAS-style（修正）")),
        ("T-NAS", get_nat("T-NAS-style")),
        ("PAVE", get_sb("PAVE-style")),
        ("PT-FT", get_nat("PT-FT")),
        ("Zero-step-Val", get_sb("零步验证选型")),
        ("Full-Val", get_sb("全部适配后验证")),
        ("MSA-DTI (Ours)", get_unified("MSA-DTI (Full)")),
    ]
    return rows, scpu


# =============================================================================
# FIG. 5 (Bitbrains E ONLY, crisp 1x2 layout, NO Standalone-CPU line — matches orig_fig5!)
# =============================================================================
def build_fig5():
    rows, _ = load_curated_medium_table()
    names = [r[0] for r in rows]
    colors = [
        "#C77979",  # Reuse-Aware
        "#B39BBC",  # LEAD
        "#8F99A6",  # H-Meta-NAS
        "#D9A25F",  # T-NAS
        "#8DB8B0",  # PAVE
        "#6E8FB3",  # PT-FT
        "#9FB6C8",  # Zero-step-Val
        "#7D8C7D",  # Full-Val
        FOREST,     # MSA-DTI (Ours)
    ]

    fig = new_canvas(178)
    specs = [
        ("E_MSE", "(a) Bitbrains E — Medium-budget MSE", r"MSE ($\times 10^{-3}$)"),
        ("E_CVaR", "(b) Bitbrains E — Medium-budget CVaR-10%", r"CVaR-10% ($\times 10^{-3}$)"),
    ]

    panels = []
    y = np.arange(len(names))
    for i, (key, title, xlabel) in enumerate(specs):
        left = 86 if i == 0 else 314
        ax = add_ax(fig, left, 42, 140, 124)
        vals = np.array([r[1][key] for r in rows])
        xmax = float(vals.max() * 1.24)

        bars = ax.barh(y, vals, height=0.62, color=colors, edgecolor="#444444", lw=0.5)
        bars[-1].set_hatch("//")
        bars[-1].set_edgecolor("#1D3E2B")
        bars[-1].set_linewidth(0.8)

        ax.set_yticks(y, names)
        ax.set_ylim(len(names) - 0.35, -0.58)
        ax.set_xlim(0, xmax)
        for tick in ax.get_yticklabels():
            if tick.get_text().startswith("MSA-DTI"):
                tick.set_fontweight("bold")
                tick.set_color("#1D3E2B")

        for j, v in enumerate(vals):
            ax.text(v + 0.018 * xmax, j, f"{v:.2f}", va="center", ha="left",
                    fontsize=7.4,
                    fontweight="bold" if j == len(vals) - 1 else "normal",
                    color="#1D3E2B" if j == len(vals) - 1 else "#222222",
                    zorder=6)

        ax.set_xlabel(xlabel, labelpad=2.5)
        clean_ax(ax, "x")
        panels.append((ax, title))

    place_panel_labels(fig, panels, gap_pt=4.5)
    save_figure(fig, "fig5_overall_comparison", [SRC_SAME_BANK, SRC_PRIMARY_BASELINES, SRC_UNIFIED_T1, SRC_EVENTS_720])


# =============================================================================
# FIG. 6 (Bitbrains E ONLY, 1x2 square log-log paired MSE scatter)
# =============================================================================
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


def build_fig6():
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
    save_figure(fig, "fig6_paired_reliability", [SRC_EVENTS_720, SRC_TASK_SCORES])


# =============================================================================
# FIG. 7 (Bitbrains E ONLY, 1x2 95% VM-Bootstrap CI Forest Plot)
# =============================================================================
def build_fig7_forest():
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
    save_figure(fig, "fig7_robustness_forest", [SRC_EVENTS_720, SRC_TASK_SCORES])


# =============================================================================
# FIG. 8 (Restored Original 3D Feasibility Surface + GnBu Final Model Selection Heatmap!)
# =============================================================================
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


def build_fig8_budget_feasibility():
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
    save_figure(fig, "fig8_budget_feasibility", [SRC_BUDGET_PROF, SRC_SEL_DEC], pdf_dpi=600)


# =============================================================================
# FIG. 9 (Bitbrains E Ablation & Coupling: EXACT Original Soft GnBu Heatmap + Clean Reduction Bars)
# =============================================================================
def build_fig9_coupling():
    mat = pd.read_csv(SRC_COUPLING_MAT)

    sels = [
        ("RA", "Reuse-Aware"),
        ("ZeroStep", "Zero-step-Val"),
        ("MSA", "Ref-Margin"),
        ("FullVal", "Full-Val"),
        ("RGMS", "MSA-DTI Sel."),
    ]
    insts = [
        ("StandardFT", "Standard\nFT"),
        ("EarlyAdapt", "Early\nAdapt"),
        ("OutputCalib", "Output\nCalib."),
        ("SPA", "Full\nAdapt."),
    ]

    grid = np.zeros((len(sels), len(insts)))
    for i, (sk, _) in enumerate(sels):
        for j, (ik, _) in enumerate(insts):
            grid[i, j] = float(mat[(mat["selector"] == sk) & (mat["instantiation"] == ik)]["E_risk"].iloc[0])

    fig = new_canvas(184)

    # Panel (a): Annotated 5x4 Heatmap in EXACT original soft GnBu color palette!
    ax_a = add_ax(fig, 70, 42, 174, 126)
    ax_cb = add_ax(fig, 250, 42, 7.5, 126)
    cmap_gnbu = plt.get_cmap("GnBu")
    norm_r = Normalize(vmin=0.80, vmax=1.20)
    im = ax_a.imshow(np.clip(grid, 0.80, 1.20), cmap=cmap_gnbu, norm=norm_r, aspect="auto", interpolation="nearest")
    ax_a.set_xticks(np.arange(len(insts)), [x[1] for x in insts])
    ax_a.set_yticks(np.arange(len(sels)), [x[1] for x in sels])
    for i in range(len(sels)):
        for j in range(len(insts)):
            v = grid[i, j]
            is_best = (i == len(sels) - 1 and j == len(insts) - 1)
            rgba = cmap_gnbu(norm_r(np.clip(v, 0.80, 1.20)))
            ax_a.text(j, i, f"{v:.3f}", ha="center", va="center",
                      fontsize=7.8, fontweight="bold" if is_best else "normal",
                      color=contrast_text_color(rgba))
    win_rect = plt.Rectangle((len(insts) - 1.5, len(sels) - 1.5), 1, 1, fill=False, edgecolor="#1D3E2B", lw=1.7, zorder=5)
    ax_a.add_patch(win_rect)
    for tick in ax_a.get_yticklabels():
        if "MSA-DTI" in tick.get_text():
            tick.set_fontweight("bold"); tick.set_color("#1D3E2B")
    for tick in ax_a.get_xticklabels():
        if "Full" in tick.get_text():
            tick.set_fontweight("bold"); tick.set_color("#1D3E2B")
    ax_a.tick_params(length=0)
    cb = fig.colorbar(im, cax=ax_cb, ticks=[0.80, 0.90, 1.00, 1.10, 1.20], extend="max")
    cb.ax.tick_params(labelsize=7.3, length=2.2, pad=2)
    cb.set_label("Normalized risk", fontsize=7.8, labelpad=3)

    # Panel (b): Crystal-clear aligned horizontal bar chart of risk reduction from adaptation (%)
    ax_b = add_ax(fig, 314, 42, 138, 126)
    y = np.arange(len(sels))
    pct_red = 100.0 * (1.0 - grid[:, 3] / grid[:, 0])
    bar_cols = ["#78B9AF", "#8DB8B0", "#8DB8B0", "#6E8FB3", FOREST]
    bars = ax_b.barh(y, pct_red, height=0.56, color=bar_cols, edgecolor="#444444", lw=0.5)
    bars[-1].set_hatch("//")
    bars[-1].set_edgecolor("#1D3E2B")
    bars[-1].set_linewidth(0.8)
    for i, pr in enumerate(pct_red):
        ax_b.text(pr + 1.4, i, f"{pr:.1f}%", va="center", ha="left", fontsize=7.5,
                  fontweight="bold" if i == len(sels) - 1 else "normal",
                  color="#1D3E2B" if i == len(sels) - 1 else "#222222")
    ax_b.set_yticks(y, [""] * len(sels))
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 68)
    ax_b.xaxis.set_major_locator(MultipleLocator(15))
    ax_b.set_xlabel("Risk reduction vs. Standard FT (%)", labelpad=2.5)
    clean_ax(ax_b, "x")

    place_panel_labels(fig, [(ax_a, "(a) Selection × adaptation risk (Bitbrains E)"),
                             (ax_b, "(b) Adaptation gain per selector")], gap_pt=4.5)
    save_figure(fig, "fig9_ablation_coupling", [SRC_COUPLING_MAT])


# =============================================================================
# FIG. 10 (Section 5.6 External Evaluation — Bar Comparison WITHOUT Standalone-CPU line!)
# =============================================================================
def build_fig10_external():
    ev = pd.read_csv(SRC_EVENTS_720)
    dep = pd.read_csv(SRC_EXT_DEP)
    az = ev[ev["stage"] == "azure"]
    ufcg = dep[dep["cohort"] == "ufcg"]

    fig = new_canvas(168)

    # Panel (a): Microsoft Azure Public Dataset V2 (30 subscriptions, 360 deployments)
    ax_a = add_ax(fig, 80, 42, 134, 114)
    az_items = [
        ("Zero-step-Val", float(az["ZeroStep__StandardFT"].mean()), "#D9A25F"),
        ("Reuse-Aware", float(az["RA__StandardFT"].mean()), "#C77979"),
        ("Ref-Margin", float(az["MSA__StandardFT"].mean()), "#8DB8B0"),
        ("Full-Val", float(az["FullVal__StandardFT"].mean()), "#6E8FB3"),
        ("MSA-DTI (Ours)", float(az["RGMS__SPA"].mean()), FOREST),
    ]
    y = np.arange(len(az_items))
    vals = [x[1] for x in az_items]
    bars = ax_a.barh(y, vals, height=0.56, color=[x[2] for x in az_items], edgecolor="#444444", lw=0.5)
    bars[-1].set_hatch("//"); bars[-1].set_edgecolor("#1D3E2B"); bars[-1].set_linewidth(0.85)
    for j, v in enumerate(vals):
        ax_a.text(v + 0.03, j, f"{v:.3f}", va="center", ha="left", fontsize=7.4,
                  fontweight="bold" if j == len(vals) - 1 else "normal",
                  color="#1D3E2B" if j == len(vals) - 1 else "#222222")
    ax_a.set_yticks(y, [x[0] for x in az_items])
    ax_a.set_ylim(len(az_items) - 0.35, -0.58)
    ax_a.set_xlim(0, 2.08)
    ax_a.set_xlabel("Three-budget normalized risk", labelpad=2.5)
    for tick in ax_a.get_yticklabels():
        if tick.get_text().startswith("MSA-DTI"):
            tick.set_fontweight("bold"); tick.set_color("#1D3E2B")
    clean_ax(ax_a, "x")

    # Panel (b): UFCG OpenStack 2024-2025 Private Cloud (12 projects, 144 deployments, prospective)
    ax_b = add_ax(fig, 306, 42, 134, 114)
    uf_items = [
        ("Zero-step-Val", float(ufcg["ZeroStep__StandardFT"].mean()), "#D9A25F"),
        ("Ref-Margin", float(ufcg["MSA__StandardFT"].mean()), "#8DB8B0"),
        ("NTK-Q", float(ufcg["NTKQ__StandardFT"].mean()), "#C77979"),
        ("Full-Val", float(ufcg["FullVal__StandardFT"].mean()), "#6E8FB3"),
        ("MSA-DTI (Ours)", float(ufcg["RGMS__SPA"].mean()), FOREST),
    ]
    y2 = np.arange(len(uf_items))
    vals2 = [x[1] for x in uf_items]
    bars2 = ax_b.barh(y2, vals2, height=0.56, color=[x[2] for x in uf_items], edgecolor="#444444", lw=0.5)
    bars2[-1].set_hatch("//"); bars2[-1].set_edgecolor("#1D3E2B"); bars2[-1].set_linewidth(0.85)
    for j, v in enumerate(vals2):
        ax_b.text(v + 0.02, j, f"{v:.3f}", va="center", ha="left", fontsize=7.4,
                  fontweight="bold" if j == len(vals2) - 1 else "normal",
                  color="#1D3E2B" if j == len(vals2) - 1 else "#222222")
    ax_b.set_yticks(y2, [x[0] for x in uf_items])
    ax_b.set_ylim(len(uf_items) - 0.35, -0.58)
    ax_b.set_xlim(0, 1.30)
    ax_b.set_xlabel("Three-budget normalized risk", labelpad=2.5)
    for tick in ax_b.get_yticklabels():
        if tick.get_text().startswith("MSA-DTI"):
            tick.set_fontweight("bold"); tick.set_color("#1D3E2B")
    clean_ax(ax_b, "x")

    place_panel_labels(fig, [(ax_a, "(a) Azure Public Dataset V2 (360 dep.)"),
                             (ax_b, "(b) UFCG 2024–2025 (144 dep.)")], gap_pt=4.5)
    save_figure(fig, "fig10_external_evaluation", [SRC_EVENTS_720, SRC_EXT_DEP])


# =============================================================================
# FIG. 11 (Section 5.6 External Evaluation — Dual Soft GnBu Coupling Heatmaps!)
# =============================================================================
def build_fig11_external_coupling():
    mat = pd.read_csv(SRC_COUPLING_MAT)
    dep = pd.read_csv(SRC_EXT_DEP)
    uf = dep[dep["cohort"] == "ufcg"]

    insts = [
        ("StandardFT", "Standard\nFT"),
        ("EarlyAdapt", "Early\nAdapt"),
        ("OutputCalib", "Output\nCalib."),
        ("SPA", "Full\nAdapt."),
    ]
    az_sels = [
        ("RA", "Reuse-Aware"),
        ("ZeroStep", "Zero-step-Val"),
        ("MSA", "Ref-Margin"),
        ("FullVal", "Full-Val"),
        ("RGMS", "MSA-DTI Sel."),
    ]
    uf_sels = [
        ("ZeroStep", "Zero-step-Val"),
        ("MSA", "Ref-Margin"),
        ("NTKQ", "NTK-Q"),
        ("FullVal", "Full-Val"),
        ("RGMS", "MSA-DTI Sel."),
    ]

    grid_az = np.array([
        [float(mat[(mat["selector"] == sk) & (mat["instantiation"] == ik)]["azure_risk"].iloc[0]) for ik, _ in insts]
        for sk, _ in az_sels
    ])
    grid_uf = np.array([
        [float(uf[f"{sk}__{ik}"].mean()) for ik, _ in insts]
        for sk, _ in uf_sels
    ])

    fig = new_canvas(184)
    cmap_gnbu = plt.get_cmap("GnBu")
    norm_ext = Normalize(vmin=0.65, vmax=1.25)

    ax_a = add_ax(fig, 60, 42, 144, 126)
    ax_b = add_ax(fig, 266, 42, 144, 126)
    ax_cb = add_ax(fig, 417, 42, 7.5, 126)

    for ax, grid, sels, highlight_best in [
        (ax_a, grid_az, az_sels, (len(az_sels) - 1, len(insts) - 1)),
        (ax_b, grid_uf, uf_sels, None),
    ]:
        im = ax.imshow(np.clip(grid, 0.65, 1.25), cmap=cmap_gnbu, norm=norm_ext, aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(len(insts)), [x[1] for x in insts])
        ax.set_yticks(np.arange(len(sels)), [x[1] for x in sels])
        for i in range(len(sels)):
            for j in range(len(insts)):
                v = grid[i, j]
                is_hi = (highlight_best is not None and (i, j) == highlight_best) or (i == len(sels) - 1 and j == len(insts) - 1)
                rgba = cmap_gnbu(norm_ext(np.clip(v, 0.65, 1.25)))
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=7.6, fontweight="bold" if is_hi else "normal",
                        color=contrast_text_color(rgba))
        win_r = plt.Rectangle((len(insts) - 1.5, len(sels) - 1.5), 1, 1, fill=False, edgecolor="#1D3E2B", lw=1.6, zorder=5)
        ax.add_patch(win_r)
        for tick in ax.get_yticklabels():
            if "MSA-DTI" in tick.get_text():
                tick.set_fontweight("bold"); tick.set_color("#1D3E2B")
        for tick in ax.get_xticklabels():
            if "Full" in tick.get_text():
                tick.set_fontweight("bold"); tick.set_color("#1D3E2B")
        ax.tick_params(length=0)

    cb = fig.colorbar(im, cax=ax_cb, ticks=[0.65, 0.80, 0.95, 1.10, 1.25], extend="max")
    cb.ax.tick_params(labelsize=7.3, length=2.2, pad=2)
    cb.set_label("Normalized risk", fontsize=7.8, labelpad=3)

    place_panel_labels(fig, [(ax_a, "(a) Azure Public Dataset V2 (360 dep.)"),
                             (ax_b, "(b) UFCG 2024–2025 (144 dep.)")], gap_pt=4.5)
    save_figure(fig, "fig11_external_coupling", [SRC_COUPLING_MAT, SRC_EXT_DEP])


# =============================================================================
# LATEX TABLES (Main Tables 1--6 + Supplementary Tables)
# Strictly zero semicolons, zero coined acronyms, cas-sc [pos=!htbp] for main
# =============================================================================
def write_table_all(fname: str, content: str):
    assert ";" not in content, f"Semicolon found in {fname}!"
    for bad in ["BC-ARS", "AP-NCI", "(M1)", "(M2)"]:
        assert bad not in content, f"Forbidden token {bad} found in {fname}!"
    for d in TAB_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(content, encoding="utf-8")


def build_tables():
    sb = pd.read_csv(SRC_SAME_BANK)
    nat = pd.read_csv(SRC_PRIMARY_BASELINES)
    t1 = pd.read_csv(SRC_UNIFIED_T1)
    mat_df = pd.read_csv(SRC_COUPLING_MAT)
    ev_df = pd.read_csv(SRC_EVENTS_720)
    timing_sum = json.loads(SRC_TIMING_SUM.read_text(encoding="utf-8"))
    ext_t1 = pd.read_csv(SRC_EXT_T1)
    ext_dep = pd.read_csv(SRC_EXT_DEP)

    # --- Table 1: Datasets & Experimental Configuration ---
    tab1 = r"""\begin{table}[pos=!htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\caption{Datasets, cohort roles, and experimental settings used for primary and external evaluations.}
\label{tab:setup}
\begin{tabularx}{\linewidth}{llllY}
\toprule
Dataset / Cohort & Role & Centers (Tasks) & Observed Channels & Evaluation Protocol \\
\midrule
Bitbrains (Source, A, B) & Candidate bank construction & 60 VMs & 10 raw (25 feat.) & Frozen 19-model bank \\
Bitbrains (C, D) & Historical records / dev & 20 + 20 VMs (160) & 10 raw (25 feat.) & Calibration (C), development (D) \\
Bitbrains (E) & Primary evaluation & 30 VMs (120 $\times$ 3) & 10 raw (25 feat.) & Retrospective evaluation \\
Azure Public Dataset V2 & Cross-platform external & 30 subs (120 $\times$ 3) & 1 CPU (9 missing) & Retrospective evaluation \\
UFCG OpenStack 2024--2025 & Prospective external & 12 proj (48 $\times$ 3) & 1 vCPU (9 missing) & Prospective, fixed method \\
\midrule
\multicolumn{5}{p{\dimexpr\linewidth-2\tabcolsep\relax}}{\textbf{Common settings}: Lookback $L=96$, horizons $H\in\{1,4\}$, support sizes $K\in\{10,20\}$, and update cap $T=50$. \textbf{Deployment budgets} (Ops / Params): Tight ($1.5\times 10^6$ / $3\times 10^4$, 10), Medium ($5\times 10^6$ / $10^5$, 17), Loose ($20\times 10^6$ / $5\times 10^5$, 19).} \\
\bottomrule
\end{tabularx}
\end{table}
"""
    write_table_all("table1_setup.tex", tab1)

    # --- Table 2: Curated Baseline Methods ---
    tab2 = r"""\begin{table}[pos=!htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\caption{Comparison methods and their selection and adaptation rules.}
\label{tab:baselines}
\begin{tabularx}{\linewidth}{lllY}
\toprule
Method & Category & Native Limit Filter & Selection and Adaptation Rule \\
\midrule
PT-FT & Fixed-backbone transfer & No (fixed A39) & Fixed source backbone with standard fine-tuning \\
T-NAS & Differentiable Meta-NAS & No & Meta-initialized weights and architecture decoding on support \\
H-Meta-NAS & Hardware-aware Meta-NAS & Yes & Shared supernet with limit-filtered evolutionary search \\
Reuse-Aware & Reusable model market & No (also filtered) & Rank by distances between encoded specifications \\
PAVE & Proxy model valuation & No (also filtered) & Rank by cosine similarity of probe parameter differences \\
LEAD (Full-NTK) & Transferability estimation & No (also filtered) & Rank estimated responses using the NTK of the backbone and added head \\
Zero-step-Val & Pre-adaptation validation & Yes & Rank feasible candidates on validation before standard fine-tuning \\
Full-Val & Post-adaptation validation & Yes & Fine-tune all feasible candidates and pick lowest validation MSE \\
\textbf{MSA-DTI (Ours)} & \textbf{Selection and adaptation} & \textbf{Yes} & \textbf{Response-guided selection and structure-preserving adaptation} \\
\bottomrule
\end{tabularx}
\end{table}
"""
    write_table_all("table2_baselines.tex", tab2)

    # Helpers for Table 3, Table 6, and Supplement 3-budget tables
    def fmt3(x):
        return f"{x:.3f}"

    def row_6col(label, col2, vals, bold=False):
        def f(v):
            if v is None:
                return "N/F"
            s = fmt3(v)
            return rf"\textbf{{{s}}}" if bold else s
        lbl = rf"\textbf{{{label}}}" if bold else label
        c2 = rf"\textbf{{{col2}}}" if bold else col2
        return f"{lbl} & {c2} & {f(vals[0])} & {f(vals[1])} & {f(vals[2])} & {f(vals[3])} \\\\"

    def extract_sb(cn_name, b_cn):
        r = sb[(sb["method"] == cn_name) & (sb["budget"] == b_cn)].iloc[0]
        return ([float(r["bitbrains_mse_x1000"]), float(r["bitbrains_mae_x1000"]),
                 float(r["bitbrains_worst10_x1000"]), float(r["bitbrains_cvar10_x1000"])],
                [float(r["azure_mse_x1000"]), float(r["azure_mae_x1000"]),
                 float(r["azure_worst10_x1000"]), float(r["azure_cvar10_x1000"])])

    def extract_nat(cn_name, b_cn):
        r = nat[(nat["method"] == cn_name) & (nat["budget"] == b_cn)].iloc[0]
        ef = str(r["bitbrains_feasible"])
        af = str(r["azure_feasible"])
        if str(r["bitbrains_mse_x1000"]) == "N/F":
            return ef, af, [None] * 4, [None] * 4
        return (ef, af,
                [float(r["bitbrains_mse_x1000"]), float(r["bitbrains_mae_x1000"]),
                 float(r["bitbrains_worst10_x1000"]), float(r["bitbrains_cvar10_x1000"])],
                [float(r["azure_mse_x1000"]), float(r["azure_mae_x1000"]),
                 float(r["azure_worst10_x1000"]), float(r["azure_cvar10_x1000"])])

    def extract_unified(method, b_en):
        rE = t1[(t1["stage"] == "E") & (t1["budget"] == b_en) & (t1["method"] == method)].iloc[0]
        rA = t1[(t1["stage"] == "azure") & (t1["budget"] == b_en) & (t1["method"] == method)].iloc[0]
        return ([float(rE["MSE"]) * 1e3, float(rE["MAE"]) * 1e3, float(rE["Worst10"]) * 1e3, float(rE["CVaR10"]) * 1e3],
                [float(rA["MSE"]) * 1e3, float(rA["MAE"]) * 1e3, float(rA["Worst10"]) * 1e3, float(rA["CVaR10"]) * 1e3])

    # --- Table 3: Primary Bitbrains E Results (Medium Budget) ---
    sub_e = ev_df[(ev_df["stage"] == "E") & (ev_df["budget"] == "medium")]
    cpu_e_m = float(sub_e.groupby("VM")["MSE_standalone_cpu"].mean().mean()) * 1e3
    cpu_e_c = float(sub_e["MSE_standalone_cpu"].nlargest(12).mean()) * 1e3

    t3_lines = [
        r"\begin{table}[pos=!htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        rf"\caption{{Prediction accuracy ($\times 10^{{-3}}$, where lower is better) and native deployment feasibility on Bitbrains E (30 VMs, 120 tasks) under the medium deployment budget ($\le 5.0\times 10^6$ Ops, $\le 10^5$ Params, 17 feasible candidates). The non-reuse linear anchor Standalone-CPU achieves MSE $={cpu_e_m:.3f}$ and CVaR-10\% $={cpu_e_c:.3f}$. Feas.\ counts tasks that satisfy both limits before any additional filtering. For rows marked $^{{\dagger}}$, errors cover all 120 tasks after restricting selection to feasible candidates.}}",
        r"\label{tab:main_bitbrains}",
        r"\begin{tabular}{l c rrrr}",
        r"\toprule",
        r"Method & Feas. & MSE & MAE & Worst-10\% & CVaR-10\% \\",
        r"\midrule",
    ]
    for cn, lab in [("PT-FT", "PT-FT"), ("T-NAS-style", "T-NAS"), ("H-Meta-NAS-style（修正）", "H-Meta-NAS")]:
        ef, _, evs, _ = extract_nat(cn, "中")
        t3_lines.append(row_6col(lab, ef, evs))
    for cn, lab in [("Reuse-Aware-style", r"Reuse-Aware$^{\dagger}$"),
                    ("PAVE-style", r"PAVE$^{\dagger}$"),
                    ("LEAD-style（完整NTK）", r"LEAD (Full-NTK)$^{\dagger}$")]:
        ef, _, _, _ = extract_nat(cn, "中")
        evs, _ = extract_sb(cn, "中")
        t3_lines.append(row_6col(lab, ef, evs))
    for cn, lab in [("零步验证选型", "Zero-step-Val"), ("全部适配后验证", "Full-Val")]:
        evs, _ = extract_sb(cn, "中")
        t3_lines.append(row_6col(lab, "120/120", evs))
    evs, _ = extract_unified("MSA-DTI (Full)", "medium")
    t3_lines.append(row_6col("MSA-DTI (Ours)", "120/120", evs, bold=True))
    t3_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write_table_all("table3_main_bitbrains.tex", "\n".join(t3_lines))

    # --- Table 4: Dedicated Hardware Onboarding Cost & Zero Deployment Overhead ---
    sel_by_b = {r["budget"]: r for r in timing_sum["selection_by_budget"]}
    inst = timing_sum["instantiation_mean"]
    inf = timing_sum["inference_mean"]
    tab4 = rf"""\begin{{table}}[pos=!htbp]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{4.5pt}}
\caption{{Isolated single-center onboarding cost and deployed inference latency measured on a dedicated Tesla V100S-32GB GPU and single-thread CPU (mean over 12 center and task cases).}}
\label{{tab:cost}}
\begin{{tabular}}{{lrrrrr}}
\toprule
\multicolumn{{6}}{{l}}{{\textbf{{Part A: One-time model selection cost by budget (wall-clock time and peak GPU VRAM)}}}} \\
\midrule
Budget (Feasible) & Zero-step-Val (s) & Full-Val (s) & \textbf{{MSA-DTI Sel.\ (s)}} & \textbf{{Peak VRAM (MB)}} & \textbf{{Compliant}} \\
\midrule
Tight (10 cand.) & {sel_by_b['tight']['zero_step_sec']:.2f} & {sel_by_b['tight']['full_val_sec']:.2f} & \textbf{{{sel_by_b['tight']['r44_probe_sec']:.2f}}} & \textbf{{{sel_by_b['tight']['r44_vram_mb']:.1f}}} & \textbf{{120/120}} \\
Medium (17 cand.) & {sel_by_b['medium']['zero_step_sec']:.2f} & {sel_by_b['medium']['full_val_sec']:.2f} & \textbf{{{sel_by_b['medium']['r44_probe_sec']:.2f}}} & \textbf{{{sel_by_b['medium']['r44_vram_mb']:.1f}}} & \textbf{{120/120}} \\
Loose (19 cand.) & {sel_by_b['loose']['zero_step_sec']:.2f} & {sel_by_b['loose']['full_val_sec']:.2f} & \textbf{{{sel_by_b['loose']['r44_probe_sec']:.2f}}} & \textbf{{{sel_by_b['loose']['r44_vram_mb']:.1f}}} & \textbf{{120/120}} \\
\midrule
\multicolumn{{6}}{{l}}{{\textbf{{Part B: One-time adaptation cost and deployed inference latency}}}} \\
\midrule
Adaptation Rule & Onboard (s) & Peak VRAM (MB) & $\Delta$Params / $\Delta$Ops & GPU 1-win / b=256 (ms) & CPU 1-win (ms) \\
\midrule
Standard FT & {inst['u50_sec']:.2f} & {inst['u50_vram_mb']:.1f} & 0 / 0 & {inf['u50_gpu_1win_ms']:.3f} / {inf['u50_gpu_batch_ms_per_win']:.3f} & {inf['u50_cpu_1win_ms']:.3f} \\
\textbf{{MSA-DTI}} & \textbf{{{inst['r64_total_sec']:.2f}}} & \textbf{{{inst['r64_vram_mb']:.1f}}} & \textbf{{+0 / +0}} & \textbf{{{inf['r64_gpu_1win_ms']:.3f} / {inf['r64_gpu_batch_ms_per_win']:.3f}}} & \textbf{{{inf['r64_cpu_1win_ms']:.3f}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write_table_all("table4_feasibility_cost.tex", tab4)

    # --- Table 5: Clean 5-Row Leave-One-Out Ablation on Bitbrains E ---
    def ab_metrics(sel_key: str, inst_key: str):
        row_m = mat_df[(mat_df["selector"] == sel_key) & (mat_df["instantiation"] == inst_key)].iloc[0]
        out = {"E_risk": float(row_m["E_risk"]), "A_risk": float(row_m["azure_risk"])}
        for st, pref in [("E", "E"), ("azure", "A")]:
            sub = ev_df[(ev_df["stage"] == st) & (ev_df["budget"] == "medium")]
            m_col = f"MSE_{sel_key}__{inst_key}"
            a_col = f"MAE_{sel_key}__{inst_key}"
            w_col = f"Worst10_{sel_key}__{inst_key}"
            out[f"{pref}_MSE"] = float(sub.groupby("VM")[m_col].mean().mean()) * 1e3
            out[f"{pref}_MAE"] = float(sub.groupby("VM")[a_col].mean().mean()) * 1e3
            out[f"{pref}_W10"] = float(sub.groupby("VM")[w_col].mean().mean()) * 1e3
            out[f"{pref}_CVaR"] = float(sub[m_col].nlargest(12).mean()) * 1e3
        return out

    ab_rows_5 = [
        ("MSA-DTI", "RGMS", "SPA", True),
        ("w/o Response-guided selection", "ZeroStep", "SPA", False),
        ("w/o Hidden-channel routing", "RGMS", "OutputCalib", False),
        ("w/o Output calibration and channel routing", "RGMS", "EarlyAdapt", False),
        ("w/o Structure-preserving adaptation", "RGMS", "StandardFT", False),
    ]

    def build_ablation_tex(pref: str, tab_label: str, caption: str, env_opt: str = "[pos=!htbp]"):
        lines = [
            rf"\begin{{table}}{env_opt}",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{5pt}",
            rf"\caption{{{caption}}}",
            rf"\label{{{tab_label}}}",
            r"\begin{tabular}{l rrrr r}",
            r"\toprule",
            r"Variant & MSE & MAE & Worst-10\% & CVaR-10\% & Norm.\ Risk \\",
            r"\midrule",
        ]
        for v_name, sk, ik, is_full in ab_rows_5:
            m = ab_metrics(sk, ik)
            b0, b1 = (r"\textbf{", "}") if is_full else ("", "")
            lines.append(
                f"{b0}{v_name}{b1} & {b0}{m[f'{pref}_MSE']:.3f}{b1} & {b0}{m[f'{pref}_MAE']:.3f}{b1} & "
                f"{b0}{m[f'{pref}_W10']:.3f}{b1} & {b0}{m[f'{pref}_CVaR']:.3f}{b1} & {b0}{m[f'{pref}_risk']:.4f}{b1} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        return "\n".join(lines)

    tab5_main = build_ablation_tex(
        "E",
        "tab:ablation",
        r"Component-wise ablation study of MSA-DTI on Bitbrains E (30 VMs, 120 medium-budget tasks and 360 three-budget deployments). Medium-budget errors are reported in $\times 10^{-3}$ alongside three-budget normalized risk.",
        "[pos=!htbp]",
    )
    write_table_all("table5_ablation.tex", tab5_main)

    # --- Table 6: Merged Two-Panel External Evaluation (Azure + UFCG 2024--2025) ---
    az_all = ev_df[ev_df["stage"] == "azure"]
    az_med = az_all[az_all["budget"] == "medium"]
    cpu_az_risk = float(az_all["standalone_cpu"].mean())
    cpu_az_mse = float(az_med.groupby("VM")["MSE_standalone_cpu"].mean().mean()) * 1e3
    cpu_az_cvar = float(az_med["MSE_standalone_cpu"].nlargest(12).mean()) * 1e3

    def az_row(lab, col_key, bold=False):
        rk = float(az_all[col_key].mean())
        ms = float(az_med.groupby("VM")[f"MSE_{col_key}"].mean().mean()) * 1e3
        ma = float(az_med.groupby("VM")[f"MAE_{col_key}"].mean().mean()) * 1e3
        w10 = float(az_med.groupby("VM")[f"Worst10_{col_key}"].mean().mean()) * 1e3
        cv = float(az_med[f"MSE_{col_key}"].nlargest(12).mean()) * 1e3
        b0, b1 = (r"\textbf{", "}") if bold else ("", "")
        return f"{b0}{lab}{b1} & {b0}{rk:.4f}{b1} & {b0}{ms:.3f}{b1} & {b0}{ma:.3f}{b1} & {b0}{w10:.3f}{b1} & {b0}{cv:.3f}{b1} \\\\"

    _, _, _, pt_az = extract_nat("PT-FT", "中")
    t6_lines = [
        r"\begin{table}[pos=!htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.8pt}",
        rf"\caption{{Cross-platform external evaluation on Microsoft Azure Public Dataset V2 (30 subscriptions, 120 medium tasks and 360 deployments, errors $\times 10^{{-3}}$) and the zero-tuning prospective UFCG OpenStack 2024--2025 trace (12 projects, 48 medium tasks and 144 deployments, MSE and MAE $\times 10^{{-2}}$). On Azure, Standalone-CPU attains three-budget normalized risk {cpu_az_risk:.4f}, medium MSE {cpu_az_mse:.3f}, and CVaR-10\% {cpu_az_cvar:.3f}. PT-FT violates the tight budget on Azure (0/120 feasible), so its three-budget normalized risk is omitted. Reuse-Aware ($^{{\dagger}}$) reports budget-filtered errors (native feasibility 102/120).}}",
        r"\label{tab:external}",
        r"\begin{tabular}{l r rrrr}",
        r"\toprule",
        r"Method & Norm.\ Risk & MSE & MAE & Worst-10\% & CVaR-10\% \\",
        r"\midrule",
        r"\multicolumn{6}{l}{\textbf{(a) Microsoft Azure Public Dataset V2 (cross-platform benchmark, errors $\times 10^{-3}$)}} \\",
        r"\midrule",
        f"PT-FT & -- & {pt_az[0]:.3f} & {pt_az[1]:.3f} & {pt_az[2]:.3f} & {pt_az[3]:.3f} \\\\",
        az_row(r"Reuse-Aware$^{\dagger}$", "RA__StandardFT"),
        az_row("Zero-step-Val", "ZeroStep__StandardFT"),
        az_row("Ref-Margin", "MSA__StandardFT"),
        az_row("Full-Val", "FullVal__StandardFT"),
        az_row("MSA-DTI (Ours)", "RGMS__SPA", bold=True),
        r"\midrule",
        r"\multicolumn{6}{l}{\textbf{(b) UFCG OpenStack 2024--2025 (zero-tuning prospective cohort, MSE and MAE $\times 10^{-2}$)}} \\",
        r"\midrule",
    ]

    def get_ext_t1(coh, method, bud="medium"):
        return ext_t1[(ext_t1["cohort"] == coh) & (ext_t1["budget"] == bud) & (ext_t1["method"] == method)].iloc[0]

    ext_methods = [
        ("NTK-Q (StandardFT)", "NTK-Q", "NTKQ__StandardFT", False),
        ("Zero-step-val (StandardFT)", "Zero-step-Val", "ZeroStep__StandardFT", False),
        ("Original MSA-DTI (StandardFT)", "Ref-Margin", "MSA__StandardFT", False),
        ("Full-validation (StandardFT)", "Full-Val", "FullVal__StandardFT", False),
        ("standalone_cpu", "Standalone-CPU", "standalone_cpu", False),
        ("MSA-DTI (Full)", "MSA-DTI (Ours)", "RGMS__SPA", True),
    ]
    for key_t1, lab, dep_col, is_ours in ext_methods:
        u_risk = float(ext_dep[ext_dep["cohort"] == "ufcg"][dep_col].mean())
        uu = get_ext_t1("ufcg", key_t1)
        u_mse, u_mae, u_w10, u_cvar = float(uu["MSE"]), float(uu["MAE"]), float(uu["Worst10"]), float(uu["CVaR10"])
        b0, b1 = (r"\textbf{", "}") if is_ours else ("", "")
        t6_lines.append(
            f"{b0}{lab}{b1} & {b0}{u_risk:.4f}{b1} & {b0}{u_mse*1e2:.3f}{b1} & "
            f"{b0}{u_mae*1e2:.3f}{b1} & {b0}{u_w10:.4f}{b1} & {b0}{u_cvar:.4f}{b1} \\\\"
        )
    t6_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write_table_all("table6_external.tex", "\n".join(t6_lines))

    # =========================================================================
    # SUPPLEMENTARY TABLES (Article class [t], zero semicolons, zero acronyms)
    # =========================================================================
    def build_full_3budget_supp(is_azure: bool, tab_label: str, caption_ds: str):
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\supptablefont",
            r"\setlength{\tabcolsep}{5pt}",
            rf"\caption{{Full three-budget prediction accuracy ($\times 10^{{-3}}$) and native deployment feasibility on {caption_ds} across Tight, Medium, and Loose deployment budgets. Feas.\ and $^{{\dagger}}$ follow Table 3 of the main article. N/F indicates incomplete feasibility across the 120 tasks.}}",
            rf"\label{{{tab_label}}}",
            r"\begin{tabular}{l c rrrr}",
            r"\toprule",
            r"Method & Feas. & MSE & MAE & Worst-10\% & CVaR-10\% \\",
        ]
        for b_en, b_cn, b_title in [
            ("tight", "紧", r"Tight Budget ($\le 1.5\times 10^6$ Ops, $\le 3\times 10^4$ Params, 10 feasible)"),
            ("medium", "中", r"Medium Budget ($\le 5.0\times 10^6$ Ops, $\le 10^5$ Params, 17 feasible)"),
            ("loose", "宽", r"Loose Budget ($\le 20\times 10^6$ Ops, $\le 5\times 10^5$ Params, 19 feasible)"),
        ]:
            lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{b_title}}}}} \\")
            lines.append(r"\midrule")
            for cn, lab in [("PT-FT", "PT-FT"), ("T-NAS-style", "T-NAS"), ("H-Meta-NAS-style（修正）", "H-Meta-NAS")]:
                ef, af, evs, avs = extract_nat(cn, b_cn)
                lines.append(row_6col(lab, af if is_azure else ef, avs if is_azure else evs))
            for cn, lab in [("Reuse-Aware-style", r"Reuse-Aware$^{\dagger}$"),
                            ("PAVE-style", r"PAVE$^{\dagger}$"),
                            ("LEAD-style（完整NTK）", r"LEAD (Full-NTK)$^{\dagger}$")]:
                ef, af, _, _ = extract_nat(cn, b_cn)
                evs, avs = extract_sb(cn, b_cn)
                lines.append(row_6col(lab, af if is_azure else ef, avs if is_azure else evs))
            for cn, lab in [("零步验证选型", "Zero-step-Val"), ("全部适配后验证", "Full-Val")]:
                evs, avs = extract_sb(cn, b_cn)
                lines.append(row_6col(lab, "120/120", avs if is_azure else evs))
            evs, avs = extract_unified("MSA-DTI (Full)", b_en)
            lines.append(row_6col("MSA-DTI (Ours)", "120/120", avs if is_azure else evs, bold=True))
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        return "\n".join(lines)

    write_table_all(
        "art_table_supp_3budgets_bitbrains.tex",
        build_full_3budget_supp(False, "tab:supp_3budgets_bitbrains", "Bitbrains E (30 VMs, 120 tasks)"),
    )
    write_table_all(
        "art_table_supp_3budgets_azure.tex",
        build_full_3budget_supp(True, "tab:supp_3budgets_azure", "Azure Public Dataset V2 (30 subscriptions, 120 tasks)"),
    )

    tab_supp_ab_azure = build_ablation_tex(
        "A",
        "tab:supp_ablation_azure",
        r"Component-wise ablation study of MSA-DTI on Microsoft Azure Public Dataset V2 (30 subscriptions, 120 medium-budget tasks and 360 three-budget deployments, medium errors $\times 10^{-3}$).",
        "[t]",
    ).replace(r"\footnotesize", r"\supptablefont")
    write_table_all("art_table_supp_ablation_azure.tex", tab_supp_ab_azure)

    # Bitbrains Rnd6 prospective cohort table (Supplementary)
    rnd_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\supptablefont",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Evaluation on Bitbrains Rnd6 after method freezing (6 VMs, 24 medium tasks and 72 deployments).}",
        r"\label{tab:supp_rnd6}",
        r"\begin{tabular}{l r rrrr}",
        r"\toprule",
        r"Method & Norm.\ Risk & MSE ($\times 10^{-5}$) & MAE ($\times 10^{-3}$) & W-10\% ($\times 10^{-4}$) & CVaR ($\times 10^{-4}$) \\",
        r"\midrule",
    ]
    for key_t1, lab, dep_col, is_ours in ext_methods:
        r_risk = float(ext_dep[ext_dep["cohort"] == "rnd6"][dep_col].mean())
        rr = get_ext_t1("rnd6", key_t1)
        r_mse, r_mae, r_w10, r_cvar = float(rr["MSE"]), float(rr["MAE"]), float(rr["Worst10"]), float(rr["CVaR10"])
        b0, b1 = (r"\textbf{", "}") if is_ours else ("", "")
        rnd_lines.append(
            f"{b0}{lab}{b1} & {b0}{r_risk:.4f}{b1} & {b0}{r_mse*1e5:.3f}{b1} & "
            f"{b0}{r_mae*1e3:.3f}{b1} & {b0}{r_w10*1e4:.3f}{b1} & {b0}{r_cvar*1e4:.3f}{b1} \\\\"
        )
    rnd_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write_table_all("art_table_supp_rnd6.tex", "\n".join(rnd_lines))

    print("OK generated all 6 main tables and 4 supplementary tables cleanly (0 semicolons, 0 coined acronyms).")


def main():
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    build_fig5()
    build_fig6()
    build_fig7_forest()
    build_fig8_budget_feasibility()
    build_fig9_coupling()
    build_fig10_external()
    build_fig11_external_coupling()
    build_tables()
    (OUT_DATA / "asset_audit.json").write_text(json.dumps(AUDIT_RECORDS, indent=2), encoding="utf-8")
    print("ALL PAPER ASSETS GENERATED & AUDITED!")


if __name__ == "__main__":
    main()
