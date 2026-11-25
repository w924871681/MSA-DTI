"""Render Figs. 5, 9, and 10 from the adjacent frozen numeric inputs.

Usage: python plot_figures.py path/to/latex/figures
The default destination is a figures subdirectory in the current directory.
"""
import argparse
import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from plot_adaptation_pairs import render_adaptation_figure
import numpy as np

ROOT = Path(__file__).resolve().parent
WIDTH = 164 / 25.4 * 72
INK = "#222626"
RULE = "#a8b0b1"
GREEN = "#326b55"
STAGES = ("StandardFT", "EarlyAdapt", "OutputCalib", "SPA")
STAGE_NAMES = ("Standard FT", "Early state selection", "Output calibration", "Full adaptation")
STAGE_COLORS = ("#5a6371", "#a7622a", "#476e98", GREEN)
STAGE_MARKERS = ("s", "^", "o", "D")
PT = 1 / 72


def canvas(height):
    return plt.figure(figsize=(WIDTH * PT, height * PT), dpi=144)


def axes(fig, x, y, w, h):
    fw, fh = fig.get_size_inches() * 72
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh])


def style(ax, grid):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color(RULE)
    ax.tick_params(axis="both", color=RULE, direction="out", length=2.5, width=.6, pad=2.8)
    ax.grid(axis=grid, color="#d8dede", ls=(0, (1.2, 2.1)), lw=.55)
    ax.set_axisbelow(True)


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, metadata={"Title": name[:-4], "CreationDate": None, "ModDate": None})
    plt.close(fig)


def read_records(filename):
    with (ROOT / filename).open(newline="") as stream:
        return list(csv.DictReader(stream))


def main_comparison(folder):
    records = read_records("medium_errors_x1000.csv")
    # Preserve the Table 2 order in every panel. Baseline colors also follow
    # their identities in the previous edition of Fig. 5.
    colors = ["#7995b4", "#d7a264", "#959da9", "#c78484", "#88afa7",
              "#b29bbb", "#aec2d1", "#8a9d8c", "#67917d"]
    fig = canvas(300)
    filtered = {"Reuse-Aware", "PAVE", "LEAD (Full-NTK)"}
    labels = [r["method"] + ("†" if r["method"] in filtered else "") for r in records]
    specs = [("MSE", "(a) MSE", (0, 11.3), [0, 2.5, 5, 7.5, 10]),
             ("MAE", "(b) MAE", (0, 52), [0, 10, 20, 30, 40]),
             ("Worst10", "(c) Worst-10%", (0, 56), [0, 10, 20, 30, 40, 50]),
             ("CVaR10", "(d) CVaR-10%", (0, 92), [0, 20, 40, 60, 80])]
    for i, (metric, title, limits, ticks) in enumerate(specs):
        ax = axes(fig, 73 + (i % 2) * 231, 197 - (i // 2) * 150, 155, 98)
        values = [float(row[metric]) for row in records]
        y = np.arange(len(values))
        bars = ax.barh(y, values, color=colors, height=.61, edgecolor="#42484b", linewidth=.43)
        bars[-1].set(hatch="//", edgecolor="#284d39", linewidth=.6)
        ax.set_ylim(len(values) - .5, -.5)
        ax.set_yticks(y, labels)
        ax.get_yticklabels()[-1].set(fontweight="bold", color="#214c36")
        ax.set_xlim(limits)
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.set_xlabel(r"Error ($\times 10^{-3}$)", labelpad=1.7)
        ax.annotate(title, xy=(0.5, 0), xycoords="axes fraction",
                    xytext=(0, -38), textcoords="offset points",
                    ha="center", va="center", fontsize=9)
        style(ax, "x")
        ax.tick_params(axis="y", length=0)
        for idx, value in enumerate(values):
            ours = idx == len(values) - 1
            ax.annotate(f"{value:.3f}", (value, idx), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=7.7,
                        fontweight="bold" if ours else "normal", color="#214c36" if ours else INK)
    save(fig, folder, "fig5_overall_comparison.pdf")


def get_risk():
    data = read_records("normalized_risk.csv")
    return {(r["cohort"], r["selector"], r["setting"]): float(r["normalized_risk"]) for r in data}


def bitbrains_ablation(folder, risk):
    selectors = ["Reuse-Aware", "Zero-step-Val", "Ref-Margin", "Full-Val", "MSA-DTI selector"]
    fig = canvas(172)
    left = axes(fig, 83, 45, 148, 111)
    right = axes(fig, 325, 45, 134, 111)
    order = np.arange(len(selectors), dtype=float)
    legs = []
    for k, (stage, name, color, marker) in enumerate(zip(STAGES, STAGE_NAMES, STAGE_COLORS, STAGE_MARKERS)):
        x = [risk["Bitbrains E", sel, stage] for sel in selectors]
        y = order + (k - 1.5) * .17
        stage_points, = left.plot(x, y, ls="none", marker=marker, markersize=4.2,
                                  markeredgewidth=.75, markeredgecolor=color,
                                  markerfacecolor=color if k in (0, 3) else "white", label=name)
        legs.append(stage_points)
    fig.legend(handles=legs, loc="upper center", bbox_to_anchor=(.5, .997), ncol=4,
               handletextpad=.3, columnspacing=2.1, borderaxespad=0, frameon=False)
    left.set(xlim=(.78, 2.085), xticks=[.8, 1.1, 1.4, 1.7, 2.0], xlabel="Normalized risk")
    right.set(xlim=(0, 68), xticks=[0, 15, 30, 45, 60], xlabel="Risk reduction vs. standard FT (%)")
    panels = [(left, "(a) Risk by adaptation setting"), (right, "(b) Gain from full adaptation")]
    for ax, label in panels:
        ax.annotate(label, xy=(0.5, 0), xycoords="axes fraction",
                    xytext=(0, -38), textcoords="offset points",
                    ha="center", va="center", fontsize=9)
        ax.set_yticks(order, selectors)
        ax.set_ylim(len(selectors) - .5, -.5)
        ax.set_ylabel("Selector", labelpad=2)
        ax.get_yticklabels()[-1].set(fontweight="bold", color="#214c36")
        style(ax, "x")
        ax.tick_params(axis="y", length=0)
        for j in range(4):
            ax.axhline(j + .5, lw=.4, color="#e0e5e5", zorder=0)
    for j, selector in enumerate(selectors):
        gain = 100 * (1 - risk["Bitbrains E", selector, "SPA"] / risk["Bitbrains E", selector, "StandardFT"])
        right.hlines(j, 0, gain, color="#83968d", lw=1.2)
        right.plot(gain, j, "D", ms=4.7, color=GREEN, markeredgecolor="#244f3e", markeredgewidth=.6)
        right.annotate(f"{gain:.1f}%", (gain, j), xytext=(4.8, 0), textcoords="offset points",
                       va="center", fontsize=7.8)
    save(fig, folder, "fig9_ablation_coupling.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", type=Path, nargs="?", default=Path("figures"))
    output = parser.parse_args().figures
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "font.size": 8.5, "axes.titlesize": 9.2, "axes.titleweight": "bold",
        "axes.labelsize": 8.6, "xtick.labelsize": 8.1, "ytick.labelsize": 8.3,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.linewidth": .6, "hatch.linewidth": .5,
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "none", "legend.fontsize": 8.4,
    })
    main_comparison(output)
    frozen_risk = get_risk()
    bitbrains_ablation(output, frozen_risk)
    render_adaptation_figure(output / "fig10_external_adaptation.pdf")
