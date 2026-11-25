"""Plot paired relative reductions with the MSA-DTI selector held fixed."""
import argparse
import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
WIDTH = 164 / 25.4 * 72
HEIGHT = 148
BLUE = "#0072b2"
ORANGE = "#d55e00"
OUTLINE = "#4e6673"


def draw_points(ax, values, baseline):
    # Keep each x coordinate exact. Stack points only when their printed
    # symbols would overlap, packing occupied rows upward as in the reference.
    gap_pt = 3.85
    collision = gap_pt / (WIDTH - 110) * np.diff(ax.get_xlim())[0]
    last_in_row = []
    for value in sorted(values):
        level = next((i for i, prev in enumerate(last_in_row) if value-prev >= collision),
                     len(last_in_row))
        if level == len(last_in_row):
            last_in_row.append(value)
        else:
            last_in_row[level] = value
        marker, color = ("o", BLUE) if value > 0 else ("^", ORANGE) if value < 0 else ("s", "#666666")
        ax.plot(value, baseline + level * gap_pt, marker, ms=3.45 if value >= 0 else 4.4,
                mfc=color, mec=color, mew=0, zorder=3)


def draw_box(ax, values, y):
    q1, med, q3 = np.percentile(values, (25, 50, 75))
    fence = 1.5 * (q3 - q1)
    inside = values[(values >= q1 - fence) & (values <= q3 + fence)]
    ax.hlines(y, inside.min(), inside.max(), colors=OUTLINE, linewidth=.8, zorder=1)
    ax.vlines([inside.min(), inside.max()], y-2.5, y+2.5, colors=OUTLINE, linewidth=.8)
    ax.add_patch(Rectangle((q1, y-4), q3-q1, 8, facecolor="#d5e4e8",
                           edgecolor=OUTLINE, linewidth=.85, zorder=2))
    ax.vlines(med, y-4, y+4, color="#192d36", linewidth=1.5, zorder=3)


def plot(filename):
    with (HERE / "adaptation_pairs.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fig = plt.figure(figsize=(WIDTH/72, HEIGHT/72))
    ax = fig.add_axes([98/WIDTH, 31/HEIGHT, (WIDTH-110)/WIDTH, 94/HEIGHT])
    ax.set_xlim(-40, 104)
    ax.set_ylim(0, 94)
    ax.set_xticks([-40, -20, 0, 20, 40, 60, 80, 100])
    ax.set_yticks([61, 18], ["Azure\n30 subscriptions", "UFCG\n12 projects"])
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=3, width=.75, pad=2.5)
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(.75)
    ax.axvline(0, color="#465965", linestyle="--", linewidth=.78)
    for cohort, baseline in [("Azure", 65), ("UFCG", 22)]:
        values = np.array([float(r["risk_reduction_pct"]) for r in rows if r["cohort"] == cohort])
        draw_points(ax, values, baseline)
        draw_box(ax, values, baseline-10)
    ax.set_xlabel("Reduction in normalized risk with full adaptation (%)", labelpad=2)
    handles = [Line2D([], [], color=BLUE, marker="o", linestyle="none", markersize=5.3,
                      markeredgewidth=0, label="Lower risk"),
               Line2D([], [], color=ORANGE, marker="^", linestyle="none", markersize=5.7,
                      markeredgewidth=0, label="Higher risk")]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=2,
               bbox_to_anchor=(.60, .995), borderaxespad=0, handlelength=.85,
               columnspacing=1.8, handletextpad=.4)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, metadata={"Title":"Paired adaptation effects on Azure and UFCG",
                                   "CreationDate":None, "ModDate":None})
    plt.close(fig)


def render_adaptation_figure(filename):
    with mpl.rc_context({**mpl.rcParamsDefault,
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "font.size": 9.3,
        "pdf.fonttype": 42,
        "axes.unicode_minus": True,
    }):
        plot(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    render_adaptation_figure(parser.parse_args().output)
