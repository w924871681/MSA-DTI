"""Plot maximum deployment-budget utilization and feasibility counts for two methods."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import transforms
from unified_style import canvas, FONT_SIZE


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("MSA-DTI", "PT-FT")
TIERS = ("tight", "medium", "loose")
COLORS = {"MSA-DTI": "#176b80", "PT-FT": "#ad572f"}


def load_groups():
    groups = defaultdict(dict)
    with (ROOT / "data/deployment.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            method, tier = r["method"], r["budget"]
            if method not in METHODS:
                continue
            key = (r["VM"], int(r["H"]), int(r["K"]))
            group = groups[method, tier]
            assert tier in TIERS and key not in group
            assert method != "PT-FT" or r["arch"] == "39"
            p, fops = int(r["parameters"]), int(r["operations"])
            pcap, fcap = int(r["parameter_cap"]), int(r["operation_cap"])
            assert pcap > 0 and fcap > 0
            share = max(p / pcap, fops / fcap)
            assert 0 < share and (share <= 1) == bool(int(r["feasible"]))
            group[key] = share
    tasks = set(groups["MSA-DTI", "tight"])
    assert len(tasks) == 120 and len({k[0] for k in tasks}) == 30
    assert all(set(groups[m, t]) == tasks for m in METHODS for t in TIERS)
    return groups


def main():
    groups = load_groups()
    fig = canvas(169)
    ax = fig.add_axes([.255, .209, .56, .642])
    for x, label, ha in ((.027, "Budget", "left"), (.138, "Method", "left"),
                         (.255 + .56 * 100 / 255, "100% limit", "center"),
                         (.906, "Feasible tasks", "center")):
        fig.text(x, .9, label, ha=ha, fontsize=FONT_SIZE, fontweight="bold", color="#303c43")
    ax.set(xlim=(0, 255), ylim=(-.57, 5.57), yticks=[], xticks=[0, 50, 100, 150, 200, 250])
    ax.set_xlabel("Maximum utilization of either deployment limit (%)", labelpad=4)
    ax.tick_params(axis="x", labelsize=FONT_SIZE, width=.6, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#6f797e")
    ax.axvspan(100, 255, color="#f9f5ef", zorder=-2)
    ax.axvline(100, color="#49545c", ls=(0, (3, 2)), lw=.9, zorder=3)
    side = transforms.blended_transform_factory(fig.transFigure, ax.transData)
    for split in (1.5, 3.5):
        ax.plot([.027, .975], [split, split], transform=side, color="#d6dddf",
                lw=.65, clip_on=False, zorder=0)
    for index, tier in enumerate(TIERS):
        top = 5 - 2 * index
        ax.text(.027, top - .5, tier.capitalize(), transform=side, va="center",
                fontweight="bold", clip_on=False)
        for i, method in enumerate(METHODS):
            values = list(groups[method, tier].values())
            high = 100 * max(values)
            count = sum(v <= 1 for v in values)
            y = top - i
            color = COLORS[method]
            ax.text(.138, y, method, transform=side, va="center", clip_on=False)
            ax.barh(y, high, height=.59, color=color, linewidth=0)
            inside = high >= 55
            ax.annotate(f"{high:.1f}%", (high, y), (-3 if inside else 3, 0), textcoords="offset points",
                        ha="right" if inside else "left", va="center", fontsize=FONT_SIZE,
                        color="white" if inside else "#26353d")
            ax.text(.906, y, f"{count}/{len(values)}", ha="center", va="center", transform=side,
                    color=color if count < len(values) else "#26353d",
                    fontweight="bold" if count < len(values) else "normal", clip_on=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [*fig.texts, *ax.texts, *ax.get_xticklabels(), ax.xaxis.label]
    for label in labels:
        box = label.get_window_extent(renderer)
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1), label.get_text()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"fig9_deployment.{ext}", dpi=350, metadata={"CreationDate":None,"ModDate":None} if ext=="pdf" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
