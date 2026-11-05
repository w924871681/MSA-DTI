"""Plot MSA-DTI decisions for the same Azure tasks under three deployment budgets."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

import unified_style as style

TIERS = ("tight", "medium", "loose")
DATA = style.DATA / "azure_external/budget_paths_19.csv"
OUTPUT = style.ROOT / "figures/fig10_azure_budget.pdf"


def read_json(path):
    return json.loads(path.read_bytes())


def load_run(directory):
    protocol = read_json(directory / "protocol.json")
    bank = read_json(directory / "inputs/bank.json")
    cases = read_json(directory / "inputs/cases.json")
    tasks = {(r["VM"], str(r["H"]), str(r["K"])) for r in cases}
    summary = directory / "outputs/verified_summary.json"
    receipt = read_json(directory / "COMPLETE.json")
    assert hashlib.sha256(summary.read_bytes()).hexdigest() == receipt["summary_sha256"]
    assert read_json(summary)["status"] == "PASS"
    assert len(tasks) == len(cases) == 120 and len({k[0] for k in tasks}) == 30
    assert len({r["project_id"] for r in cases}) == 30 and len(protocol["bank"]) == 19
    limits = protocol["budgets"]
    bank_ids = set(protocol["bank"])
    eligible = {}
    for tier in TIERS:
        p, ops = limits[tier]
        by_h = [{int(a) for a, v in bank[h].items()
                 if v["parameters"] <= p and v["operations"] <= ops} for h in ("1", "4")]
        assert by_h[0] == by_h[1]
        eligible[tier] = by_h[0]
    assert eligible["tight"] <= eligible["medium"] <= eligible["loose"] == bank_ids
    minimum = {a: next(t for t in TIERS if a in eligible[t]) for a in bank_ids}
    selected = {key: {} for key in tasks}
    with (directory / "outputs/task_metrics.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] != "msa":
                continue
            key = row["VM"], row["H"], row["K"]
            tier, arch = row["budget"], int(row["arch"])
            assert tier not in selected[key] and arch in eligible[tier]
            assert int(row["feasible"]) == len(eligible[tier])
            selected[key][tier] = (arch, row["MSE"], int(row["n_valid"]))
    records = []
    # Keep the original test manifest order, which predates the budget decisions.
    for case in cases:
        key = case["VM"], str(case["H"]), str(case["K"])
        choices = selected[key]
        record = dict(zip(("vm", "h", "k"), key))
        assert len({v[2] for v in choices.values()}) == 1
        for tier in TIERS:
            arch, mse, _ = choices[tier]
            record.update({tier: arch, f"min_{tier}": minimum[arch],
                           f"feasible_{tier}": len(eligible[tier]), f"mse_{tier}": mse})
        records.append(record)
    return records


def paired_changes(records):
    assert len(records) == len({(r["vm"], r["h"], r["k"]) for r in records}) == 120
    samples = []
    for i, (lower, higher) in enumerate(zip(TIERS, TIERS[1:])):
        changes = []
        for row in records:
            before, after = (float(row[f"mse_{tier}"]) for tier in (lower, higher))
            assert math.isfinite(before) and math.isfinite(after) and before > 0 and after >= 0
            if row[lower] == row[higher]:
                assert before == after
            else:
                # In this run, each switch selected a previously ineligible architecture.
                assert TIERS.index(row[f"min_{higher}"]) > i
                changes.append(100 * (before - after) / before)
        samples.append(np.asarray(changes))
    return samples


def draw(records):
    samples = paired_changes(records)
    fig = style.canvas(148)
    ax = style.axis(fig, 91, 33, style.WIDTH - 100, 91)
    ax.set_xlim(-130, 100)
    ax.set_ylim(-.47, 1.65)
    ax.set_xticks([-125, -100, -75, -50, -25, 0, 25, 50, 75, 100])
    ax.set_xlabel("Reduction in test MSE at the looser budget (%)", labelpad=2)
    ax.axvline(0, color="#545D63", lw=.85, ls="--", zorder=1)
    labels = []
    for y, name, values in zip((1, 0), ("Tight vs. medium", "Medium vs. loose"), samples):
        # An exact-value strip sits just above its box. Offsets resolve close points
        # without moving any measured value along the horizontal axis.
        order = np.argsort(values, kind="stable")
        placed = []
        for idx in order:
            x = values[idx]
            nearby = {level for px, level in reversed(placed) if x - px < 2.65}
            level = next(k for k in range(len(nearby) + 1) if k not in nearby)
            placed.append((x, level))
            ax.scatter(x, y + .09 + .065 * level, s=9 if x >= 0 else 16,
                       color=style.BLUE if x >= 0 else style.ORANGE,
                       marker="o" if x >= 0 else "^", edgecolors="none", zorder=3)
        ax.boxplot([values], positions=[y - .135], vert=False, widths=.16,
                   whis=1.5, showfliers=False, patch_artist=True, manage_ticks=False,
                   boxprops={"facecolor": "#D5E4EA", "edgecolor": "#455861", "linewidth": .8},
                   medianprops={"color": "#182C36", "linewidth": 1.4},
                   whiskerprops={"color": "#536770", "linewidth": .75},
                   capprops={"color": "#536770", "linewidth": .75})
        labels.append(f"{name}\n{len(values)}/{len(records)} switched")
    ax.set_yticks([1, 0], labels)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=3, width=.65, pad=3)
    handles = [Line2D([], [], marker=m, color=c, markersize=4.8, linestyle="none", label=label)
               for m, c, label in (("o", style.BLUE, "Lower test MSE"), ("^", style.ORANGE, "Higher test MSE"))]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, 1.045), ncol=2,
              frameon=False, columnspacing=1.7, handlelength=.9, handletextpad=.35, borderpad=0)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-run", type=Path, help="Rebuild the paired CSV from an Azure run.")
    parser.add_argument("--preview", type=Path, help="Also write a PNG preview.")
    args = parser.parse_args()
    if args.extract_run:
        records = load_run(args.extract_run)
        with DATA.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0])
            writer.writeheader()
            writer.writerows(records)
    else:
        with DATA.open(encoding="utf-8", newline="") as f:
            records = list(csv.DictReader(f))
    fig = draw(records)
    fig.savefig(OUTPUT, metadata={"CreationDate": None, "ModDate": None})
    if args.preview:
        fig.savefig(args.preview, dpi=250)
    style.plt.close(fig)
