"""Summarize newly fitted target models using their scores and architecture resource profiles."""
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inputs import CODE, METHODS, TIERS, cases, read
from plotting.unified_style import style
from verify_results import KEY, METRICS


def load_scores(directory, partial):
    frames = []
    for dataset, stage in (("bitbrains", "E"), ("azure", "azure")):
        files = sorted(directory.glob(f"{dataset}-*.csv"))
        if not files:
            if partial:
                continue
            raise ValueError(f"No {dataset} target results in {directory}")
        rows = pd.concat([pd.read_csv(f, usecols=[*KEY, "arch", *METRICS]) for f in files],
                         ignore_index=True)
        if rows.empty or rows.isna().any().any() or rows.duplicated(KEY).any():
            raise ValueError(f"{dataset}: empty, missing or duplicate task records")
        if not np.isfinite(rows[METRICS]).all().all() or (rows[METRICS] < 0).any().any():
            raise ValueError(f"{dataset}: expected finite, nonnegative errors")
        if not rows.arch.isin(range(66)).all():
            raise ValueError(f"{dataset}: unknown architecture ID")
        manifest = {(c["VM"], c["H"], c["K"]) for c in cases(stage)}
        tasks = set(rows[["VM", "H", "K"]].itertuples(index=False, name=None))
        methods = set(rows.method)
        if not tasks <= manifest or not methods <= set(METHODS.values()) or set(rows.budget) != set(TIERS):
            raise ValueError(f"{dataset}: unexpected task, method or budget")
        if len(rows) != len(tasks) * len(methods) * len(TIERS):
            raise ValueError(f"{dataset}: each method and budget must cover the same tasks")
        if not partial and (tasks != manifest or methods != set(METHODS.values())):
            raise ValueError(f"{dataset}: full reports need eight methods and 120 tasks; use --partial for a subset")
        frames.append(rows.assign(dataset=dataset))
    if not frames:
        raise ValueError(f"No target CSVs in {directory}")
    return pd.concat(frames, ignore_index=True)


def add_resources(rows):
    # The two datasets use the same 66 architectures, horizons and deployment limits.
    spec = read(CODE / "data/azure_external/resources.json")
    profile = [spec["profiles"][str(int(h))][str(int(i))] for h, i in zip(rows.H, rows.arch)]
    rows[["parameters", "operations"]] = profile
    rows[["parameter_cap", "operation_cap"]] = [spec["caps"][tier] for tier in rows.budget]
    rows["feasible"] = (rows.parameters <= rows.parameter_cap) & (rows.operations <= rows.operation_cap)


def summarize(rows):
    records = []
    for (dataset, method, tier), group in rows.groupby(["dataset", "method", "budget"], sort=False):
        feasible = group.loc[group.feasible]
        rec = dict(dataset=dataset, method=method, budget=tier, tasks=len(group), feasible=len(feasible))
        # As in the paper, average task errors within each VM, then weight VMs equally.
        for prefix, sample in (("", group), ("feasible_", feasible)):
            means = sample.groupby("VM")[METRICS].mean().mean()
            rec.update({prefix + key: means[key] for key in METRICS})
            tail = math.ceil(len(sample) / 10)
            rec[prefix + "CVaR90"] = sample.MSE.nlargest(tail).mean()
        records.append(rec)
    return pd.DataFrame(records)


def plot(summary, dest):
    style()
    datasets = summary.dataset.unique()
    fig, axes = plt.subplots(len(datasets), 2, figsize=(9, 1.5 + len(datasets) * 3.2),
                             squeeze=False, layout="constrained")
    for (error, coverage), dataset in zip(axes, datasets):
        group = summary.loc[summary.dataset == dataset]
        names = [m for m in METHODS.values() if m in set(group.method)]
        n_tasks = int(group.tasks.iloc[0])
        pos = np.arange(len(names))
        for offset, (tier, color, marker) in enumerate(zip(TIERS, ["#0072B2", "#D55E00", "#009E73"], ["o", "s", "^"])):
            block = group.loc[group.budget == tier].set_index("method").loc[names]
            y = pos + (offset - 1) * .22
            error.scatter(block.MSE, y, label=tier.title(), color=color, marker=marker, s=23)
            coverage.scatter(100 * block.feasible / block.tasks, y, color=color, marker=marker, s=23)
        for ax in (error, coverage):
            ax.set_yticks(pos, names)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=.25)
        error.set_title(f"{dataset.title()} · {n_tasks} task{'s' if n_tasks != 1 else ''} per budget")
        error.set_xlabel("Mean squared error (all outputs)")
        coverage.set_xlabel("Outputs meeting both limits (%)")
        coverage.set_xlim(-4, 104)
        coverage.legend(*error.get_legend_handles_labels(), loc="lower center",
                        bbox_to_anchor=(.5, 1), ncol=3, frameon=False)
    fig.savefig(dest)
    plt.close(fig)


def report(directory, partial=False):
    directory = Path(directory).resolve()
    rows = load_scores(directory, partial)
    add_resources(rows)
    summary = summarize(rows)
    out = directory / "report"
    out.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=out) as temp:
        stage = Path(temp)
        rows.to_csv(stage / "task_scores.csv", index=False)
        summary.to_csv(stage / "performance_and_feasibility.csv", index=False)
        plot(summary, stage / "performance_and_feasibility.pdf")
        for path in stage.iterdir():
            os.replace(path, out / path.name)
    print(f"Reported {len(rows)} target records in {out}")
