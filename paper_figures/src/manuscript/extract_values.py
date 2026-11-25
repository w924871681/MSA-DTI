"""Extract the plotted estimates from the frozen MSA-DTI experiment archive.

Usage: python extract_values.py MSA-DTI-experiment-data-and-scripts.zip
Overwrites the two plotting CSVs in this directory.
"""
import argparse
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parent
METRICS = ["MSE", "MAE", "Worst10", "CVaR10"]
STAGES = ["StandardFT", "EarlyAdapt", "OutputCalib", "SPA"]
KEY_TO_NAME = {
    "RA": "Reuse-Aware",
    "ZeroStep": "Zero-step-Val",
    "MSA": "Ref-Margin",
    "NTKQ": "NTK-Q",
    "FullVal": "Full-Val",
    "RGMS": "MSA-DTI selector",
}


def read_csv(archive, name):
    compressed = name.endswith(".gz")
    with archive.open("code/data/" + name) as stream:
        return pd.read_csv(stream, compression="gzip" if compressed else None)


def exactly_one(frame, **fields):
    matches = pd.Series(True, index=frame.index)
    for key, value in fields.items():
        matches &= frame[key] == value
    selected = frame.loc[matches]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {fields}. Found {len(selected)}.")
    return selected.iloc[0]


def medium_errors(archive):
    native = read_csv(archive, "primary_baselines.csv")
    shared = read_csv(archive, "same_bank_control.csv")
    unified = read_csv(archive, "unified_benchmark_summary.csv")
    native_file = "primary_baselines.csv"
    shared_file = "same_bank_control.csv"
    method_rows = [
        ("PT-FT", native_file, "PT-FT"),
        ("T-NAS", native_file, "T-NAS-style"),
        ("H-Meta-NAS", native_file, "H-Meta-NAS-style（修正）"),
        ("Reuse-Aware", shared_file, "Reuse-Aware-style"),
        ("PAVE", shared_file, "PAVE-style"),
        ("LEAD (Full-NTK)", shared_file, "LEAD-style（完整NTK）"),
        ("Zero-step-Val", shared_file, "零步验证选型"),
        ("Full-Val", shared_file, "全部适配后验证"),
        ("MSA-DTI (Ours)", "unified_benchmark_summary.csv", "MSA-DTI (Full)"),
    ]
    rows = []
    for method, filename, registered_name in method_rows:
        if filename == "unified_benchmark_summary.csv":
            row = exactly_one(unified, stage="E", budget="medium", method=registered_name)
            errors = {metric: row[metric] * 1000 for metric in METRICS}
        else:
            source = shared if filename == shared_file else native
            row = exactly_one(source, budget="中", method=registered_name)
            errors = {metric: row["bitbrains_" + metric.lower() + "_x1000"] for metric in METRICS}
        rows.append({"method": method, "cohort": "Bitbrains E", "budget": "medium",
                     "tasks": 120, "source_csv": filename, **errors})
    return pd.DataFrame(rows)


def risk_estimates(archive):
    coupling = read_csv(archive, "coupling_matrix_risk.csv")
    events = read_csv(archive, "external_deployments.csv.gz")
    prospective = events[events["cohort"] == "ufcg"]
    if len(prospective) != 144 or not prospective.groupby("project").size().eq(12).all():
        raise ValueError("Unexpected number of UFCG projects or deployments.")
    development = ["RA", "ZeroStep", "MSA", "FullVal", "RGMS"]
    external = ["ZeroStep", "MSA", "NTKQ", "FullVal", "RGMS"]
    rows = []
    for cohort, recorded in [("Bitbrains E", development), ("Azure", development), ("UFCG", external)]:
        for key in recorded:
            for stage in STAGES:
                if cohort == "UFCG":
                    risk = prospective[key + "__" + stage].mean()
                    source = "external_deployments.csv.gz"
                    deployments = 144
                else:
                    column = "E_risk" if cohort == "Bitbrains E" else "azure_risk"
                    row = exactly_one(coupling, selector=key, instantiation=stage)
                    risk = row[column]
                    source = "coupling_matrix_risk.csv"
                    deployments = 360
                rows.append({"cohort": cohort, "selector": KEY_TO_NAME[key], "setting": stage,
                             "normalized_risk": risk, "deployments": deployments, "source_csv": source})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    with ZipFile(args.archive) as archive:
        medium_errors(archive).to_csv(ROOT / "medium_errors_x1000.csv", index=False)
        risk_estimates(archive).to_csv(ROOT / "normalized_risk.csv", index=False)
