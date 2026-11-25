"""Rebuild the per-unit comparison from the original frozen MSA-DTI archive."""
from pathlib import Path
from zipfile import ZipFile
import argparse
import pandas as pd

HERE = Path(__file__).resolve().parent
STANDARD = "RGMS__StandardFT"
FULL = "RGMS__SPA"


def collect(archive, filename, filter_key, filter_value, unit_key, cohort, unit_count):
    with archive.open("code/data/" + filename) as file:
        records = pd.read_csv(file, compression="gzip")
    selected = records[records[filter_key] == filter_value]
    settings = ["H", "K", "budget"]
    groups = selected.groupby(unit_key, sort=True)
    if (len(groups) != unit_count or not groups.size().eq(12).all()
            or len(selected[settings].drop_duplicates()) != 12
            or selected.duplicated([unit_key, *settings]).any()
            or selected[[STANDARD, FULL]].isna().any().any()):
        raise ValueError(f"Unexpected task schedule for {cohort}")
    means = groups[[STANDARD, FULL]].mean()
    means.columns = ["standard_ft_risk", "full_adaptation_risk"]
    if not means.standard_ft_risk.gt(0).all():
        raise ValueError(f"Non-positive reference risk in {cohort}")
    means["risk_reduction_pct"] = 100 * (1 - means.full_adaptation_risk / means.standard_ft_risk)
    means["deployments"] = groups.size()
    means = means.rename_axis("unit_id").reset_index()
    means.insert(0, "cohort", cohort)
    return means


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    with ZipFile(args.archive) as archive:
        azure = collect(archive, "deployment_events_720.csv.gz", "stage", "azure", "VM", "Azure", 30)
        ufcg = collect(archive, "external_deployments.csv.gz", "cohort", "ufcg", "project", "UFCG", 12)
    pd.concat([azure, ufcg], ignore_index=True).to_csv(HERE / "adaptation_pairs.csv", index=False)
