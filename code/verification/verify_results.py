"""Automated integrity and numerical verification for frozen MSA-DTI artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "code" / "data"
PROTO_DIR = REPO_ROOT / "code" / "protocol"


def main() -> int:
    required_files = [
        DATA_DIR / "primary_baselines.csv",
        DATA_DIR / "same_bank_control.csv",
        DATA_DIR / "primary_task_scores.csv",
        DATA_DIR / "unified_benchmark_summary.csv",
        DATA_DIR / "coupling_matrix_risk.csv",
        DATA_DIR / "pairwise_comparisons.csv",
        DATA_DIR / "deployment_events_720.csv.gz",
        DATA_DIR / "selection_decisions.csv.gz",
        DATA_DIR / "timing_summary.json",
        DATA_DIR / "instantiation_timing.csv",
        DATA_DIR / "selection_timing.csv",
        DATA_DIR / "inference_latency.csv",
        DATA_DIR / "external_prospective_summary.csv",
        DATA_DIR / "external_deployments.csv.gz",
        DATA_DIR / "sensitivity_summary.json",
        DATA_DIR / "budget_resource_profiles.json",
        PROTO_DIR / "frozen_config.json",
        PROTO_DIR / "ufcg_cases.csv",
        PROTO_DIR / "rnd6_cases.csv",
    ]
    for p in required_files:
        assert p.exists() and p.stat().st_size > 0, f"Missing artifact: {p}"

    ev = pd.read_csv(DATA_DIR / "deployment_events_720.csv.gz")
    assert len(ev) == 720, f"Expected 720 development deployments, got {len(ev)}"
    assert (ev["stage"] == "E").sum() == 360, "Expected 360 Bitbrains E deployments"
    assert (ev["stage"] == "azure").sum() == 360, "Expected 360 Azure V2 deployments"

    t1 = pd.read_csv(DATA_DIR / "unified_benchmark_summary.csv")
    e_med = t1[(t1["stage"] == "E") & (t1["budget"] == "medium") & (t1["method"] == "MSA-DTI (Full)")].iloc[0]
    assert abs(float(e_med["MSE"]) * 1000.0 - 3.867) < 0.01, f"Unexpected Bitbrains E medium MSE: {e_med['MSE']}"
    assert int(e_med["feasible"]) == 120, f"Expected 120/120 feasibility, got {e_med['feasible']}"

    az_med = t1[(t1["stage"] == "azure") & (t1["budget"] == "medium") & (t1["method"] == "MSA-DTI (Full)")].iloc[0]
    assert abs(float(az_med["MSE"]) * 1000.0 - 1.262) < 0.01, f"Unexpected Azure V2 medium MSE: {az_med['MSE']}"
    assert int(az_med["feasible"]) == 120, f"Expected 120/120 Azure feasibility, got {az_med['feasible']}"

    ext = pd.read_csv(DATA_DIR / "external_prospective_summary.csv")
    ufcg_med = ext[(ext["cohort"] == "ufcg") & (ext["budget"] == "medium") & (ext["method"] == "MSA-DTI (Full)")].iloc[0]
    assert abs(float(ufcg_med["norm_risk"]) - 0.840) < 0.01, f"Unexpected UFCG medium normalized risk: {ufcg_med['norm_risk']}"

    # Import core library to confirm zero syntax or dependency errors
    import sys
    sys.path.insert(0, str(REPO_ROOT / "code"))
    import msa_dti  # noqa: F401
    assert len(msa_dti.CANDIDATE_BANK) == 19

    print("PASS: All 19 frozen datasets, 720 development deployments, prospective metrics, and msa_dti modules verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
