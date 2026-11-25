"""Check the manuscript plot inputs against an existing public MSA-DTI checkout."""
from pathlib import Path
import argparse
import hashlib
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
MANUSCRIPT = ROOT / "src/manuscript"
sys.path.insert(0, str(MANUSCRIPT))
from collect_adaptation_pairs import collect
from extract_values import medium_errors, risk_estimates


class Checkout:
    """Expose the same binary input streams as the original experiment zip."""
    def __init__(self, root):
        self.root = root

    def open(self, relative_path):
        return (self.root / relative_path).open("rb")


def same_table(generated, stored):
    expected = pd.read_csv(stored, float_precision="round_trip")
    generated = generated.copy()
    for column in expected.select_dtypes(include="number"):
        generated[column] = pd.to_numeric(generated[column], errors="raise")
    pd.testing.assert_frame_equal(generated.reset_index(drop=True), expected,
                                  check_exact=False, rtol=1e-12, atol=1e-14)


def verify(checkout_path):
    archive = Checkout(checkout_path)
    risks = risk_estimates(archive)
    same_table(medium_errors(archive), MANUSCRIPT / "medium_errors_x1000.csv")
    same_table(risks, MANUSCRIPT / "normalized_risk.csv")
    azure = collect(archive, "deployment_events_720.csv.gz", "stage", "azure", "VM", "Azure", 30)
    ufcg = collect(archive, "external_deployments.csv.gz", "cohort", "ufcg", "project", "UFCG", 12)
    same_table(pd.concat([azure, ufcg], ignore_index=True), MANUSCRIPT / "adaptation_pairs.csv")

    for local in sorted((ROOT / "src/bitbrains/data").iterdir()):
        upstream = checkout_path / "code/data" / local.name
        if hashlib.sha256(local.read_bytes()).digest() != hashlib.sha256(upstream.read_bytes()).digest():
            raise ValueError(f"Frozen input differs from checkout: {local.name}")

    ref = pd.read_csv(ROOT / "Supplementary_Data_S1.csv", float_precision="round_trip")
    keys = ["cohort", "deployments", "selector"]
    stages = {"StandardFT": "standard_fine_tuning_normalized_risk",
              "EarlyAdapt": "early_state_selection_normalized_risk",
              "OutputCalib": "output_calibration_normalized_risk",
              "SPA": "full_adaptation_normalized_risk"}
    wide = risks[risks.cohort.isin(["Azure", "UFCG"])].pivot(index=keys, columns="setting",
                                                            values="normalized_risk")
    got = wide.rename(columns=stages).sort_index()[list(stages.values())]
    got.columns.name = None
    expected = ref.set_index(keys).sort_index()[list(stages.values())]
    pd.testing.assert_frame_equal(got, expected, check_exact=False, rtol=1e-12, atol=1e-14)
    print("PASS: 3 derived CSVs, 4 frozen inputs, and all 40 risk entries in Supplementary Data S1.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    verify(parser.parse_args().checkout.resolve())
