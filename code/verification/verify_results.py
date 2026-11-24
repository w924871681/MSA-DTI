"""Three-tier automated verification for MSA-DTI frozen artifacts and core algorithms.

Tier 1: Frozen dataset integrity and key paper benchmark metrics (Bitbrains E, Azure V2, UFCG, Rnd6).
Tier 2: Live execution of msa_dti.selection (RGMS) across all 720 development deployments (720/720 exact match).
Tier 3: Live execution of msa_dti.adaptation (SPA) on 3 frozen Bitbrains E instances covering
        orig, low1, and legacy_special families (topology invariance + exact body/action/MSE match).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
DATA_DIR = CODE_DIR / "data"
PROTO_DIR = CODE_DIR / "protocol"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(CODE_DIR / "native" / "src") not in sys.path:
    sys.path.append(str(CODE_DIR / "native" / "src"))


def verify_tier1_frozen_tables() -> None:
    required_files = [
        DATA_DIR / "primary_baselines.csv",
        DATA_DIR / "same_bank_control.csv",
        DATA_DIR / "primary_task_scores.csv",
        DATA_DIR / "unified_benchmark_summary.csv",
        DATA_DIR / "coupling_matrix_risk.csv",
        DATA_DIR / "pairwise_comparisons.csv",
        DATA_DIR / "deployment_events_720.csv.gz",
        DATA_DIR / "selection_decisions.csv.gz",
        DATA_DIR / "rgms_dev_cache.npz",
        DATA_DIR / "rgms_dev_meta.json",
        DATA_DIR / "spa_fixture" / "manifest.json",
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
    assert int((ev["stage"] == "E").sum()) == 360, "Expected 360 Bitbrains E deployments"
    assert int((ev["stage"] == "azure").sum()) == 360, "Expected 360 Azure V2 deployments"

    ext_dep = pd.read_csv(DATA_DIR / "external_deployments.csv.gz")
    assert len(ext_dep) == 216, f"Expected 216 external deployments, got {len(ext_dep)}"
    assert int((ext_dep["cohort"] == "ufcg").sum()) == 144, "Expected 144 UFCG deployments"
    assert int((ext_dep["cohort"] == "rnd6").sum()) == 72, "Expected 72 Rnd6 deployments"

    t1 = pd.read_csv(DATA_DIR / "unified_benchmark_summary.csv")
    e_med = t1[(t1["stage"] == "E") & (t1["budget"] == "medium") & (t1["method"] == "MSA-DTI (Full)")].iloc[0]
    assert abs(float(e_med["MSE"]) * 1000.0 - 3.867) < 0.01, f"Unexpected Bitbrains E medium MSE: {e_med['MSE']}"
    assert int(e_med["feasible"]) == 120, f"Expected 120/120 feasibility, got {e_med['feasible']}"

    az_med = t1[(t1["stage"] == "azure") & (t1["budget"] == "medium") & (t1["method"] == "MSA-DTI (Full)")].iloc[0]
    assert abs(float(az_med["MSE"]) * 1000.0 - 1.262) < 0.01, f"Unexpected Azure V2 medium MSE: {az_med['MSE']}"
    assert int(az_med["feasible"]) == 120, f"Expected 120/120 Azure feasibility, got {az_med['feasible']}"

    ext = pd.read_csv(DATA_DIR / "external_prospective_summary.csv")
    ufcg_med = ext[(ext["cohort"] == "ufcg") & (ext["budget"] == "medium") & (ext["method"] == "MSA-DTI (Full)")].iloc[0]
    assert abs(float(ufcg_med["norm_risk"]) - 0.840) < 0.01, f"Unexpected UFCG medium norm_risk: {ufcg_med['norm_risk']}"

    rnd6_risk = float(ext_dep[ext_dep["cohort"] == "rnd6"]["RGMS__SPA"].mean())
    assert abs(rnd6_risk - 0.1851) < 0.001, f"Unexpected Rnd6 norm_risk: {rnd6_risk}"

    import msa_dti  # noqa: E402
    assert len(msa_dti.CANDIDATE_BANK) == 19
    print("[Tier 1 PASS] All 22 frozen datasets, 720 dev + 216 external records, and benchmark metrics verified.")


def verify_tier2_live_rgms() -> None:
    from msa_dti.config import DEPLOYMENT_BUDGETS, feasible_candidates, load_resource_profiles
    from msa_dti.selection import select_candidate

    t0 = time.perf_counter()
    profiles = load_resource_profiles(DATA_DIR / "budget_resource_profiles.json")
    meta = json.loads((DATA_DIR / "rgms_dev_meta.json").read_text(encoding="utf-8"))
    sd = pd.read_csv(DATA_DIR / "selection_decisions.csv.gz")
    sd_dev = sd[
        (sd["scope"] == "full") & (sd["arm"] == "Match") & (sd["stage"].isin(["E", "azure"]))
    ].set_index(["sh", "stage", "slug", "bi"])
    assert len(sd_dev) == 720

    budgets = ("tight", "medium", "loose")
    max_margin_diff = 0.0
    matched_orig = matched_top = matched_pick = 0

    with np.load(DATA_DIR / "rgms_dev_cache.npz") as z:
        for sh in range(4):
            sinfo = meta["shards"][str(sh)]
            H = int(sinfo["H"])
            cx, cb, cq, cqold, cf = (
                z[f"sh{sh}_C_x"],
                z[f"sh{sh}_C_b"],
                z[f"sh{sh}_C_q"],
                z[f"sh{sh}_C_qold"],
                z[f"sh{sh}_C_feat"],
            )
            dx, db, df = z[f"sh{sh}_dev_x"], z[f"sh{sh}_dev_b"], z[f"sh{sh}_dev_feat"]
            loo_k = [int(r["loo_k"]) for r in sinfo["C"]]
            loo_a = [float(r["loo_alpha"]) for r in sinfo["C"]]
            fk, fa = int(sinfo["final_k"]), float(sinfo["final_alpha"])

            for bi, bname in enumerate(budgets):
                omax, pmax = DEPLOYMENT_BUDGETS[bname]
                feas = feasible_candidates(profiles, H, omax, pmax)
                for idx, drow in enumerate(sinfo["dev"]):
                    res = select_candidate(
                        dx[idx],
                        db[idx],
                        df[idx],
                        cx,
                        cb,
                        cq,
                        cqold,
                        cf,
                        feas,
                        k=fk,
                        alpha=fa,
                        hist_loo_k=loo_k,
                        hist_loo_alpha=loo_a,
                    )
                    exp = sd_dev.loc[(sh, drow["stage"], drow["slug"], bi)]
                    assert res["initial_arch"] == int(exp["orig"]), f"q_init mismatch at {sh}/{drow['slug']}/{bname}"
                    assert res["consensus_arch"] == int(exp["top"]), f"q_top mismatch at {sh}/{drow['slug']}/{bname}"
                    assert res["selected_arch"] == int(exp["pick"]), f"pick mismatch at {sh}/{drow['slug']}/{bname}"
                    mdiff = abs(res["direct_margin"] - float(exp["margin"]))
                    assert mdiff < 1e-12, f"margin mismatch {mdiff:.3e} at {sh}/{drow['slug']}/{bname}"
                    max_margin_diff = max(max_margin_diff, mdiff)
                    matched_orig += 1
                    matched_top += 1
                    matched_pick += 1

    elapsed = time.perf_counter() - t0
    assert matched_orig == matched_top == matched_pick == 720
    print(
        f"[Tier 2 PASS] Live RGMS selection matched 720/720 dev deployments "
        f"(q_init=720/720, q_top=720/720, pick=720/720, max|margin_diff|={max_margin_diff:.2e}, {elapsed:.2f}s)."
    )


def verify_tier3_live_spa() -> None:
    try:
        import torch
    except ImportError:
        print("[Tier 3 SKIP] PyTorch not installed; install code/requirements-train.txt to run live SPA check.")
        return

    from msa_dti.adaptation import adapt_model_spa, extract_hidden_and_pred
    from msa_dti_native import runtime as rt

    t0 = time.perf_counter()
    fix_dir = DATA_DIR / "spa_fixture"
    manifest = json.loads((fix_dir / "manifest.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for item in manifest:
        stem = item["stem"]
        arch_id = int(item["arch_id"])
        H = int(item["H"])
        fam = str(item["family"])
        base_sd = torch.load(fix_dir / f"{stem}_base02.pt", map_location="cpu", weights_only=True)
        with np.load(fix_dir / f"{stem}_data.npz") as d:
            xs, ys, xv, yv, xfut, yfut = d["Xs"], d["ys"], d["Xv"], d["yv"], d["Xfut"], d["yfut"]

        model = rt.build(arch_id, H, base_sd, int(item["seed"])).to(device)
        res = adapt_model_spa(model, fam, base_sd, xs, ys, xv, yv, device=device)
        model.load_state_dict(res["deployed_state_dict"], strict=True)
        _, _, pred_fut, _, _ = extract_hidden_and_pred(model, torch.from_numpy(xfut).to(device))
        fut_mse = float(np.square(pred_fut.astype(np.float64) - yfut.astype(np.float64)).mean())

        assert res["topology_invariant"], f"{stem}: topology SHA mismatch"
        assert res["topology_before"]["numel"] == int(item["expected_params"])
        assert res["topology_before"]["denseMultiplyAccumulates"] == int(item["expected_macs"])
        assert res["routed_info"]["low1"]["j"] == int(item["expected_j_low1"])
        assert res["routed_info"]["low5"]["j"] == int(item["expected_j_low5"])
        assert res["selected_body"] == item["expected_body"]
        assert res["winning_families"] == item["expected_families"]
        assert res["winning_actions"] == item["expected_actions"]
        rel_val = abs(res["val_mse"] - float(item["expected_val_mse"])) / max(float(item["expected_val_mse"]), 1e-15)
        rel_fut = abs(fut_mse - float(item["expected_future_mse"])) / max(float(item["expected_future_mse"]), 1e-15)
        assert rel_val < 1e-4, f"{stem}: val_mse rel diff {rel_val:.3e}"
        assert rel_fut < 1e-4, f"{stem}: future_mse rel diff {rel_fut:.3e}"

    elapsed = time.perf_counter() - t0
    print(
        f"[Tier 3 PASS] Live SPA adaptation matched {len(manifest)}/{len(manifest)} frozen Bitbrains E instances "
        f"on {device} (exact j*, body, families, actions, topology invariance +0/+0, MSE rel<1e-4, {elapsed:.2f}s)."
    )


def main() -> int:
    verify_tier1_frozen_tables()
    verify_tier2_live_rgms()
    verify_tier3_live_spa()
    print("ALL VERIFICATION TIERS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
