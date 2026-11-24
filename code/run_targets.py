"""Run frozen MSA-DTI (RGMS selection + SPA structure-preserving adaptation) on target centers.

Uses the frozen RGMS candidate selection cache (or newly generated response probes)
together with the exact frozen SPA pipeline (`base02` 5-state early adaptation,
22-config support-only initial readout, optional single-channel causal load routing,
and multi-scale load-decomposed output calibration folded into the existing linear head).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(CODE_DIR / "native" / "src") not in sys.path:
    sys.path.append(str(CODE_DIR / "native" / "src"))

from inputs import TIERS, WORK, carrier, cases, digest, metrics, read, truth  # noqa: E402
from msa_dti.adaptation import adapt_model_spa, early_stop_base02, extract_hidden_and_pred  # noqa: E402
from msa_dti.config import (  # noqa: E402
    CANDIDATE_BANK,
    DEPLOYMENT_BUDGETS,
    HORIZON_SUPPORT_GRID,
    feasible_candidates,
    load_resource_profiles,
)
from msa_dti.selection import select_candidate  # noqa: E402
from msa_dti_native import runtime as rt  # noqa: E402
from msa_dti_native.features import causal_features  # noqa: E402
from msa_dti_native.prefix import DTYPE, prefix_selection  # noqa: E402

DATA_DIR = CODE_DIR / "data"
PROTO_DIR = CODE_DIR / "protocol"


def load_source_state(weights_dir: Path, horizon: int, arch_id: int) -> dict[str, torch.Tensor]:
    """Load source-trained weights from either H{H}_A{id}.pt or strong_H{H}_A{id}.pt."""
    candidates = [
        weights_dir / f"H{horizon}_A{arch_id}.pt",
        weights_dir / "msa" / f"strong_H{horizon}_A{arch_id}.pt",
        weights_dir / f"strong_H{horizon}_A{arch_id}.pt",
        weights_dir / f"H{horizon}" / f"A{arch_id}.pt",
    ]
    for path in candidates:
        if path.is_file():
            obj = torch.load(path, map_location="cpu", weights_only=True)
            return obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    raise FileNotFoundError(
        f"Source checkpoint for H={horizon}, A{arch_id} not found under {weights_dir}"
    )


def select_rgms_from_cache(stage: str, slug: str, horizon: int, support_k: int, budget: str) -> dict:
    """Run exact frozen RGMS selection for one (stage, slug, budget) using rgms_dev_cache.npz."""
    meta = read(DATA_DIR / "rgms_dev_meta.json")
    profiles = load_resource_profiles(DATA_DIR / "budget_resource_profiles.json")
    sh = HORIZON_SUPPORT_GRID.index((int(horizon), int(support_k)))
    sinfo = meta["shards"][str(sh)]
    dev_idx = next(
        idx for idx, row in enumerate(sinfo["dev"]) if row["stage"] == stage and row["slug"] == slug
    )
    omax, pmax = DEPLOYMENT_BUDGETS[budget]
    feas = feasible_candidates(profiles, horizon, omax, pmax)
    loo_k = [r["loo_k"] for r in sinfo["C"]]
    loo_a = [r["loo_alpha"] for r in sinfo["C"]]

    with np.load(DATA_DIR / "rgms_dev_cache.npz") as z:
        return select_candidate(
            z[f"sh{sh}_dev_x"][dev_idx],
            z[f"sh{sh}_dev_b"][dev_idx],
            z[f"sh{sh}_dev_feat"][dev_idx],
            z[f"sh{sh}_C_x"],
            z[f"sh{sh}_C_b"],
            z[f"sh{sh}_C_q"],
            z[f"sh{sh}_C_qold"],
            z[f"sh{sh}_C_feat"],
            feas,
            k=sinfo["final_k"],
            alpha=sinfo["final_alpha"],
            hist_loo_k=loo_k,
            hist_loo_alpha=loo_a,
        )


def run_spa_fixture(device: torch.device) -> int:
    """Execute live SPA verification on the bundled 3-instance Bitbrains E fixture."""
    fix_dir = DATA_DIR / "spa_fixture"
    manifest = read(fix_dir / "manifest.json")
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

        assert res["topology_invariant"], f"{stem}: topology invariance violated"
        assert res["selected_body"] == item["expected_body"], (
            f"{stem}: body {res['selected_body']} != {item['expected_body']}"
        )
        assert res["winning_families"] == item["expected_families"], (
            f"{stem}: families {res['winning_families']} != {item['expected_families']}"
        )
        assert res["winning_actions"] == item["expected_actions"], (
            f"{stem}: actions {res['winning_actions']} != {item['expected_actions']}"
        )
        rel_val = abs(res["val_mse"] - item["expected_val_mse"]) / max(item["expected_val_mse"], 1e-15)
        rel_fut = abs(fut_mse - item["expected_future_mse"]) / max(item["expected_future_mse"], 1e-15)
        assert rel_val < 1e-4, f"{stem}: val_mse rel diff {rel_val:.3e}"
        assert rel_fut < 1e-4, f"{stem}: future_mse rel diff {rel_fut:.3e}"
        print(
            f"PASS [{item['tag']}] {stem}: body={res['selected_body']} "
            f"j*={res['routed_info']['low1']['j']} val_rel={rel_val:.2e} fut_rel={rel_fut:.2e}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["bitbrains", "azure"], default="bitbrains")
    parser.add_argument("--bitbrains", type=Path, help="Path to GWA-T-12 fastStorage.zip (for Bitbrains E)")
    parser.add_argument("--weights-dir", type=Path, help="Directory holding source checkpoints H{H}_A{arch}.pt")
    parser.add_argument("--budgets", choices=list(TIERS), nargs="+", default=list(TIERS))
    parser.add_argument("--tasks", type=int, default=120)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=WORK / "runs" / "msa_dti")
    parser.add_argument("--verify-fixture", action="store_true", help="Run bundled 3-instance SPA fixture check and exit")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.verify_fixture:
        return run_spa_fixture(device)

    if args.dataset == "bitbrains" and args.bitbrains is None:
        parser.error("--bitbrains PATH_TO_FASTSTORAGE_ZIP is required when --dataset bitbrains")
    if args.weights_dir is None:
        parser.error("--weights-dir PATH is required for full target instantiation")
    if not 1 <= args.tasks <= 120:
        parser.error("--tasks must be between 1 and 120")

    stage = "E" if args.dataset == "bitbrains" else "azure"
    norm = read(PROTO_DIR / "E" / "normalization.json")
    cohort = cases(stage)[: args.tasks]
    rows: list[dict] = []
    WORK.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="msa-target-", dir=WORK) as tmp:
        tmp_dir = Path(tmp)
        for c in cohort:
            pre_bytes = carrier(stage, c, args.bitbrains, prefix=True)
            if digest(pre_bytes) != c["prefix_hash"]:
                raise ValueError(f"Prefix SHA-256 mismatch for {c['slug']}")
            pre_path = tmp_dir / f"{c['slug']}.bin"
            pre_path.write_bytes(pre_bytes)
            pref = prefix_selection(c, pre_path, norm)
            xs, ys, xv, yv = pref["Xs"], pref["ys"], pref["Xv"], pref["yv"]

            # Future evaluation tensors are opened only after prefix adaptation completes
            raw_fut = carrier(stage, c, args.bitbrains, prefix=False)
            y_fut, valid_fut = truth(c, raw_fut)
            raw_rows = np.frombuffer(raw_fut, dtype=DTYPE)
            feat_all = causal_features(raw_rows["timestamp"], raw_rows["raw"], norm)
            origins = np.arange(c["T"], c["T"] + c["B"])
            x_fut_t = torch.from_numpy(
                np.stack([feat_all[o - 95 : o + 1] for o in origins]).astype(np.float32)[valid_fut]
            ).to(device)

            adapted_cache: dict[int, tuple[dict, float, dict]] = {}
            for budget in args.budgets:
                sel = select_rgms_from_cache(stage, c["slug"], int(c["H"]), int(c["K"]), budget)
                arch_id = int(sel["selected_arch"])
                if arch_id not in adapted_cache:
                    sd0 = load_source_state(args.weights_dir, int(c["H"]), arch_id)
                    fam = rt.ARCH[arch_id].family
                    model = rt.build(arch_id, int(c["H"]), sd0, int(c["seed"])).to(device)
                    with rt.candidate_backend_context(rt.ARCH[arch_id], device, "gru-native"):
                        base_name, base_sd, _ = early_stop_base02(
                            model,
                            sd0,
                            torch.from_numpy(xs).to(device),
                            torch.from_numpy(ys).to(device),
                            torch.from_numpy(xv).to(device),
                            yv,
                            seed=int(c["seed"]),
                        )
                        spa_res = adapt_model_spa(model, fam, base_sd, xs, ys, xv, yv, device=device)
                        model.load_state_dict(spa_res["deployed_state_dict"], strict=True)
                        _, _, pred_fut, _, _ = extract_hidden_and_pred(model, x_fut_t)
                    m_dict = metrics(pred_fut, y_fut)
                    adapted_cache[arch_id] = (spa_res, base_name, m_dict)

                spa_res, base_name, m_dict = adapted_cache[arch_id]
                rows.append(
                    {
                        "stage": stage,
                        "VM": c["VM"],
                        "slug": c["slug"],
                        "H": int(c["H"]),
                        "K": int(c["K"]),
                        "budget": budget,
                        "arch": arch_id,
                        "q_init": sel["initial_arch"],
                        "q_top": sel["consensus_arch"],
                        "direct_margin": sel["direct_margin"],
                        "base02_state": base_name,
                        "spa_body": spa_res["selected_body"],
                        "topology_invariant": spa_res["topology_invariant"],
                        **m_dict,
                    }
                )
            print(f"completed {stage}/{c['slug']}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / f"{args.dataset}_{args.tasks}_msa_dti.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {len(rows)} target deployment results to {out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
