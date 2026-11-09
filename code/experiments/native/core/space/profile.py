# core/space/profile.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .types import ArchSpec, get_budget_tier
from .models import build_model


def _count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _mlp_flops(L: int, input_dim: int, H: int, n_layers: int, hidden_dim: int) -> int:
    # Linear flops: 2 * in * out (mul+add), ignore bias
    fl = 0
    in_dim = L * input_dim
    d_in = in_dim
    for _ in range(int(n_layers)):
        fl += 2 * d_in * int(hidden_dim)
        d_in = int(hidden_dim)
    fl += 2 * d_in * int(H)
    return int(fl)


def _gru_flops(L: int, input_dim: int, H: int, n_layers: int, hidden_dim: int) -> int:
    """
    Rough FLOPs estimate for GRU forward for a single sample (B=1).
    GRU has 3 gates. For each time step:
      gate linear: xW (D*H) + hU (H*H) per gate; mul+add => 2*(D*H + H*H)
      total gates => *3
    For stacked layers, D changes to hidden_dim after first layer.
    We ignore elementwise ops (sigmoid/tanh) as constant factor.
    """
    fl = 0
    d_in = int(input_dim)
    for layer in range(int(n_layers)):
        h = int(hidden_dim)
        per_t = 2 * (d_in * h + h * h) * 3
        fl += int(L) * per_t
        d_in = h
    # head
    fl += 2 * int(hidden_dim) * int(H)
    return int(fl)


def _tcn_flops(L: int, input_dim: int, H: int, n_blocks: int, channels: int, kernel: int) -> int:
    """
    Rough FLOPs for 1D conv:
      per conv: 2 * L * out_ch * (in_ch * k)
    Our block has conv1 + conv2 + residual(1x1 if in!=out).
    We keep sequence length constant via padding.
    """
    fl = 0
    in_ch = int(input_dim)
    out_ch = int(channels)
    k = int(kernel)
    for _ in range(int(n_blocks)):
        # conv1
        fl += 2 * int(L) * out_ch * (in_ch * k)
        # conv2
        fl += 2 * int(L) * out_ch * (out_ch * k)
        # residual (1x1 conv only if channel mismatch)
        if in_ch != out_ch:
            fl += 2 * int(L) * out_ch * (in_ch * 1)
        in_ch = out_ch
    # head
    fl += 2 * out_ch * int(H)
    return int(fl)


def profile_arch(
    arch_spec: ArchSpec,
    L: int,
    input_dim: int,
    H: int,
    device: str = "cpu",
) -> Tuple[int, int]:
    """
    Unified Params/FLOPs estimator.

    Params: exact count from instantiated model.
    FLOPs: deterministic analytic estimate (no dependency on torch profiler versions).
    """
    # exact params
    m = build_model(arch_spec, input_dim=input_dim, H=H, L=L, device=device)
    params = _count_params(m)

    fam = arch_spec.family
    hp = arch_spec.hparams

    if fam == "MLP":
        flops = _mlp_flops(L=L, input_dim=input_dim, H=H,
                           n_layers=int(hp["n_layers"]), hidden_dim=int(hp["hidden_dim"]))
    elif fam == "GRU":
        flops = _gru_flops(L=L, input_dim=input_dim, H=H,
                           n_layers=int(hp["n_layers"]), hidden_dim=int(hp["hidden_dim"]))
    elif fam == "TCN":
        flops = _tcn_flops(L=L, input_dim=input_dim, H=H,
                           n_blocks=int(hp["n_blocks"]), channels=int(hp["channels"]), kernel=int(hp["kernel"]))
    else:
        raise ValueError(f"Unknown family={fam}")

    return int(params), int(flops)


def is_feasible(
    arch_spec: ArchSpec,
    main_budget_cfg: Any,
    budget_tier: Union[str, Any],
    L: int,
    input_dim: int,
    H: int,
) -> bool:
    """
    Hard feasibility check under BudgetCfg.

    If BudgetCfg.hard_filter == True:
        feasible iff params<=B^P and flops<=B^F
    Else:
        always True (penalty-based handled elsewhere; module2 only provides hard_filter semantics)
    """
    hard_filter = bool(getattr(main_budget_cfg, "hard_filter", True))
    if not hard_filter:
        return True

    tier_obj = budget_tier
    if isinstance(budget_tier, str):
        tier_obj = get_budget_tier(main_budget_cfg, budget_tier)

    params, flops = profile_arch(arch_spec, L=L, input_dim=input_dim, H=H, device="cpu")
    return (params <= float(tier_obj.params)) and (flops <= float(tier_obj.flops))


def smoke_space(
    main_cfg: Any,
    input_dim: int,
    H: int,
    L: int = 96,
    topk_print: int = 5,
) -> None:
    """
    Space-level smoke test:
      - enumerate 66 architectures
      - print params/flops summary
      - feasible rate under tight/medium/loose

    Only analyzes the architecture space; does NOT train.
    """
    from .enumerator import enumerate_A_base

    A = enumerate_A_base(main_cfg.arch)

    params_list = []
    flops_list = []

    for s in A:
        p, f = profile_arch(s, L=L, input_dim=input_dim, H=H, device="cpu")
        params_list.append(p)
        flops_list.append(f)

    params_arr = np.asarray(params_list, dtype=np.float64)
    flops_arr = np.asarray(flops_list, dtype=np.float64)

    def _stat(x: np.ndarray) -> Dict[str, float]:
        return {
            "min": float(np.min(x)),
            "p25": float(np.percentile(x, 25)),
            "median": float(np.median(x)),
            "p75": float(np.percentile(x, 75)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
        }

    print("=== [smoke_space] A_base summary ===")
    print(f"|A_base| = {len(A)} (expect 66)")
    print(f"L={L}, input_dim={input_dim}, H={H}")

    ps = _stat(params_arr)
    fs = _stat(flops_arr)
    print("\n[Params]  min/p25/median/p75/max/mean")
    print(f"         {ps['min']:.0f} / {ps['p25']:.0f} / {ps['median']:.0f} / {ps['p75']:.0f} / {ps['max']:.0f} / {ps['mean']:.0f}")
    print("\n[FLOPs]   min/p25/median/p75/max/mean")
    print(f"         {fs['min']:.0f} / {fs['p25']:.0f} / {fs['median']:.0f} / {fs['p75']:.0f} / {fs['max']:.0f} / {fs['mean']:.0f}")

    bcfg = main_cfg.budget
    for tier_name in ("tight", "medium", "loose"):
        tier = getattr(bcfg, tier_name)
        feas = 0
        for s in A:
            if is_feasible(s, bcfg, tier, L=L, input_dim=input_dim, H=H):
                feas += 1
        rate = feas / len(A)
        print(f"\n[FeasibleRate] tier={tier_name:<6s} "
              f"(B_flops={float(tier.flops):.2e}, B_params={float(tier.params):.2e}, hard_filter={bool(getattr(bcfg,'hard_filter',True))}) -> "
              f"{rate*100:.1f}% ({feas}/{len(A)})")

    # Print a few smallest / largest for quick inspection
    idx_small_p = np.argsort(params_arr)[:topk_print]
    idx_large_p = np.argsort(-params_arr)[:topk_print]
    idx_small_f = np.argsort(flops_arr)[:topk_print]
    idx_large_f = np.argsort(-flops_arr)[:topk_print]

    def _dump(title: str, idxs: np.ndarray):
        print(f"\n--- {title} ---")
        for i in idxs:
            s = A[int(i)]
            print(f"{s.short():<35s}  params={int(params_arr[i])}  flops={int(flops_arr[i])}")

    _dump(f"Top-{topk_print} smallest Params", idx_small_p)
    _dump(f"Top-{topk_print} largest  Params", idx_large_p)
    _dump(f"Top-{topk_print} smallest FLOPs", idx_small_f)
    _dump(f"Top-{topk_print} largest  FLOPs", idx_large_f)
