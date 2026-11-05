# core/utils/metrics.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch


# -----------------------------
# Basic point metrics
# -----------------------------
def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean()


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


@torch.no_grad()
def samplewise_se(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred/target: (N, H) or (B, H)
    return: (N,) sample-wise squared error averaged over horizon dim.
    """
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(f"samplewise_se expects 2D tensors (N,H); got pred={pred.shape}, target={target.shape}")
    return ((pred - target) ** 2).mean(dim=1)


@torch.no_grad()
def worst_k_percent_from_se(se: torch.Tensor, p: float = 0.1) -> float:
    """
    Worst-10% (or p) based on sample-wise SE.
    se: (N,)
    """
    if se.numel() == 0:
        return float("nan")
    se_sorted, _ = torch.sort(se, descending=True)
    k = max(1, int(round(float(p) * float(se_sorted.numel()))))
    return float(se_sorted[:k].mean().item())


# -----------------------------
# Consistency (PPT口径)
#   - F_pred: on D_val
#   - F_cons: on D_chk with fixed W
# -----------------------------
@dataclass(frozen=True)
class ConsistencyCfg:
    """
    W 的固定构造口径（可复现）：
      - mode="exp_decay": w_t = decay^t, t=0..H-1
      - normalize: 是否归一化使 sum(w)=1
      - seed: 预留（如果未来要随机W/扰动W），目前不使用，确保可复现
    """
    mode: str = "exp_decay"
    decay: float = 0.7
    normalize: bool = True
    seed: int = 2026


@torch.no_grad()
def build_W(H: int, device: str | torch.device, cfg: Optional[ConsistencyCfg] = None) -> torch.Tensor:
    """
    返回权重向量 w: (H,)
    ——工程实现用向量更稳定、也更快；PPT里矩阵W可等价理解为 diag(w)
    """
    if cfg is None:
        cfg = ConsistencyCfg()
    H = int(H)
    if H <= 0:
        raise ValueError("H must be positive")

    if cfg.mode == "exp_decay":
        decay = float(cfg.decay)
        w = torch.tensor([decay ** t for t in range(H)], dtype=torch.float32, device=device)
    elif cfg.mode == "uniform":
        w = torch.ones(H, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unknown ConsistencyCfg.mode={cfg.mode}")

    if cfg.normalize:
        s = w.sum().clamp_min(1e-12)
        w = w / s
    return w


@torch.no_grad()
def consistency_from_residual(residual: torch.Tensor, w: torch.Tensor) -> float:
    """
    residual: (N, H) = (pred - target)
    w: (H,)
    返回：mean_n sum_h w_h * residual_{n,h}^2
    """
    if residual.ndim != 2:
        raise ValueError(f"residual must be 2D (N,H), got {residual.shape}")
    if w.ndim != 1 or w.numel() != residual.shape[1]:
        raise ValueError(f"w must be (H,), got w={w.shape}, H={residual.shape[1]}")

    se = (residual ** 2)  # (N,H)
    val = (se * w.view(1, -1)).sum(dim=1).mean()
    return float(val.item())


@torch.no_grad()
def eval_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(mse(pred, target).item())


@torch.no_grad()
def eval_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(mae(pred, target).item())


@torch.no_grad()
def eval_consistency(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> float:
    return consistency_from_residual(pred - target, w)


@torch.no_grad()
def eval_worst10(pred: torch.Tensor, target: torch.Tensor, p: float = 0.1) -> float:
    se = samplewise_se(pred, target)
    return worst_k_percent_from_se(se, p=p)


# -----------------------------
# Aggregation helpers
# -----------------------------
def mean_std(xs: List[float]) -> Tuple[float, float]:
    if len(xs) == 0:
        return float("nan"), float("nan")
    x = torch.tensor(xs, dtype=torch.float64)
    return float(x.mean().item()), float(x.std(unbiased=False).item())


def safe_mean(xs: List[float]) -> float:
    if len(xs) == 0:
        return float("nan")
    return float(sum(xs) / len(xs))


def flatten_time_breakdowns(rows: List[dict], key: str = "time_breakdown") -> Dict[str, float]:
    """
    对某一组 rows，按 time_breakdown 的每个子键求平均（仅对 evaluated=1 的记录统计更合理，
    但这里保持通用：调用方先过滤）。
    """
    acc: Dict[str, List[float]] = {}
    for r in rows:
        tb = r.get(key, {}) or {}
        for k, v in tb.items():
            try:
                acc.setdefault(k, []).append(float(v))
            except Exception:
                pass
    return {k: safe_mean(vs) for k, vs in acc.items()}


@torch.no_grad()
def trend_direction_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mismatch ratio of first-order direction changes, minimized."""
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(f"trend_direction_error expects 2D tensors (N,H); got pred={pred.shape}, target={target.shape}")
    if pred.shape[1] <= 1:
        return 0.0
    pd = pred[:, 1:] - pred[:, :-1]
    td = target[:, 1:] - target[:, :-1]
    ps = torch.sign(pd)
    ts = torch.sign(td)
    # treat tiny fluctuations as 0 to reduce noise sensitivity
    ps = torch.where(pd.abs() < 1e-8, torch.zeros_like(ps), ps)
    ts = torch.where(td.abs() < 1e-8, torch.zeros_like(ts), ts)
    mismatch = (ps != ts).float().mean()
    return float(mismatch.item())


@torch.no_grad()
def peak_shape_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Combined peak index / peak value error, minimized."""
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(f"peak_shape_error expects 2D tensors (N,H); got pred={pred.shape}, target={target.shape}")
    H = int(pred.shape[1])
    if H <= 0:
        return 0.0
    p_idx = torch.argmax(pred, dim=1).float()
    t_idx = torch.argmax(target, dim=1).float()
    idx_err = (p_idx - t_idx).abs() / max(1.0, float(H - 1))
    p_val = torch.gather(pred, 1, p_idx.long().unsqueeze(1)).squeeze(1)
    t_val = torch.gather(target, 1, t_idx.long().unsqueeze(1)).squeeze(1)
    scale = target.abs().mean().clamp_min(1e-6)
    val_err = (p_val - t_val).abs() / scale
    return float((0.5 * idx_err.mean() + 0.5 * val_err.mean()).item())


@torch.no_grad()
def eval_shape_consistency(pred: torch.Tensor, target: torch.Tensor, trend_weight: float = 0.7, peak_weight: float = 0.3) -> float:
    tw = max(0.0, float(trend_weight))
    pw = max(0.0, float(peak_weight))
    s = max(1e-12, tw + pw)
    tw /= s
    pw /= s
    return tw * trend_direction_error(pred, target) + pw * peak_shape_error(pred, target)


@torch.no_grad()
def eval_weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: Optional[torch.Tensor] = None) -> float:
    """Validation objective E_val with normalized horizon weights."""
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(f"eval_weighted_mse expects (N,H), got {pred.shape}, {target.shape}")
    H = int(pred.shape[1])
    if weights is None:
        weights = torch.ones(H, device=pred.device, dtype=pred.dtype) / max(1, H)
    else:
        weights = weights.to(device=pred.device, dtype=pred.dtype).reshape(-1)
        if int(weights.numel()) != H:
            raise ValueError(f"weights length {weights.numel()} != H={H}")
        weights = weights / weights.sum().clamp_min(1e-12)
    value = (((pred - target) ** 2) * weights.view(1, -1)).sum(dim=1).mean()
    return float(value.item())


@torch.no_grad()
def eval_paper_sequence_consistency(
    pred: torch.Tensor,
    target: torch.Tensor,
    eta: Tuple[float, float, float] = (1/3, 1/3, 1/3),
) -> Tuple[float, Dict[str, float]]:
    """Paper consistency C_val over chronologically ordered predictions.

    The temporal difference is taken along the sample-time axis independently
    for each forecast horizon, then averaged across horizons. This definition is
    valid for both H=1 and H>1.
    """
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(f"Expected pred/target shape (N,H), got {pred.shape}, {target.shape}")
    if pred.shape != target.shape:
        raise ValueError("pred and target must have identical shapes")
    if int(pred.shape[0]) < 2:
        trend = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        var = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    else:
        dp = pred[1:] - pred[:-1]
        dt = target[1:] - target[:-1]
        trend = (dp - dt).abs().mean()
        var = (dp.var(dim=0, unbiased=False) - dt.var(dim=0, unbiased=False)).abs().mean()
    peak = (
        (pred.max(dim=0).values - target.max(dim=0).values).abs()
        + (pred.min(dim=0).values - target.min(dim=0).values).abs()
    ).mean()

    e1, e2, e3 = [max(0.0, float(x)) for x in eta]
    s = max(1e-12, e1 + e2 + e3)
    e1, e2, e3 = e1 / s, e2 / s, e3 / s
    total = e1 * trend + e2 * peak + e3 * var
    parts = {
        "trend": float(trend.item()),
        "peak": float(peak.item()),
        "variation": float(var.item()),
    }
    return float(total.item()), parts
