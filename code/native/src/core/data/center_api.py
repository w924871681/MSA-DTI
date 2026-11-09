# core/data/center_api.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import torch

from core.data.sim import simulate_centers, CenterData


@dataclass
class CenterRecord:
    center_id: int
    center_type: str   # evaluation stratum only
    tier_name: str

    u: torch.Tensor
    m: torch.Tensor
    dt: Optional[torch.Tensor] = None
    y: Optional[torch.Tensor] = None
    schema_mask: Optional[torch.Tensor] = None
    timestamps: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaDatasetCache:
    centers: Dict[int, CenterRecord]


# -----------------------------
# Feature builder
# -----------------------------
def build_feature_matrix(cfg, rec: CenterRecord) -> torch.Tensor:
    feats = [rec.u]
    if bool(cfg.main.task.include_mask):
        feats.append(rec.m)
    if bool(cfg.main.task.include_time_gap):
        if rec.dt is None:
            T = rec.u.shape[0]
            dt_feat = torch.ones(T, 1, device=rec.u.device, dtype=rec.u.dtype)
        else:
            dt_feat = rec.dt
        feats.append(dt_feat)
    return torch.cat(feats, dim=1)


# -----------------------------
# Windowing & leakage-safe chronological splitting
# -----------------------------
def make_windows(X_series: torch.Tensor, y_series: torch.Tensor, L: int, H: int) -> Tuple[torch.Tensor, torch.Tensor]:
    T = int(X_series.shape[0])
    N = T - int(L) - int(H) + 1
    if N <= 0:
        raise ValueError(f"Not enough length: T={T}, L={L}, H={H}")
    X = torch.stack([X_series[i:i + L] for i in range(N)], dim=0)
    y = torch.stack([y_series[i + L:i + L + H] for i in range(N)], dim=0)
    return X, y


def _allocate_three_way(n: int, p_val: float, p_chk: float, p_test: float) -> Tuple[int, int, int]:
    """Allocate an integer total to validation/check/test by normalized ratios."""
    if n < 3:
        raise ValueError(f"Need at least 3 units after support, got {n}")
    p = np.asarray([p_val, p_chk, p_test], dtype=np.float64)
    if np.any(p <= 0):
        raise ValueError(f"val/chk/test ratios must be positive, got {p.tolist()}")
    p = p / p.sum()
    raw = p * int(n)
    sizes = np.floor(raw).astype(int)
    sizes = np.maximum(sizes, 1)

    while int(sizes.sum()) < int(n):
        idx = int(np.argmax(raw - sizes))
        sizes[idx] += 1
    while int(sizes.sum()) > int(n):
        candidates = np.where(sizes > 1)[0]
        if candidates.size == 0:
            raise ValueError("Cannot allocate non-empty val/chk/test segments")
        idx = int(candidates[np.argmin(raw[candidates] - sizes[candidates])])
        sizes[idx] -= 1
    return int(sizes[0]), int(sizes[1]), int(sizes[2])


def _allocate_raw_segments(
    total_steps: int,
    support_steps: int,
    min_eval_steps: int,
    val_ratio: float,
    chk_ratio: float,
    test_ratio: float,
) -> Tuple[int, int, int, int]:
    """Allocate the raw timeline before any sliding windows are generated."""
    total_steps = int(total_steps)
    support_steps = int(support_steps)
    min_eval_steps = int(min_eval_steps)
    remaining = total_steps - support_steps
    if remaining < 3 * min_eval_steps:
        raise ValueError(
            f"Timeline too short after support: T={total_steps}, support={support_steps}, "
            f"need at least {3 * min_eval_steps} remaining steps"
        )

    n_val, n_chk, n_test = _allocate_three_way(
        remaining, val_ratio, chk_ratio, test_ratio
    )
    sizes = [n_val, n_chk, n_test]
    # In ordinary experiments the ratios already satisfy this. The correction
    # below only protects unusually short external timelines.
    for i in range(3):
        if sizes[i] >= min_eval_steps:
            continue
        need = min_eval_steps - sizes[i]
        donors = sorted(range(3), key=lambda j: sizes[j], reverse=True)
        for j in donors:
            if j == i:
                continue
            give = min(need, max(0, sizes[j] - min_eval_steps))
            sizes[j] -= give
            sizes[i] += give
            need -= give
            if need == 0:
                break
        if need > 0:
            raise ValueError("Cannot allocate non-empty chronological evaluation segments")
    return support_steps, int(sizes[0]), int(sizes[1]), int(sizes[2])


def split_timeline_then_window(
    X_series: torch.Tensor,
    y_series: torch.Tensor,
    L: int,
    H: int,
    K: int,
    val_ratio: float,
    chk_ratio: float,
    test_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Chronologically split the raw timeline, then build windows per segment.

    This is stricter than generating all overlapping windows and separating
    their start indices afterwards. Because each window is built entirely inside
    one raw segment, support/validation/check/test share no raw time point.
    """
    L, H, K = int(L), int(H), int(K)
    T = int(X_series.shape[0])
    if int(y_series.shape[0]) != T:
        raise ValueError("X_series and y_series must have the same timeline length")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")

    # A segment of length L + H + K - 1 yields exactly K windows.
    support_steps = L + H + K - 1
    min_eval_steps = L + H  # one valid window
    n_sup, n_val, n_chk, n_test = _allocate_raw_segments(
        total_steps=T,
        support_steps=support_steps,
        min_eval_steps=min_eval_steps,
        val_ratio=float(val_ratio),
        chk_ratio=float(chk_ratio),
        test_ratio=float(test_ratio),
    )

    boundaries = [0, n_sup, n_sup + n_val, n_sup + n_val + n_chk, T]
    if boundaries[-2] + n_test != T:
        raise RuntimeError("Internal raw-segment allocation mismatch")

    segments = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        Xp, yp = make_windows(X_series[left:right], y_series[left:right], L=L, H=H)
        segments.extend([Xp, yp])

    if int(segments[0].shape[0]) != K:
        raise RuntimeError(
            f"Support window count mismatch: got {segments[0].shape[0]}, expected K={K}"
        )
    return tuple(segments)  # type: ignore[return-value]


def split_support_val_chk_test(
    X: torch.Tensor,
    y: torch.Tensor,
    K: int,
    support_ratio: float,
    val_ratio: float,
    chk_ratio: float,
    test_ratio: float,
    seed: int,
    gap_windows: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Legacy window-level split retained for external compatibility.

    Main experiment runners no longer use this function. They call
    :func:`split_timeline_then_window`, which performs the split on the raw
    timeline before window generation.
    """
    del seed, support_ratio
    N = int(X.shape[0])
    K = int(K)
    gap = max(0, int(gap_windows))
    usable_rest = N - K - 3 * gap
    if usable_rest < 3:
        raise ValueError(f"Not enough windows: N={N}, K={K}, gap={gap}")
    n_val, n_chk, n_test = _allocate_three_way(
        usable_rest, val_ratio, chk_ratio, test_ratio
    )
    a = K
    b = a + gap
    c = b + n_val
    d = c + gap
    e = d + n_chk
    f = e + gap
    g = f + n_test
    return X[:a], y[:a], X[b:c], y[b:c], X[d:e], y[d:e], X[f:g], y[f:g]


def _normalize_value_channels_from_support(
    splits: Tuple[torch.Tensor, ...],
    k_max: int,
    include_mask: bool,
) -> Tuple[torch.Tensor, ...]:
    """Optional support-only normalization for value channels.

    Missing entries remain zero after normalization. Mask and time-gap channels
    are left unchanged. Labels are not normalized because the generator already
    produces a bounded latent workload in [0, 1].
    """
    X_sup = splits[0]
    values_sup = X_sup[..., :k_max]
    if include_mask:
        mask_sup = X_sup[..., k_max:2 * k_max].clamp(0.0, 1.0)
    else:
        mask_sup = torch.ones_like(values_sup)

    denom = mask_sup.sum(dim=(0, 1)).clamp_min(1.0)
    mean = (values_sup * mask_sup).sum(dim=(0, 1)) / denom
    centered = (values_sup - mean.view(1, 1, -1)) * mask_sup
    var = centered.pow(2).sum(dim=(0, 1)) / denom
    std = torch.sqrt(var + 1e-6)

    out: List[torch.Tensor] = []
    for X_part in splits:
        Xn = X_part.clone()
        values = Xn[..., :k_max]
        if include_mask:
            mask = Xn[..., k_max:2 * k_max].clamp(0.0, 1.0)
        else:
            mask = torch.ones_like(values)
        values = ((values - mean.view(1, 1, -1)) / std.view(1, 1, -1)) * mask
        Xn[..., :k_max] = values
        out.append(Xn)
    return tuple(out)


# -----------------------------
# sim.py connector
# -----------------------------
def _centerdata_to_record(cfg, cd: CenterData, device: str) -> CenterRecord:
    dtype = torch.float32
    u = torch.from_numpy(cd.X).to(device=device, dtype=dtype)
    m = torch.from_numpy(cd.M).to(device=device, dtype=dtype)

    if bool(cfg.main.task.include_time_gap):
        T = u.shape[0]
        dt_value = float(cd.dt)
        if bool(getattr(cfg.main.task, "normalize_time_gap", True)):
            dt_value = dt_value / max(1.0, float(cfg.main.sim.max_dt))
        dt_series = torch.full((T, 1), dt_value, device=device, dtype=dtype)
    else:
        dt_series = None

    y = torch.from_numpy(cd.y).to(device=device, dtype=dtype)
    schema_mask = torch.from_numpy(cd.schema_mask).to(device=device, dtype=dtype)
    timestamps = torch.from_numpy(cd.timestamps).to(device=device)

    return CenterRecord(
        center_id=int(cd.center_id),
        center_type=str(cd.center_type),
        tier_name=str(cd.budget_tier),
        u=u,
        m=m,
        dt=dt_series,
        y=y,
        schema_mask=schema_mask,
        timestamps=timestamps,
        metadata=dict(cd.metadata),
    )


def build_meta_dataset_cache(cfg, allowed_types: Optional[List[str]] = None) -> MetaDatasetCache:
    """Build nested source pools and a fixed target pool.

    Source and target centers are generated separately. The source master pool
    uses max(scale_centers_list), and the current run takes its first n_train
    centers. Therefore source-scale experiments are nested. The target pool uses
    a fixed seed and remains identical across source scales.
    """
    n_train = int(cfg.main.split.n_train_centers)
    n_test = int(cfg.main.split.n_test_centers)
    max_source_pool = max(
        n_train,
        max(int(x) for x in tuple(cfg.main.split.scale_centers_list)),
    )
    device = str(cfg.main.device)

    source_master = simulate_centers(
        main_cfg=cfg.main,
        n_centers=max_source_pool,
        seed_offset=int(cfg.main.sim.source_seed_offset),
        allowed_types=allowed_types,
        start_center_id=0,
    )
    source_centers = source_master[:n_train]

    target_centers = simulate_centers(
        main_cfg=cfg.main,
        n_centers=n_test,
        seed_offset=int(cfg.main.sim.target_seed_offset),
        allowed_types=allowed_types,
        start_center_id=n_train,
    )

    centers: Dict[int, CenterRecord] = {}
    for cd in [*source_centers, *target_centers]:
        rec = _centerdata_to_record(cfg, cd, device=device)
        centers[int(rec.center_id)] = rec

    expected = set(range(n_train + n_test))
    if set(centers.keys()) != expected:
        raise RuntimeError("Center IDs are not contiguous after source/target generation")
    return MetaDatasetCache(centers=centers)


def get_center_split_from_cache(
    cfg,
    cache: MetaDatasetCache,
    center_id: int,
    H: int,
    K: int,
) -> Tuple[torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor,
           str, str]:
    """Return support/validation/check/test sets and center metadata labels."""
    cfg.main.task.validate_split()

    rec = cache.centers[int(center_id)]
    if rec.y is None:
        raise ValueError("CenterRecord.y is None")

    L = int(cfg.main.task.L)
    H = int(H)
    X_series = build_feature_matrix(cfg, rec)
    Xs, ys, Xv, yv, Xchk, ychk, Xt, yt = split_timeline_then_window(
        X_series=X_series,
        y_series=rec.y,
        L=L,
        H=H,
        K=int(K),
        val_ratio=float(cfg.main.task.val_ratio),
        chk_ratio=float(cfg.main.task.chk_ratio),
        test_ratio=float(cfg.main.task.test_ratio),
    )

    if bool(getattr(cfg.main.task, "normalize_values_from_support", False)):
        Xs, Xv, Xchk, Xt = _normalize_value_channels_from_support(
            (Xs, Xv, Xchk, Xt),
            k_max=int(cfg.main.sim.k_max),
            include_mask=bool(cfg.main.task.include_mask),
        )

    return Xs, ys, Xv, yv, Xchk, ychk, Xt, yt, rec.tier_name, rec.center_type


def get_center_support_validation_from_cache(
    cfg,
    cache: MetaDatasetCache,
    center_id: int,
    H: int,
    K: int,
) -> Tuple[torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor,
           str, str]:
    """Return only Support and Validation for Stage 3.

    The raw chronological boundaries are identical to
    :func:`get_center_split_from_cache`, but Check and Test windows are never
    materialized or returned. Support is the only optimization split;
    Validation is evaluation-only and cannot determine a stopping point because
    Stage 3 always uses the frozen 50-update checkpoint.
    """
    cfg.main.task.validate_split()
    rec = cache.centers[int(center_id)]
    if rec.y is None:
        raise ValueError("CenterRecord.y is None")

    L = int(cfg.main.task.L)
    H = int(H)
    K = int(K)
    X_series = build_feature_matrix(cfg, rec)
    y_series = rec.y
    total_steps = int(X_series.shape[0])
    support_steps = L + H + K - 1
    min_eval_steps = L + H
    n_sup, n_val, _n_chk, _n_test = _allocate_raw_segments(
        total_steps=total_steps,
        support_steps=support_steps,
        min_eval_steps=min_eval_steps,
        val_ratio=float(cfg.main.task.val_ratio),
        chk_ratio=float(cfg.main.task.chk_ratio),
        test_ratio=float(cfg.main.task.test_ratio),
    )
    if n_sup != support_steps:
        raise RuntimeError("Internal support boundary mismatch")

    Xs, ys = make_windows(
        X_series[:n_sup], y_series[:n_sup], L=L, H=H
    )
    Xv, yv = make_windows(
        X_series[n_sup:n_sup + n_val],
        y_series[n_sup:n_sup + n_val],
        L=L,
        H=H,
    )
    if int(Xs.shape[0]) != K:
        raise RuntimeError(
            f"Support window count mismatch: got {Xs.shape[0]}, expected K={K}"
        )

    if bool(getattr(cfg.main.task, "normalize_values_from_support", False)):
        Xs, Xv = _normalize_value_channels_from_support(
            (Xs, Xv),
            k_max=int(cfg.main.sim.k_max),
            include_mask=bool(cfg.main.task.include_mask),
        )
    return Xs, ys, Xv, yv, rec.tier_name, rec.center_type


def get_center_support_validation_check_from_cache(
    cfg,
    cache: MetaDatasetCache,
    center_id: int,
    H: int,
    K: int,
) -> Tuple[torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor,
           str, str]:
    """Return Support, Validation and Check for Stage 4 without Test.

    The chronological boundaries are identical to :func:`get_center_split_from_cache`.
    Support is the only optimization split. Validation is used only to select one
    final adapted twin. Check is materialized only after the frozen 50-update
    adaptation and is used solely for independent Stage-4 evaluation. Test is
    neither windowed nor returned.
    """
    cfg.main.task.validate_split()
    rec = cache.centers[int(center_id)]
    if rec.y is None:
        raise ValueError("CenterRecord.y is None")

    L = int(cfg.main.task.L)
    H = int(H)
    K = int(K)
    X_series = build_feature_matrix(cfg, rec)
    y_series = rec.y
    total_steps = int(X_series.shape[0])
    support_steps = L + H + K - 1
    min_eval_steps = L + H
    n_sup, n_val, n_chk, _n_test = _allocate_raw_segments(
        total_steps=total_steps,
        support_steps=support_steps,
        min_eval_steps=min_eval_steps,
        val_ratio=float(cfg.main.task.val_ratio),
        chk_ratio=float(cfg.main.task.chk_ratio),
        test_ratio=float(cfg.main.task.test_ratio),
    )
    if n_sup != support_steps:
        raise RuntimeError("Internal support boundary mismatch")

    left_val = n_sup
    right_val = left_val + n_val
    left_chk = right_val
    right_chk = left_chk + n_chk

    Xs, ys = make_windows(X_series[:n_sup], y_series[:n_sup], L=L, H=H)
    Xv, yv = make_windows(
        X_series[left_val:right_val], y_series[left_val:right_val], L=L, H=H
    )
    Xchk, ychk = make_windows(
        X_series[left_chk:right_chk], y_series[left_chk:right_chk], L=L, H=H
    )
    if int(Xs.shape[0]) != K:
        raise RuntimeError(
            f"Support window count mismatch: got {Xs.shape[0]}, expected K={K}"
        )

    if bool(getattr(cfg.main.task, "normalize_values_from_support", False)):
        Xs, Xv, Xchk = _normalize_value_channels_from_support(
            (Xs, Xv, Xchk),
            k_max=int(cfg.main.sim.k_max),
            include_mask=bool(cfg.main.task.include_mask),
        )
    return Xs, ys, Xv, yv, Xchk, ychk, rec.tier_name, rec.center_type
