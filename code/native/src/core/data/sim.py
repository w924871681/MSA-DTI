# core/data/sim.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class CenterData:
    center_id: int
    center_type: str                 # evaluation stratum only: A/B/C
    dt: int                          # true sampling interval in base steps
    budget_tier: str                 # tight/medium/loose
    X: np.ndarray                    # (T, Kmax), observed values with zero fill
    M: np.ndarray                    # (T, Kmax), explicit observation mask
    schema_mask: np.ndarray          # (Kmax,), permanently available metrics
    y: np.ndarray                    # (T,), latent true workload in [0, 1]
    timestamps: np.ndarray           # (T,), sampled physical-time indices
    X_clean: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -12.0, 12.0)
    return 1.0 / (1.0 + np.exp(-x))


def _sample_budget_tier(budget_cfg: Any, center_type: str, rng: np.random.RandomState) -> str:
    pt = np.asarray(budget_cfg.p_by_type[center_type], dtype=np.float64)
    pt = pt / pt.sum()
    return str(rng.choice(["tight", "medium", "loose"], p=pt))


def _make_type_schedule(
    n_centers: int,
    type_ratio: Tuple[int, int, int],
    allowed_types: Sequence[str],
    rng: np.random.RandomState,
) -> List[str]:
    """Build a prefix-balanced schedule.

    For the default 3:4:3 ratio, every complete 10-center block contains
    exactly 3 A, 4 B, and 3 C centers. Each block is shuffled, so prefixes used
    by source-scale experiments remain balanced without grouping all A centers
    first.
    """
    ratio_map = {"A": int(type_ratio[0]), "B": int(type_ratio[1]), "C": int(type_ratio[2])}
    allowed_types = list(allowed_types)
    if not allowed_types:
        raise ValueError("allowed_types must not be empty")

    base_block: List[str] = []
    for t in allowed_types:
        base_block.extend([t] * max(1, ratio_map[t]))

    schedule: List[str] = []
    while len(schedule) < int(n_centers):
        block = list(base_block)
        rng.shuffle(block)
        schedule.extend(block)
    return schedule[: int(n_centers)]


def _generate_latent_workload(
    T_base: int,
    type_cfg: Any,
    sim_cfg: Any,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate the latent physical workload before monitoring defects."""
    t = np.arange(T_base, dtype=np.float64)

    base_logit = float(rng.uniform(*type_cfg.base_logit_range))
    amp = float(rng.uniform(*type_cfg.cycle_amp_range))
    phase_short = float(rng.uniform(0.0, 2.0 * np.pi))
    phase_day = float(rng.uniform(0.0, 2.0 * np.pi))
    phase_long = float(rng.uniform(0.0, 2.0 * np.pi))

    seasonal = (
        0.35 * amp * np.sin(2.0 * np.pi * t / float(sim_cfg.T_short) + phase_short)
        + 0.75 * amp * np.sin(2.0 * np.pi * t / float(sim_cfg.T_day) + phase_day)
        + 0.25 * amp * np.sin(2.0 * np.pi * t / float(sim_cfg.T_long) + phase_long)
    )

    phi = float(rng.uniform(*type_cfg.ar_phi_range))
    sigma_process = float(rng.uniform(*type_cfg.process_sigma_range))
    ar = np.zeros(T_base, dtype=np.float64)
    eps = rng.normal(0.0, sigma_process, size=T_base)
    for i in range(1, T_base):
        ar[i] = phi * ar[i - 1] + eps[i]

    burst_rate = float(rng.uniform(*type_cfg.burst_rate_range))
    n_bursts = int(rng.poisson(burst_rate * T_base / 1000.0))
    burst = np.zeros(T_base, dtype=np.float64)
    burst_events: List[Tuple[int, float, float]] = []
    for _ in range(n_bursts):
        center = int(rng.randint(0, T_base))
        width = float(rng.uniform(2.0, 14.0))
        strength = float(rng.uniform(*type_cfg.burst_amp_range))
        burst += strength * np.exp(-0.5 * ((t - center) / width) ** 2)
        burst_events.append((center, strength, width))

    drift = np.zeros(T_base, dtype=np.float64)
    drift_type = "none"
    drift_start = -1
    drift_strength = 0.0
    if rng.rand() < float(type_cfg.drift_prob):
        drift_start = int(rng.randint(int(0.45 * T_base), max(int(0.46 * T_base), int(0.75 * T_base))))
        drift_strength = float(rng.uniform(*type_cfg.drift_strength_range))
        if rng.rand() < 0.25:
            drift_strength *= -1.0
        if rng.rand() < 0.5:
            drift_type = "abrupt"
            drift[drift_start:] = drift_strength
        else:
            drift_type = "gradual"
            width = max(8.0, 0.04 * T_base)
            drift = drift_strength * _sigmoid((t - float(drift_start)) / width)

    latent_logit = base_logit + seasonal + ar + burst + drift
    y_true = _sigmoid(latent_logit).astype(np.float32)

    meta = {
        "base_logit": base_logit,
        "cycle_amp": amp,
        "ar_phi": phi,
        "process_sigma": sigma_process,
        "n_bursts": n_bursts,
        "burst_events": burst_events,
        "drift_type": drift_type,
        "drift_start_base": drift_start,
        "drift_strength": drift_strength,
    }
    return y_true, meta


def _generate_clean_metrics(
    y_true_base: np.ndarray,
    Kmax: int,
    sim_cfg: Any,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Map latent workload to multivariate clean monitoring signals."""
    T_base = int(y_true_base.shape[0])
    t = np.arange(T_base, dtype=np.float64)
    eps = 1e-5
    y_logit = np.log(np.clip(y_true_base, eps, 1.0 - eps) / np.clip(1.0 - y_true_base, eps, 1.0))

    X = np.zeros((T_base, Kmax), dtype=np.float32)
    delays: List[int] = []
    gains: List[float] = []
    biases: List[float] = []
    quads: List[float] = []

    for k in range(Kmax):
        delay = int(rng.randint(0, int(sim_cfg.max_metric_delay) + 1))
        delayed = np.empty_like(y_logit)
        if delay == 0:
            delayed[:] = y_logit
        else:
            delayed[:delay] = y_logit[0]
            delayed[delay:] = y_logit[:-delay]

        gain = float(rng.uniform(*sim_cfg.metric_gain_range))
        bias = float(rng.uniform(*sim_cfg.metric_bias_range))
        quad = float(rng.uniform(*sim_cfg.metric_quad_range))
        metric_sigma = float(rng.uniform(*sim_cfg.metric_process_sigma))
        metric_phase = float(rng.uniform(0.0, 2.0 * np.pi))
        local_cycle = 0.08 * np.sin(2.0 * np.pi * t / float(sim_cfg.T_day) + metric_phase)
        local_noise = rng.normal(0.0, metric_sigma, size=T_base)

        signal = bias + gain * delayed + quad * (y_true_base - 0.5) ** 2 + local_cycle + local_noise
        X[:, k] = _sigmoid(signal).astype(np.float32)

        delays.append(delay)
        gains.append(gain)
        biases.append(bias)
        quads.append(quad)

    return X, {
        "metric_delays": delays,
        "metric_gains": gains,
        "metric_biases": biases,
        "metric_quads": quads,
    }


def _apply_monitoring_missingness(
    M: np.ndarray,
    schema_mask: np.ndarray,
    miss_rate: float,
    miss_block_len: int,
    sim_cfg: Any,
    rng: np.random.RandomState,
) -> Dict[str, int]:
    """Add point, single-metric, grouped, and rare global missingness."""
    T, K = M.shape
    active = np.flatnonzero(schema_mask > 0.5)
    if active.size == 0:
        raise ValueError("At least one metric must remain observable")

    # Point missing only on metrics available in the center schema.
    point_drop = rng.rand(T, K) < float(miss_rate)
    point_drop[:, schema_mask < 0.5] = False
    M[point_drop] = 0.0

    mb_cfg = sim_cfg.missing_block

    n_metric_blocks = int(rng.randint(mb_cfg.metric_block_count_range[0], mb_cfg.metric_block_count_range[1] + 1))
    for _ in range(n_metric_blocks):
        k = int(rng.choice(active))
        start = int(rng.randint(0, T))
        length = int(rng.randint(max(1, miss_block_len // 3), max(2, miss_block_len + 1)))
        M[start:min(T, start + length), k] = 0.0

    n_group_blocks = int(rng.randint(mb_cfg.group_block_count_range[0], mb_cfg.group_block_count_range[1] + 1))
    for _ in range(n_group_blocks):
        gmin = min(int(mb_cfg.group_size_range[0]), int(active.size))
        gmax = min(int(mb_cfg.group_size_range[1]), int(active.size))
        gsize = int(rng.randint(gmin, gmax + 1)) if gmax >= gmin else int(active.size)
        metrics = rng.choice(active, size=gsize, replace=False)
        start = int(rng.randint(0, T))
        length = int(rng.randint(max(2, miss_block_len // 4), max(3, miss_block_len // 2 + 1)))
        M[start:min(T, start + length), metrics] = 0.0

    n_global = 0
    if rng.rand() < float(mb_cfg.global_outage_prob):
        n_global = int(rng.randint(mb_cfg.global_outage_count_range[0], mb_cfg.global_outage_count_range[1] + 1))
        for _ in range(n_global):
            start = int(rng.randint(0, T))
            length = int(rng.randint(mb_cfg.global_outage_len_range[0], mb_cfg.global_outage_len_range[1] + 1))
            M[start:min(T, start + length), :] = 0.0

    # Schema missingness is permanent and has the last word.
    M[:, schema_mask < 0.5] = 0.0
    return {
        "n_metric_blocks": n_metric_blocks,
        "n_group_blocks": n_group_blocks,
        "n_global_outages": n_global,
    }


def simulate_centers(
    main_cfg: Any,
    n_centers: int,
    seed_offset: int = 0,
    allowed_types: Optional[Sequence[str]] = None,
    start_center_id: int = 0,
) -> List[CenterData]:
    """Generate controlled synthetic multi-center workload tasks.

    Generation order:
      1) latent true workload;
      2) clean multivariate monitoring signals;
      3) true center-specific sampling;
      4) schema/noise/missingness on observations only;
      5) resource tier.
    """
    sim_cfg = main_cfg.sim
    budget_cfg = main_cfg.budget

    rng0 = np.random.RandomState(int(sim_cfg.seed) + int(seed_offset))
    Kmax = int(sim_cfg.k_max)
    T_obs = int(sim_cfg.n_history)

    if allowed_types is None:
        allowed_types = ["A", "B", "C"]
    allowed_types = list(allowed_types)
    type_schedule = _make_type_schedule(
        n_centers=int(n_centers),
        type_ratio=tuple(sim_cfg.type_ratio),
        allowed_types=allowed_types,
        rng=rng0,
    )

    centers: List[CenterData] = []
    for local_id, ctype in enumerate(type_schedule):
        rng = np.random.RandomState(int(rng0.randint(0, 2**31 - 1)))
        type_cfg = getattr(sim_cfg, f"type_{ctype}")
        dt = int(rng.choice(type_cfg.dt_choices))

        # Generate enough base-resolution points and then truly sample every dt.
        T_base = int(sim_cfg.max_metric_delay) + (T_obs - 1) * dt + 1
        y_base, workload_meta = _generate_latent_workload(T_base, type_cfg, sim_cfg, rng)
        X_base, metric_meta = _generate_clean_metrics(y_base, Kmax, sim_cfg, rng)

        sample_idx = int(sim_cfg.max_metric_delay) + np.arange(T_obs, dtype=np.int64) * dt
        y_true = y_base[sample_idx].astype(np.float32)
        X_clean = X_base[sample_idx].astype(np.float32)

        d = int(rng.randint(type_cfg.schema_drop_range[0], type_cfg.schema_drop_range[1] + 1))
        d = min(d, Kmax - 1)
        schema_mask = np.ones((Kmax,), dtype=np.float32)
        if d > 0:
            drop_idx = rng.choice(Kmax, size=d, replace=False)
            schema_mask[drop_idx] = 0.0

        miss_rate = float(rng.uniform(*type_cfg.miss_rate_range))
        miss_block_len = int(rng.randint(type_cfg.miss_block_len_range[0], type_cfg.miss_block_len_range[1] + 1))
        noise_sigma = float(rng.uniform(*type_cfg.noise_sigma_range))

        M = np.ones((T_obs, Kmax), dtype=np.float32)
        M[:, schema_mask < 0.5] = 0.0
        missing_meta = _apply_monitoring_missingness(
            M=M,
            schema_mask=schema_mask,
            miss_rate=miss_rate,
            miss_block_len=miss_block_len,
            sim_cfg=sim_cfg,
            rng=rng,
        )

        obs_noise = rng.normal(0.0, noise_sigma, size=X_clean.shape).astype(np.float32)
        X_observed = np.clip(X_clean + obs_noise, 0.0, 1.0)
        X_filled = (X_observed * M).astype(np.float32)

        budget_tier = _sample_budget_tier(budget_cfg, ctype, rng)
        center_id = int(start_center_id) + int(local_id)
        metadata: Dict[str, Any] = {
            "generator": "hierarchical_stochastic",
            "center_type": ctype,
            "dt": dt,
            "miss_rate_draw": miss_rate,
            "miss_block_len_draw": miss_block_len,
            "noise_sigma_draw": noise_sigma,
            "schema_drop_count": d,
            "observed_ratio": float(M.mean()),
            **workload_meta,
            **metric_meta,
            **missing_meta,
        }

        centers.append(
            CenterData(
                center_id=center_id,
                center_type=ctype,
                dt=dt,
                budget_tier=budget_tier,
                X=X_filled,
                M=M.astype(np.float32),
                schema_mask=schema_mask,
                y=y_true,
                timestamps=sample_idx.astype(np.int64),
                X_clean=X_clean,
                metadata=metadata,
            )
        )

    return centers
