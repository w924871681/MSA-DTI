"""Frozen bank definitions, deployment budgets, and hyperparameters for MSA-DTI."""
from __future__ import annotations

import json
from pathlib import Path

# Frozen 19-model reusable candidate bank retained by offline screening
CANDIDATE_BANK: tuple[int, ...] = (
    0, 6, 8, 13,
    27, 31, 32, 35, 40, 45, 46, 47, 50,
    54, 55, 57, 58, 61, 63,
)

ARCHITECTURE_FAMILY: dict[int, str] = {
    0: "MLP", 6: "MLP", 8: "MLP", 13: "MLP",
    27: "TCN", 31: "TCN", 32: "TCN", 35: "TCN", 40: "TCN",
    45: "TCN", 46: "TCN", 47: "TCN", 50: "TCN",
    54: "GRU", 55: "GRU", 57: "GRU", 58: "GRU", 61: "GRU", 63: "GRU",
}

# Dual-limit deployment budgets: (max_operations_per_forecast, max_parameters)
DEPLOYMENT_BUDGETS: dict[str, tuple[float, float]] = {
    "tight": (1.5e6, 3.0e4),
    "medium": (5.0e6, 1.0e5),
    "loose": (20.0e6, 5.0e5),
}

# Task configurations: (forecast_horizon H, support_size K)
HORIZON_SUPPORT_GRID: tuple[tuple[int, int], ...] = (
    (1, 10), (1, 20), (4, 10), (4, 20),
)

# Frozen algorithmic hyperparameters selected strictly on development cohorts
FROZEN_HYPERPARAMETERS: dict[str, object] = {
    "sequence_length": 96,
    "input_channels": 25,
    "chronological_blocks": 4,
    "scale_floor": 1e-12,
    "sgd_steps": 50,
    "sgd_lr": 0.02,
    "grad_clip_norm": 1.0,
    "early_checkpoints": ["b50", "b25", "b10", "bW", "a0"],
    "channel_importance_ridge": 0.01,
    "routing_load_columns": {1: 3, 5: 10},
    "causal_load_scales": [1, 5, 20, 96],
    "scale_blend_grid": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
    "calibration_mix_grid": [0.25, 0.5, 0.75, 1.0],
    "residual_gamma_grid": [0.5, 1.0],
}


def load_resource_profiles(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def feasible_candidates(
    profiles: dict,
    horizon: int,
    max_ops: float,
    max_params: float,
    bank: tuple[int, ...] = CANDIDATE_BANK,
) -> list[int]:
    """Return bank indices satisfying both hard operation and parameter limits,
    ordered deterministically by (parameters, operations, architecture_id)."""
    prof = profiles[str(horizon)]
    ok = []
    for idx, arch_id in enumerate(bank):
        p = prof[str(arch_id)]
        if p["operations"] <= max_ops and p["parameters"] <= max_params:
            ok.append((p["parameters"], p["operations"], arch_id, idx))
    ok.sort()
    return [item[3] for item in ok]
