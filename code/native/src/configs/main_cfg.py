# Shared task, architecture, and simulation settings.
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os


# -----------------------------
# Module A: controlled synthetic multi-center data
# -----------------------------
@dataclass
class CenterTypeCfg:
    """Parameters used to build one center type.

    The three types are evaluation strata only. MSA-DTI must not
    use the type label as an input feature.
    """
    name: str

    # Monitoring heterogeneity
    dt_choices: List[int]
    miss_rate_range: Tuple[float, float]
    miss_block_len_range: Tuple[int, int]
    noise_sigma_range: Tuple[float, float]
    schema_drop_range: Tuple[int, int]

    # Latent workload heterogeneity
    base_logit_range: Tuple[float, float]
    cycle_amp_range: Tuple[float, float]
    ar_phi_range: Tuple[float, float]
    process_sigma_range: Tuple[float, float]
    burst_rate_range: Tuple[float, float]       # expected bursts per 1000 base steps
    burst_amp_range: Tuple[float, float]
    drift_prob: float
    drift_strength_range: Tuple[float, float]


@dataclass
class MissingBlockCfg:
    """Missingness is added only to monitoring observations, never to labels."""
    # Single-metric outages
    metric_block_count_range: Tuple[int, int] = (2, 6)

    # Several metrics fail together
    group_block_count_range: Tuple[int, int] = (1, 3)
    group_size_range: Tuple[int, int] = (2, 4)

    # Rare short full-monitoring outage
    global_outage_prob: float = 0.20
    global_outage_count_range: Tuple[int, int] = (0, 2)
    global_outage_len_range: Tuple[int, int] = (2, 8)


@dataclass
class SimDataCfg:
    seed: int = 2026

    # Number of observed points returned for every center after true sampling
    n_history: int = 3000
    k_max: int = 12

    # Exact A:B:C ratio for each 10-center block. Source and target pools are
    # generated separately, so both sides contain all center types.
    type_ratio: Tuple[int, int, int] = (3, 4, 3)

    type_A: CenterTypeCfg = field(default_factory=lambda: CenterTypeCfg(
        name="A",
        dt_choices=[1],
        miss_rate_range=(0.02, 0.08),
        miss_block_len_range=(5, 15),
        noise_sigma_range=(0.01, 0.03),
        schema_drop_range=(0, 2),
        base_logit_range=(-0.70, 0.10),
        cycle_amp_range=(0.20, 0.45),
        ar_phi_range=(0.55, 0.80),
        process_sigma_range=(0.04, 0.08),
        burst_rate_range=(1.0, 2.5),
        burst_amp_range=(0.25, 0.55),
        drift_prob=0.20,
        drift_strength_range=(0.10, 0.25),
    ))
    type_B: CenterTypeCfg = field(default_factory=lambda: CenterTypeCfg(
        name="B",
        dt_choices=[1, 2],
        miss_rate_range=(0.10, 0.25),
        miss_block_len_range=(20, 60),
        noise_sigma_range=(0.03, 0.06),
        schema_drop_range=(2, 4),
        base_logit_range=(-0.45, 0.35),
        cycle_amp_range=(0.30, 0.60),
        ar_phi_range=(0.60, 0.88),
        process_sigma_range=(0.06, 0.11),
        burst_rate_range=(1.5, 3.5),
        burst_amp_range=(0.35, 0.75),
        drift_prob=0.40,
        drift_strength_range=(0.15, 0.35),
    ))
    type_C: CenterTypeCfg = field(default_factory=lambda: CenterTypeCfg(
        name="C",
        dt_choices=[2, 3, 4],
        miss_rate_range=(0.25, 0.45),
        miss_block_len_range=(60, 120),
        noise_sigma_range=(0.06, 0.10),
        schema_drop_range=(4, 6),
        base_logit_range=(-0.20, 0.55),
        cycle_amp_range=(0.40, 0.75),
        ar_phi_range=(0.65, 0.92),
        process_sigma_range=(0.08, 0.14),
        burst_rate_range=(2.0, 4.5),
        burst_amp_range=(0.45, 0.95),
        drift_prob=0.60,
        drift_strength_range=(0.20, 0.45),
    ))

    missing_block: MissingBlockCfg = field(default_factory=MissingBlockCfg)

    # Shared physical-time patterns at the base sampling resolution.
    T_short: int = 24
    T_day: int = 96
    T_long: int = 672

    # Monitoring generation
    max_metric_delay: int = 8
    metric_process_sigma: Tuple[float, float] = (0.01, 0.04)
    metric_gain_range: Tuple[float, float] = (0.65, 1.45)
    metric_bias_range: Tuple[float, float] = (-0.35, 0.35)
    metric_quad_range: Tuple[float, float] = (-0.20, 0.35)

    # Reproducibility and audit
    source_seed_offset: int = 0
    target_seed_offset: int = 10000

    @property
    def max_dt(self) -> int:
        return max(
            max(self.type_A.dt_choices),
            max(self.type_B.dt_choices),
            max(self.type_C.dt_choices),
        )


# -----------------------------
# Module B: forecasting/functional-validation task
# -----------------------------
@dataclass
class TaskCfg:
    L: int = 96
    H_list: Tuple[int, ...] = (1, 4)
    K_list: Tuple[int, ...] = (10, 20)

    # K support windows are always used. The remaining windows are split by
    # these ratios after chronological gaps are reserved.
    support_ratio: float = 0.6  # retained for backward-compatible configs
    val_ratio: float = 0.2
    chk_ratio: float = 0.1
    test_ratio: float = 0.1

    # New leakage-safe protocol
    split_mode: str = "chronological"
    split_gap_windows: int = -1  # -1 means L + H - 1 skipped starts
    normalize_values_from_support: bool = False

    include_mask: bool = True
    include_time_gap: bool = True
    normalize_time_gap: bool = True

    # Kept only so old configs can still be loaded. The new generator never
    # constructs labels from masked observations.
    normalize_by_mask: bool = False

    def validate_split(self) -> None:
        s = (
            float(self.support_ratio)
            + float(self.val_ratio)
            + float(self.chk_ratio)
            + float(self.test_ratio)
        )
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"TaskCfg split ratios must sum to 1.0, got {s}")
        if self.split_mode not in {"chronological"}:
            raise ValueError(f"Unsupported split_mode={self.split_mode}")


# -----------------------------
# Module C: Budget config (hard constraints)
# -----------------------------
@dataclass
class BudgetTier:
    name: str
    flops: float
    params: float


@dataclass
class BudgetCfg:
    tight: BudgetTier = field(default_factory=lambda: BudgetTier("tight", flops=1.5e6, params=3e4))
    medium: BudgetTier = field(default_factory=lambda: BudgetTier("medium", flops=5e6, params=1e5))
    loose: BudgetTier = field(default_factory=lambda: BudgetTier("loose", flops=2e7, params=5e5))

    # Budgets are only weakly related to center type. This avoids making type a
    # direct proxy for the resource tier.
    p_by_type: Dict[str, Tuple[float, float, float]] = field(default_factory=lambda: {
        "A": (0.25, 0.45, 0.30),
        "B": (0.35, 0.45, 0.20),
        "C": (0.45, 0.40, 0.15),
    })
    hard_filter: bool = True


# -----------------------------
# Shared architecture space and search protocol
# -----------------------------
@dataclass
class ArchSpaceCfg:
    families: Tuple[str, ...] = ("MLP", "TCN", "GRU")

    mlp_layers: Tuple[int, ...] = (2, 3, 4)
    mlp_hidden: Tuple[int, ...] = (32, 64, 128)
    mlp_dropout: Tuple[float, ...] = (0.0, 0.1)

    tcn_blocks: Tuple[int, ...] = (2, 3, 4)
    tcn_channels: Tuple[int, ...] = (8, 16, 32)
    tcn_kernel: Tuple[int, ...] = (3, 5)
    tcn_dilation: Tuple[int, ...] = (1, 2)

    gru_layers: Tuple[int, ...] = (1, 2)
    gru_hidden: Tuple[int, ...] = (16, 32, 64)
    gru_dropout: Tuple[float, ...] = (0.0, 0.1)

    def total_size(self) -> int:
        mlp = len(self.mlp_layers) * len(self.mlp_hidden) * len(self.mlp_dropout)
        tcn = len(self.tcn_blocks) * len(self.tcn_channels) * len(self.tcn_kernel) * len(self.tcn_dilation)
        gru = len(self.gru_layers) * len(self.gru_hidden) * len(self.gru_dropout)
        return mlp + tcn + gru


@dataclass
class SearchBudgetCfg:
    # Every search-based method sees the same 66-architecture space and may
    # adapt at most 12 candidates. This is the protocol used in the paper.
    R_candidates: int = 66
    K_arch: int = 12
    K_proxy: int = 6

    # Global target-side adaptation budget. All adaptation-based methods must
    # read this value instead of maintaining method-specific step counts.
    T_adapt_steps: int = 50

    use_early_stop: bool = False
    include_test_eval_time: bool = False
    strict_test_isolation: bool = True


@dataclass
class SplitCfg:
    n_train_centers: int = 20
    n_test_centers: int = 50
    scale_centers_list: Tuple[int, ...] = (10, 15, 20, 25, 30, 35, 40, 45, 50)


@dataclass
class MainCfg:
    sim: SimDataCfg = field(default_factory=SimDataCfg)
    task: TaskCfg = field(default_factory=TaskCfg)
    budget: BudgetCfg = field(default_factory=BudgetCfg)
    arch: ArchSpaceCfg = field(default_factory=ArchSpaceCfg)
    search: SearchBudgetCfg = field(default_factory=SearchBudgetCfg)
    split: SplitCfg = field(default_factory=SplitCfg)

    device: str = "cuda"
    num_workers: int = 4
    out_dir: str = "./outputs"


CFG = MainCfg()


def get_cfg() -> MainCfg:
    CFG.task.validate_split()
    CFG.out_dir = os.path.abspath(str(CFG.out_dir))
    return CFG
