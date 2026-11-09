# core/space/types.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Literal, Optional, Tuple

Family = Literal["MLP", "TCN", "GRU"]


@dataclass(frozen=True)
class ArchSpec:
    """
    A single architecture spec inside the shared candidate space A_base.

    Required fields:
      - arch_id: stable integer id in [0, |A|-1]
      - arch_key: stable, unique, reproducible string key
      - family: one of {"MLP","TCN","GRU"}
      - hparams: discrete hyper-parameters (family-specific)

    We intentionally store hparams in a flat dict for simplicity & serialization.
    """
    arch_id: int
    arch_key: str
    family: Family
    hparams: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def short(self) -> str:
        return f"[{self.arch_id:02d}] {self.arch_key}"


def make_arch_key(family: str, hparams: Dict[str, Any]) -> str:
    """
    Create a stable unique key; order of fields is fixed per family.
    """
    if family == "MLP":
        return f"MLP_L{hparams['n_layers']}_H{hparams['hidden_dim']}_D{hparams['dropout']}"
    if family == "TCN":
        return f"TCN_B{hparams['n_blocks']}_C{hparams['channels']}_K{hparams['kernel']}_Dil{hparams['dilation']}"
    if family == "GRU":
        return f"GRU_L{hparams['n_layers']}_H{hparams['hidden_dim']}_D{hparams['dropout']}"
    raise ValueError(f"Unknown family={family}")


def get_budget_tier(main_budget_cfg: Any, budget_tier: str) -> Any:
    """
    main_budget_cfg: cfg.main.budget (BudgetCfg)
    budget_tier: "tight" | "medium" | "loose"
    return: BudgetTier-like object with .flops .params .name
    """
    if not hasattr(main_budget_cfg, budget_tier):
        raise ValueError(f"BudgetCfg has no tier '{budget_tier}'. Expect one of tight/medium/loose.")
    return getattr(main_budget_cfg, budget_tier)
