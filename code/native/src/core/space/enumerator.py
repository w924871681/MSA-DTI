# core/space/enumerator.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from itertools import product
from typing import Any, List

from .types import ArchSpec, make_arch_key


def enumerate_A_base(arch_cfg: Any) -> List[ArchSpec]:
    """
    Enumerate the fixed base candidate space A_base.

    Must be:
      - stable order
      - reproducible
      - total size == arch_cfg.total_size() == 66

    We follow the tuple order defined in configs.main_cfg.ArchSpaceCfg:
      families=("MLP","TCN","GRU")
      and each hyperparam tuple order as written.
    """
    specs: List[ArchSpec] = []
    arch_id = 0

    # ---- MLP ----
    for n_layers, hidden_dim, dropout in product(
        tuple(arch_cfg.mlp_layers),
        tuple(arch_cfg.mlp_hidden),
        tuple(arch_cfg.mlp_dropout),
    ):
        hp = {"n_layers": int(n_layers), "hidden_dim": int(hidden_dim), "dropout": float(dropout)}
        key = make_arch_key("MLP", hp)
        specs.append(ArchSpec(arch_id=arch_id, arch_key=key, family="MLP", hparams=hp))
        arch_id += 1

    # ---- TCN ----
    for n_blocks, channels, kernel, dilation in product(
        tuple(arch_cfg.tcn_blocks),
        tuple(arch_cfg.tcn_channels),
        tuple(arch_cfg.tcn_kernel),
        tuple(arch_cfg.tcn_dilation),
    ):
        hp = {
            "n_blocks": int(n_blocks),
            "channels": int(channels),
            "kernel": int(kernel),
            "dilation": int(dilation),
        }
        key = make_arch_key("TCN", hp)
        specs.append(ArchSpec(arch_id=arch_id, arch_key=key, family="TCN", hparams=hp))
        arch_id += 1

    # ---- GRU ----
    for n_layers, hidden_dim, dropout in product(
        tuple(arch_cfg.gru_layers),
        tuple(arch_cfg.gru_hidden),
        tuple(arch_cfg.gru_dropout),
    ):
        hp = {"n_layers": int(n_layers), "hidden_dim": int(hidden_dim), "dropout": float(dropout)}
        key = make_arch_key("GRU", hp)
        specs.append(ArchSpec(arch_id=arch_id, arch_key=key, family="GRU", hparams=hp))
        arch_id += 1

    # ---- Sanity ----
    total = int(arch_cfg.total_size())
    if len(specs) != total:
        raise RuntimeError(f"A_base size mismatch: got {len(specs)} but arch_cfg.total_size()={total}")
    # Ensure stable & unique keys
    keys = [s.arch_key for s in specs]
    if len(set(keys)) != len(keys):
        raise RuntimeError("Duplicate arch_key detected in A_base. This breaks reproducibility.")

    return specs
