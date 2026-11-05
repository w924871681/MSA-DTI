# core/methods/ours/weight_bank.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import inspect
from dataclasses import dataclass
from typing import Dict, Tuple, Any, List, Optional, Literal

import torch
import torch.nn as nn
import torch.optim as optim

from core.space import build_model


# ============================================================
# Meta info
# ============================================================
@dataclass
class BankMeta:
    space_fingerprint: str
    input_dim: int
    H_list: List[int]


# ============================================================
# Key policy (single source of truth)
#   - rich   : avoid cross-setting pollution (H,K,tier,ctype,input_dim,L,arch)
#   - shared : shared θ0 base for meta-train (H,input_dim,L,arch)
#   - legacy : backward compatibility (H::arch)
# ============================================================
def make_bank_key(
    *,
    H: int,
    arch_key: str,
    K: int | None = None,
    tier: str | None = None,
    center_type: str | None = None,
    input_dim: int | None = None,
    L: int | None = None,
) -> str:
    """Rich key to avoid cross-setting pollution."""
    parts = [f"H{int(H)}"]
    if K is not None:
        parts.append(f"K{int(K)}")
    if tier is not None:
        parts.append(f"T{str(tier)}")
    if center_type is not None:
        parts.append(f"C{str(center_type)}")
    if input_dim is not None:
        parts.append(f"D{int(input_dim)}")
    if L is not None:
        parts.append(f"L{int(L)}")
    parts.append(str(arch_key))
    return "::".join(parts)


def make_bank_key_shared(*, H: int, arch_key: str, input_dim: int | None = None, L: int | None = None) -> str:
    """
    Shared base θ0 key used by meta-train.
    Rationale: meta-train learns a shared initialization per (H, arch, input_dim, L),
    not split by (K, tier, center_type), to keep cost bounded and match PPT "shared base".
    """
    return make_bank_key(
        H=int(H),
        arch_key=str(arch_key),
        K=None,
        tier=None,
        center_type=None,
        input_dim=None if input_dim is None else int(input_dim),
        L=None if L is None else int(L),
    )


def make_bank_key_legacy(H: int, arch_key: str) -> str:
    return f"H{int(H)}::{str(arch_key)}"


HitLevel = Literal["rich", "shared", "legacy", "miss"]


def resolve_bank_key(
    bank: Dict[str, Dict[str, torch.Tensor]],
    *,
    H: int,
    arch_key: str,
    K: int | None = None,
    tier: str | None = None,
    center_type: str | None = None,
    input_dim: int | None = None,
    L: int | None = None,
) -> Tuple[str | None, HitLevel]:
    """
    Unified lookup policy (runner should call this, or replicate same order):
      1) rich   : (H,K,tier,ctype,D,L,arch)
      2) shared : (H,D,L,arch)
      3) legacy : (H::arch)
    """
    k_rich = make_bank_key(H=H, arch_key=arch_key, K=K, tier=tier, center_type=center_type, input_dim=input_dim, L=L)
    if k_rich in bank:
        return k_rich, "rich"

    k_shared = make_bank_key_shared(H=H, arch_key=arch_key, input_dim=input_dim, L=L)
    if k_shared in bank:
        return k_shared, "shared"

    k_legacy = make_bank_key_legacy(H, arch_key)
    if k_legacy in bank:
        return k_legacy, "legacy"

    return None, "miss"


# ============================================================
# Robust build_model adapter (signature-agnostic)
# ============================================================
def _build_model_auto(
    spec,
    *,
    input_dim: int,
    H: int,
    L: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Compatible with variants of core.space.build_model.

    Common signatures:
      - build_model(spec, input_dim, H)
      - build_model(spec, input_dim, H, L)
      - build_model(spec, input_dim, H, L, device=...)
      - build_model(spec, input_dim=..., H=..., L=..., device=...)
    """
    fn = build_model
    dev_str = str(device) if device is not None else None

    # Try signature-based kwargs
    try:
        sig = inspect.signature(fn)
        names = list(sig.parameters.keys())
        kwargs: Dict[str, Any] = {}

        for k in names:
            lk = k.lower()
            if lk in ("spec", "arch", "arch_spec"):
                kwargs[k] = spec
            elif lk in ("input_dim", "din", "d_in"):
                kwargs[k] = int(input_dim)
            elif lk in ("h", "horizon"):
                kwargs[k] = int(H)
            elif lk == "l":
                if L is not None:
                    kwargs[k] = int(L)
            elif lk == "device":
                if dev_str is not None:
                    kwargs[k] = dev_str

        m = fn(**kwargs)
        if device is not None:
            m = m.to(device)
        return m
    except Exception:
        pass

    # Positional tries
    tries = [(spec, int(input_dim), int(H))]
    if L is not None:
        tries.append((spec, int(input_dim), int(H), int(L)))

    for args in tries:
        try:
            m = fn(*args)
            if device is not None:
                m = m.to(device)
            return m
        except Exception:
            continue

    # Final keyword try
    try:
        if L is not None and dev_str is not None:
            return fn(spec, input_dim=int(input_dim), H=int(H), L=int(L), device=dev_str)
    except Exception:
        pass

    raise RuntimeError("build_model signature not compatible with _build_model_auto attempts.")


# ============================================================
# IO
# ============================================================
def load_weight_bank(path: str, map_location: str | torch.device = "cpu") -> Tuple[BankMeta, Dict[str, Dict[str, torch.Tensor]]]:
    obj = torch.load(path, map_location=map_location)
    meta = BankMeta(**obj["meta"])
    bank = obj["bank"]
    return meta, bank


def save_weight_bank(path: str, meta: BankMeta, bank: Dict[str, Dict[str, torch.Tensor]]) -> None:
    """
    Safe save: avoid makedirs("") crash on Windows if dirname is empty.
    """
    path = str(path)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

    torch.save({
        "meta": {
            "space_fingerprint": meta.space_fingerprint,
            "input_dim": int(meta.input_dim),
            "H_list": list(meta.H_list),
        },
        "bank": bank,
    }, path)


# ============================================================
# Initialization
# ============================================================
def init_bank_from_space(
    A_specs,
    *,
    input_dim: int,
    H_list: List[int],
    L: int,
    device: torch.device,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Create initial θ0 by random init from build_model.
    Store as CPU tensors for portability.

    We write three keys:
      - rich  : includes (H,D,L,arch)  [K/tier/ctype not available at init]
      - shared: identical to above (H,D,L,arch)  [explicit name]
      - legacy: H::arch
    """
    bank: Dict[str, Dict[str, torch.Tensor]] = {}
    for H in list(H_list):
        for spec in A_specs:
            model = _build_model_auto(spec, input_dim=int(input_dim), H=int(H), L=int(L), device=device)
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            arch_key = str(spec.arch_key)
            k_shared = make_bank_key_shared(H=int(H), arch_key=arch_key, input_dim=int(input_dim), L=int(L))
            k_legacy = make_bank_key_legacy(int(H), arch_key)

            # At init time we cannot attach K/tier/ctype, so "rich" degenerates to shared.
            k_rich_degen = make_bank_key(H=int(H), arch_key=arch_key, input_dim=int(input_dim), L=int(L))

            bank[k_rich_degen] = state
            bank.setdefault(k_shared, state)
            bank.setdefault(k_legacy, state)

    return bank


# ============================================================
# Reptile meta-train (updates shared θ0 only)
# ============================================================
def reptile_meta_train(
    *,
    A_specs,
    bank: Dict[str, Dict[str, torch.Tensor]],
    input_dim: int,
    H_list: List[int],
    L: int,
    # tasks
    train_center_ids: List[int],
    get_task_split_fn,   # callable(cfg, cache, cid, H, K) -> Xs,ys,...
    cfg,
    cache,
    # hyperparams
    meta_epochs: int,
    meta_tasks_per_epoch: int,
    archs_per_task: int,
    inner_steps: int,
    inner_lr: float,
    meta_step_size: float,
    meta_seed: int,
    device: torch.device,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Reptile: θ <- θ + eps*(φ - θ), where φ is post-inner-loop weights.

    IMPORTANT POLICY:
      - We update only SHARED keys (H,input_dim,L,arch).
      - This matches the "shared base θ0" narrative and avoids exploding bank variants.
      - Runner can still save rich keys during test-time adaptation if you later choose to.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(int(meta_seed))

    def _rand_int(low: int, high: int) -> int:
        return int(torch.randint(low=low, high=high, size=(1,), generator=g).item())

    A_size = len(A_specs)
    H_list = list(H_list)

    for ep in range(int(meta_epochs)):
        for _it in range(int(meta_tasks_per_epoch)):
            cid = train_center_ids[_rand_int(0, len(train_center_ids))]
            H = H_list[_rand_int(0, len(H_list))]
            K_values = list(map(int, cfg.main.task.K_list))
            if bool(getattr(cfg.method, "meta_use_all_support_sizes", True)):
                K = K_values[_rand_int(0, len(K_values))]
            else:
                K = int(K_values[0])

            Xs, ys, *_rest = get_task_split_fn(cfg, cache, cid, int(H), int(K))
            Xs = Xs.to(device)
            ys = ys.to(device)

            arch_ids = torch.randperm(A_size, generator=g)[:min(int(archs_per_task), A_size)].tolist()

            for ai in arch_ids:
                spec = A_specs[ai]
                arch_key = str(spec.arch_key)

                # Always update shared key
                key = make_bank_key_shared(H=int(H), arch_key=arch_key, input_dim=int(input_dim), L=int(L))

                # Fallback for old banks
                if key not in bank:
                    k_degen = make_bank_key(H=int(H), arch_key=arch_key, input_dim=int(input_dim), L=int(L))
                    if k_degen in bank:
                        key = k_degen
                    else:
                        k_leg = make_bank_key_legacy(int(H), arch_key)
                        if k_leg in bank:
                            key = k_leg
                        else:
                            # missing entry: skip (or you could init here)
                            continue

                model = _build_model_auto(spec, input_dim=int(input_dim), H=int(H), L=int(L), device=device)
                model.load_state_dict(bank[key], strict=True)

                opt = optim.SGD(model.parameters(), lr=float(inner_lr), momentum=0.0)
                model.train()

                for _ in range(int(inner_steps)):
                    opt.zero_grad(set_to_none=True)
                    pred = model(Xs)
                    loss = torch.mean((pred - ys) ** 2)
                    loss.backward()
                    opt.step()

                new_state = model.state_dict()
                old_state = bank[key]
                updated = {}
                eps = float(meta_step_size)
                for k, v in new_state.items():
                    v_cpu = v.detach().cpu()
                    updated[k] = old_state[k] + eps * (v_cpu - old_state[k])

                bank[key] = updated

                # Keep legacy mirrored for backward compatibility (optional but helpful)
                bank.setdefault(make_bank_key_legacy(int(H), arch_key), updated)

        print(f"[WeightBank] meta-epoch {ep+1}/{meta_epochs} done.")

    return bank
