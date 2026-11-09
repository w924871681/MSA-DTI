# -*- coding: utf-8 -*-
"""Runtime guards for the Stage-2 admission experiments.

On some Windows/WDDM + cuDNN combinations, the first GRU backward pass can
terminate the Python process with Windows status 0xC0000409 instead of raising
an ordinary Python/CUDA exception.  That process-level failure cannot be
caught by ``try/except``.  The Stage-2 workflow therefore uses the native
PyTorch CUDA GRU path by default (cuDNN disabled only while a GRU candidate is
adapted/evaluated).  A conservative GRU-on-CPU mode is also available.

These guards do not change the architecture, loss, optimizer, number of
updates, data split, or selection protocol.
"""
from __future__ import annotations

import contextlib
from typing import ContextManager

import torch


SAFE_MODES = ("default", "gru-native", "gru-cpu")


def normalize_safe_mode(value: str) -> str:
    mode = str(value).strip().lower()
    aliases = {
        "native": "gru-native",
        "safe": "gru-native",
        "cpu-gru": "gru-cpu",
        "grucpu": "gru-cpu",
    }
    mode = aliases.get(mode, mode)
    if mode not in SAFE_MODES:
        raise ValueError(f"Unknown Stage-2 safe mode: {value!r}; expected one of {SAFE_MODES}")
    return mode


def configure_stage2_runtime(device: torch.device, safe_mode: str) -> str:
    """Configure deterministic and conservative CUDA backend settings."""
    mode = normalize_safe_mode(safe_mode)
    if device.type == "cuda":
        # Dynamic algorithm benchmarking is unnecessary for the tiny K-shot
        # tensors and is a known source of Windows/WDDM instability.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
        torch.cuda.empty_cache()
    return mode


def candidate_device(spec, requested_device: torch.device, safe_mode: str) -> torch.device:
    """Return the device used for one candidate.

    ``gru-cpu`` changes only GRU execution.  All other families remain on the
    requested device.  ``gru-native`` keeps GRU on CUDA but disables cuDNN in
    :func:`candidate_backend_context`.
    """
    mode = normalize_safe_mode(safe_mode)
    if (
        requested_device.type == "cuda"
        and str(getattr(spec, "family", "")) == "GRU"
        and mode == "gru-cpu"
    ):
        return torch.device("cpu")
    return requested_device


def candidate_backend_context(spec, actual_device: torch.device, safe_mode: str) -> ContextManager:
    """Backend context for one candidate's adaptation and evaluation."""
    mode = normalize_safe_mode(safe_mode)
    if (
        actual_device.type == "cuda"
        and str(getattr(spec, "family", "")) == "GRU"
        and mode == "gru-native"
    ):
        # Native PyTorch CUDA GRU is slower than cuDNN but avoids the native
        # Windows process abort observed before Python can raise an exception.
        return torch.backends.cudnn.flags(enabled=False)
    return contextlib.nullcontext()


def synchronize_if_cuda(device: torch.device) -> None:
    """Surface asynchronous CUDA errors at the current candidate boundary."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
