# core/space/models.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import ArchSpec


# -----------------------------
# MLP forecaster
# -----------------------------
class MLPForecaster(nn.Module):
    def __init__(self, input_dim: int, L: int, H: int, n_layers: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.input_dim = int(input_dim)
        self.L = int(L)
        self.H = int(H)

        in_dim = self.L * self.input_dim
        layers = []
        d_in = in_dim
        for _ in range(int(n_layers)):
            layers.append(nn.Linear(d_in, int(hidden_dim)))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            d_in = int(hidden_dim)
        layers.append(nn.Linear(d_in, self.H))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        if l != self.L:
            raise ValueError(f"MLPForecaster expects L={self.L}, but got {l}")
        if d != self.input_dim:
            raise ValueError(f"MLPForecaster expects input_dim={self.input_dim}, but got {d}")
        z = x.reshape(b, l * d)
        return self.net(z)


# -----------------------------
# GRU forecaster
# last/mean/max temporal pooling + LN + MLP head
# -----------------------------
class GRUForecaster(nn.Module):
    def __init__(self, input_dim: int, H: int, n_layers: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.input_dim = int(input_dim)
        self.H = int(H)
        self.n_layers = int(n_layers)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        self.rnn = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=self.dropout if self.n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(self.hidden_dim, self.H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, h_n = self.rnn(x)          # out: (B, L, hidden)
        last = out[:, -1, :]            # (B, hidden)
        y = self.head(last)             # (B, H)
        return y


# -----------------------------
# TCN block
# block-level increasing dilation + GroupNorm
# -----------------------------
class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.kernel = int(kernel)
        self.dilation = int(dilation)

        # left padding for "causal" behavior
        self.pad = (self.dilation * (self.kernel - 1), 0)
        self.conv1 = nn.Conv1d(self.in_ch, self.out_ch, kernel_size=self.kernel, dilation=self.dilation)
        self.conv2 = nn.Conv1d(self.out_ch, self.out_ch, kernel_size=self.kernel, dilation=self.dilation)
        self.res = nn.Conv1d(self.in_ch, self.out_ch, kernel_size=1) if self.in_ch != self.out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.pad(x, self.pad)
        z = F.relu(self.conv1(z))
        z = F.pad(z, self.pad)
        z = self.conv2(z)
        r = self.res(x)
        return F.relu(z + r)


class TCNForecaster(nn.Module):
    def __init__(self, input_dim: int, H: int, n_blocks: int, channels: int, kernel: int, dilation: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.H = int(H)
        self.n_blocks = int(n_blocks)
        self.channels = int(channels)
        self.kernel = int(kernel)
        self.dilation = int(dilation)

        blocks = []
        in_ch = self.input_dim
        for _ in range(self.n_blocks):
            blocks.append(_TCNBlock(in_ch=in_ch, out_ch=self.channels, kernel=self.kernel, dilation=self.dilation))
            in_ch = self.channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(self.channels, self.H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)          # (B, D, L)
        z = self.blocks(z)             # (B, C, L)
        last = z[:, :, -1]             # (B, C)
        y = self.head(last)            # (B, H)
        return y


# -----------------------------
# Public builder
# -----------------------------
def build_model(
    arch_spec: ArchSpec,
    input_dim: int,
    H: int,
    L: int = 96,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    fam = arch_spec.family
    hp = arch_spec.hparams

    if fam == "MLP":
        m = MLPForecaster(
            input_dim=int(input_dim),
            L=int(L),
            H=int(H),
            n_layers=int(hp["n_layers"]),
            hidden_dim=int(hp["hidden_dim"]),
            dropout=float(hp["dropout"]),
        )
    elif fam == "TCN":
        m = TCNForecaster(
            input_dim=int(input_dim),
            H=int(H),
            n_blocks=int(hp["n_blocks"]),
            channels=int(hp["channels"]),
            kernel=int(hp["kernel"]),
            dilation=int(hp["dilation"]),
        )
    elif fam == "GRU":
        m = GRUForecaster(
            input_dim=int(input_dim),
            H=int(H),
            n_layers=int(hp["n_layers"]),
            hidden_dim=int(hp["hidden_dim"]),
            dropout=float(hp["dropout"]),
        )
    else:
        raise ValueError(f"Unknown family={fam}")

    if dtype is not None:
        m = m.to(dtype=dtype)
    if device is not None:
        m = m.to(device=device)
    return m