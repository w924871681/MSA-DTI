"""Structure-Preserving Adaptation (SPA) for MSA-DTI.

Exact standalone implementation of the frozen SPA instantiation pipeline:
  1. Guarded 5-state early adaptation (`base02`: steps 50, 25, 10, midpoint, 0)
  2. Support-only initial readout selection (22-config raw/bias/affine menu)
  3. Optional in-place single-channel causal load routing (`orig`, `low1`, `low5`)
     using leave-one-channel-out ridge importance with exact SHA-256 topology
     invariance (+0 parameters, +0 operations)
  4. Multi-scale load-decomposed output calibration (`orig`: 1,089 candidates
     including single-scale/joint legacy families; `low1`/`low5`: 928 candidates
     including `r_base`, `r_mix`, `r_separate`, `r_reg`) folded algebraically
     into the existing linear output layer.
"""
from __future__ import annotations

import hashlib
import json
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .config import FROZEN_HYPERPARAMETERS

R62_LAMBDAS: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
R62_AMPLITUDES: tuple[float, ...] = (0.25, 0.5, 1.0)
R63_WIDTHS: tuple[int, ...] = (1, 5, 20, 96)
R63_TAU: tuple[float, ...] = (0.25, 0.5, 1.0)
R63_MORE: tuple[float, ...] = (0.125, 0.375, 0.625, 0.75, 0.875)
SPA_ELIGIBLE_FAMILIES: tuple[str, ...] = (
    "legacy",
    "legacy_special",
    "r_base",
    "r_mix",
    "r_reg",
    "r_separate",
)


def _scales(x: np.ndarray) -> np.ndarray:
    return np.maximum(x.std(axis=0), 1e-6)


def _normalizer(h: np.ndarray) -> tuple[np.ndarray, float]:
    mu = h.mean(axis=0)
    return mu, float(max(float(np.square(h - mu).mean()), 1e-12) ** 0.5)


def augment_head(W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Embed (W [H, d], b [H]) into [H, d + 5] = [W, 0_4, b] float64."""
    w = np.asarray(W, dtype=np.float64)
    bias = np.asarray(b, dtype=np.float64)
    return np.column_stack((w, np.zeros((len(bias), 4), dtype=np.float64), bias))


def eval_head(h: np.ndarray, u: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Evaluate augmented linear head params [H, d + 5] on (h [N, d], u [N, 4])."""
    d = h.shape[1]
    return h @ params[:, :d].T + u @ params[:, d:-1].T + params[:, -1]


def causal_load_summaries(x_arr: np.ndarray, cpu_col: int = 3) -> np.ndarray:
    """Compute the four causal CPU load summaries (1, 5, 20, 96 steps) from input windows."""
    cpu = np.asarray(x_arr[:, :, cpu_col], dtype=np.float64)
    return np.column_stack(
        (
            cpu[:, -1],
            cpu[:, -5:].mean(axis=1),
            cpu[:, -20:].mean(axis=1),
            cpu.mean(axis=1),
        )
    )


def per_horizon_mse(
    h_aug: np.ndarray,
    y: np.ndarray,
    original_pred: np.ndarray,
    ref_w: np.ndarray,
    ref_b: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
) -> np.ndarray:
    """Exact quadratic per-horizon validation MSE preserving the measured forward anchor."""
    n = len(h_aug)
    design = np.column_stack((h_aug, np.ones(n, dtype=np.float64)))
    error = original_pred - y
    gram = design.T @ design / n
    cross = error.T @ design / n
    base = np.square(error).mean(axis=0)
    diff = np.concatenate(
        (
            weights.astype(np.float64) - ref_w,
            (biases.astype(np.float64) - ref_b)[:, :, None],
        ),
        axis=-1,
    )
    flat = diff.reshape(-1, diff.shape[-1])
    quadratic = ((flat @ gram) * flat).sum(axis=1).reshape(diff.shape[:2])
    scores = base + 2.0 * (diff * cross).sum(axis=-1) + quadratic
    if not np.isfinite(scores).all() or float(scores.min()) < -1e-9:
        raise ArithmeticError("Non-finite or negative head risk encountered")
    return np.maximum(scores, 0.0)


def project_known(h: np.ndarray, outputs: np.ndarray, penalty: float = 0.01) -> np.ndarray:
    """Ridge projection from hidden features h [N, d] to target summaries outputs [N, M]."""
    mu, s = _normalizer(h)
    z = (h - mu) / s
    mat = z.T @ z / len(z) + penalty * np.eye(z.shape[1])
    coef = np.linalg.solve(mat, z.T @ (outputs - outputs.mean(axis=0)) / len(z)) / s
    return augment_head(coef.T, outputs.mean(axis=0) - mu @ coef)


def head_ref(model: nn.Module) -> tuple[str, nn.Linear]:
    """Return the name and module of the final linear output layer."""
    found = [(k, m) for k, m in model.named_modules() if isinstance(m, nn.Linear)]
    if not found:
        raise ValueError("Model does not contain an nn.Linear output layer")
    name, layer = found[-1]
    if layer.bias is None:
        raise ValueError("Final nn.Linear layer must include bias")
    return name, layer


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def equal_midpoint_state(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Exact elementwise midpoint 0.5 * (state_a + state_b) for guarded early stopping."""
    out: dict[str, torch.Tensor] = {}
    for k, va in state_a.items():
        vb = state_b[k]
        if torch.is_floating_point(va):
            out[k] = (0.5 * (va.detach().cpu() + vb.detach().cpu())).to(va.dtype)
        else:
            out[k] = va.detach().cpu().clone()
    return out


def extract_hidden_and_pred(
    model: nn.Module,
    x: torch.Tensor,
    batch_size: int = 256,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run forward pass and capture penultimate hidden features and output head parameters."""
    name, head = head_ref(model)
    features: list[np.ndarray] = []
    outputs: list[torch.Tensor] = []

    def _hook(_layer: nn.Module, inputs: tuple[torch.Tensor, ...], y: torch.Tensor) -> None:
        features.append(inputs[0].detach().cpu().numpy())
        outputs.append(y.detach())

    handle = head.register_forward_hook(_hook)
    preds: list[np.ndarray] = []
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                p = model(x[start : start + batch_size])
                outputs.pop()
                preds.append(p.detach().cpu().numpy())
    finally:
        handle.remove()

    h = np.concatenate(features, axis=0)
    p_arr = np.concatenate(preds, axis=0)
    W = head.weight.detach().cpu().numpy().copy()
    b = head.bias.detach().cpu().numpy().copy()
    return name, h, p_arr, W, b


def early_stop_base02(
    model: nn.Module,
    source_state: dict[str, torch.Tensor],
    xs: torch.Tensor,
    ys: torch.Tensor,
    xv: torch.Tensor,
    yv: np.ndarray,
    seed: int,
    lr: float = 0.02,
    steps: int = 50,
    grad_clip: float = 1.0,
) -> tuple[str, dict[str, torch.Tensor], dict[str, float]]:
    """Guarded 5-state early adaptation (`base02`: b50, b25, b10, bW midpoint, a0)."""
    model.load_state_dict(source_state, strict=True)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    opt = torch.optim.SGD(model.parameters(), lr=float(lr))
    snaps: dict[str, dict[str, torch.Tensor]] = {"a0": {k: v.detach().cpu().clone() for k, v in source_state.items()}}
    for step_idx in range(1, int(steps) + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = ((model(xs) - ys) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip), error_if_nonfinite=True)
        opt.step()
        if step_idx in (10, 25, 50):
            snaps[f"b{step_idx}"] = clone_state_dict(model)
    snaps["bW"] = equal_midpoint_state(snaps["a0"], snaps["b50"])

    order = ("b50", "b25", "b10", "bW", "a0")
    yv64 = np.asarray(yv, dtype=np.float64)
    mses: dict[str, float] = {}
    for nm in order:
        model.load_state_dict(snaps[nm], strict=True)
        _, _, pv, _, _ = extract_hidden_and_pred(model, xv)
        mses[nm] = float(np.square(pv.astype(np.float64) - yv64).mean())

    min_mse = min(mses.values())
    best_name = next(nm for nm in order if mses[nm] <= min_mse + 1e-14)
    return best_name, snaps[best_name], mses


def fit_initial_readout_r62(
    hs: np.ndarray,
    ps: np.ndarray,
    ys: np.ndarray,
    hv: np.ndarray,
    pv: np.ndarray,
    yv: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Fit the 22-config support-only initial readout menu (`raw`, `bias`, `affine`)."""
    dim = hs.shape[1]
    out = ys.shape[1]
    residual = ys - ps
    bias = residual.mean(axis=0)
    zero = np.zeros((out, dim), dtype=np.float64)

    states: list[tuple[str, str, np.ndarray, np.ndarray]] = []

    def _add(fam: str, lab: str, dw: np.ndarray, db: np.ndarray, factor: float = 1.0) -> None:
        states.append(
            (
                fam,
                lab,
                (W + factor * dw).astype(np.float32),
                (b + factor * db).astype(np.float32),
            )
        )

    _add("raw", "raw", zero, np.zeros(out, dtype=np.float64))
    for i, t in enumerate(R62_AMPLITUDES):
        _add("bias", f"bias_{i}", zero, bias, t)

    pc = ps - ps.mean(axis=0)
    var = np.square(pc).mean(axis=0)
    cov = (pc * (residual - bias)).mean(axis=0)
    for j, lam in enumerate(R62_LAMBDAS):
        slope = np.divide(cov, (1.0 + lam) * var, out=np.zeros_like(cov), where=var > 1e-14)
        dw = slope[:, None] * W
        db = bias + slope * (b - ps.mean(axis=0))
        for i, t in enumerate(R62_AMPLITUDES):
            _add("affine", f"affine_{j}_{i}", dw, db, t)

    _, labels, weights, biases = zip(*states, strict=True)
    w_bank = np.stack(weights)
    b_bank = np.stack(biases)
    scores = per_horizon_mse(hv, yv, pv, W, b, w_bank, b_bank).mean(axis=1)
    best_idx = int(np.flatnonzero(scores <= scores.min() + 1e-14)[0])
    return w_bank[best_idx], b_bank[best_idx], str(labels[best_idx]), float(scores[best_idx])


def r63_native_proposals(
    hs: np.ndarray,
    hv: np.ndarray,
    ps: np.ndarray,
    pv: np.ndarray,
    ys: np.ndarray,
    us: np.ndarray,
    uv: np.ndarray,
    base_aug: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the 161 frozen R63 single-scale and joint readout candidates for `orig`."""
    H, dim = ys.shape[1], hs.shape[1]
    params = [base_aug]
    families = ["raw"]
    actions = ["raw"]

    def _append(group: str, identity: str, value: np.ndarray) -> None:
        families.append(group)
        actions.append(identity)
        params.append(value)

    hf, uf, pf = hv, uv, pv
    mh = hf.mean(axis=0)
    scale = float(max(float(np.square(hf - mh).mean()), 1e-12) ** 0.5)
    z = (hf - mh) / scale
    z_s = (hs - mh) / scale
    eye = np.eye(dim, dtype=np.float64)
    coef = np.linalg.solve(z.T @ z / len(z) + 0.01 * eye, z.T @ (uf - uf.mean(axis=0)) / len(z)) / scale
    allq = np.column_stack((coef.T, np.zeros((4, 4), dtype=np.float64), uf.mean(axis=0) - mh @ coef))
    q_f = eval_head(hf, uf, allq)
    q_s = eval_head(hs, us, allq)

    for w_idx, width in enumerate(R63_WIDTHS):
        qm = q_f[:, w_idx]
        qs = q_s[:, w_idx]
        qparam = np.repeat(allq[w_idx : w_idx + 1], H, axis=0)
        centered = qm - qm.mean()
        var = float(np.square(centered).mean())
        beta = (centered[:, None] * (pf - pf.mean(axis=0))).mean(axis=0) / var if var > 1e-12 else np.zeros(H)
        component = beta[:, None] * qparam
        component[:, -1] += pf.mean(axis=0) - beta * qm.mean()

        for shift in (0, 1):
            teacher = qparam.copy()
            if shift:
                teacher[:, -1] += ys.mean(axis=0) - qs.mean()
            for t in R63_TAU + R63_MORE:
                grp = "mix" if t in R63_TAU else "wide"
                _append(f"V_{grp}", f"V_{grp}_{width}_{shift}_{t}", (1.0 - t) * base_aug + t * teacher)
            for t in R63_TAU:
                _append("V_orth", f"V_orth_{width}_{shift}_{t}", base_aug + t * (teacher - component))

        residual = ys - ps
        target = residual - residual.mean(axis=0)
        smean = np.column_stack((ps.mean(axis=0), np.full(H, qs.mean())))
        sdev = np.maximum(np.column_stack((pf.std(axis=0), np.full(H, np.sqrt(var)))), 1e-6)
        for lam in (0.01, 1.0, 100.0):
            correction = np.zeros_like(base_aug)
            for j in range(H):
                xd = (np.column_stack((ps[:, j], qs)) - smean[j]) / sdev[j]
                reg = np.linalg.solve(xd.T @ xd / len(xd) + lam * np.eye(2), xd.T @ target[:, j] / len(xd)) / sdev[j]
                correction[j] = reg[0] * base_aug[j] + reg[1] * qparam[j]
                correction[j, -1] += residual[:, j].mean() - smean[j] @ reg
            for t in R63_TAU:
                _append("V_reg", f"V_reg_{width}_{lam}_{t}", base_aug + t * correction)

        xf = np.column_stack((z, np.ones(len(z), dtype=np.float64)))
        xs_d = np.column_stack((z_s, np.ones(len(z_s), dtype=np.float64)))
        penalty = np.diag(np.r_[np.full(dim, 0.1), 0.0])
        for g in (0.1, 1.0, 10.0):
            lhs = xs_d.T @ xs_d / len(xs_d) + g * xf.T @ xf / len(xf) + penalty
            rhs = xs_d.T @ (ys - ps) / len(xs_d) + g * xf.T @ (uf[:, w_idx, None] - pf) / len(xf)
            bstd = np.linalg.solve(lhs, rhs)
            delta = augment_head(bstd[:-1].T / scale, bstd[-1] - mh @ (bstd[:-1] / scale))
            for t in R63_TAU:
                _append("V_joint", f"V_joint_{width}_{g}_{t}", base_aug + t * delta)

    fam_arr = np.asarray(families)
    act_arr = np.asarray(actions)
    par_arr = np.stack(params).astype(np.float32)

    # Order identically to R64 heads64.earlier_options(native_simple + V_orth)
    legacy_simple = ("raw", "V_mix", "V_wide", "V_reg", "V_joint")
    ix_simple = np.concatenate([np.flatnonzero(fam_arr == nm) for nm in legacy_simple])
    ix_orth = np.flatnonzero(fam_arr == "V_orth")
    ids = np.concatenate((ix_simple, ix_orth))
    kind = np.array(["legacy"] * len(ix_simple) + ["legacy_special"] * len(ix_orth))
    return kind, act_arr[ids], par_arr[ids]


def channel_importance(
    h_val: np.ndarray,
    W_ref: np.ndarray,
    ridge: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-channel-out ridge reconstruction importance score I_j (Eq. 15)."""
    x = np.asarray(h_val, dtype=np.float64).copy()
    x -= x.mean(axis=0)
    cov = x.T @ x / len(x)
    d = cov.shape[0]
    w_norm_sq = np.square(np.asarray(W_ref, dtype=np.float64)).sum(axis=0)
    variance = np.square(x).mean(axis=0)
    if d == 1:
        return w_norm_sq * variance, w_norm_sq * variance

    residual_map = np.eye(d, dtype=np.float64)
    for j in range(d):
        keep = np.arange(d) != j
        part = cov[np.ix_(keep, keep)].copy()
        lam = float(ridge) * max(float(np.trace(part)) / (d - 1), 1e-12)
        part.flat[::d] += lam
        beta = np.linalg.solve(part, cov[keep, j])
        residual_map[keep, j] = -beta
    residual = x @ residual_map
    return w_norm_sq * np.square(residual).mean(axis=0), w_norm_sq * variance


def routable_channels(
    model: nn.Module,
    family: str,
    hidden_dim: int,
    feature_col: int,
) -> list[int]:
    """Return valid internal hidden channel indices for in-place routing."""
    if family == "MLP":
        hidden = [m.out_features for m in model.net if isinstance(m, nn.Linear)][:-1]
        return list(range(min(*hidden, hidden_dim))) if len(hidden) > 1 else list(range(min(hidden[0], hidden_dim)))
    if family == "GRU":
        return list(range(min(hidden_dim, model.rnn.hidden_size)))
    if family == "TCN":
        count = min([hidden_dim] + [b.out_ch for b in model.blocks])
        if isinstance(model.blocks[0].res, nn.Identity):
            return [int(feature_col)]
        return list(range(count))
    raise ValueError(f"Unsupported architecture family: {family}")


def topology_signature(model: nn.Module, seq_len: int = 96) -> dict:
    """Compute exact parameter count, dense MACs, and SHA-256 architecture topology digest."""
    ops = []
    mac = 0
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            cost = layer.in_features * layer.out_features
            ops.append([name, "Linear", layer.in_features, layer.out_features, cost])
            mac += cost
        elif isinstance(layer, nn.Conv1d):
            cost = seq_len * layer.out_channels * (layer.in_channels // layer.groups) * layer.kernel_size[0]
            ops.append([
                name, "Conv1d", layer.in_channels, layer.out_channels,
                layer.kernel_size[0], layer.groups, layer.stride[0], layer.dilation[0], cost,
            ])
            mac += cost
        elif isinstance(layer, nn.GRU):
            cost = sum(
                seq_len * 3 * layer.hidden_size * ((layer.input_size if l == 0 else layer.hidden_size) + layer.hidden_size)
                for l in range(layer.num_layers)
            )
            ops.append([name, "GRU", layer.input_size, layer.hidden_size, layer.num_layers, cost])
            mac += cost
        elif not list(layer.children()):
            ops.append([name, type(layer).__name__])
    shapes = [[n, list(t.shape), str(t.dtype)] for n, t in model.named_parameters()]
    desc = {
        "operations": ops,
        "parameters": shapes,
        "numel": int(sum(t.numel() for t in model.parameters())),
        "denseMultiplyAccumulates": int(mac),
        "sequenceLength": int(seq_len),
    }
    desc["sha"] = hashlib.sha256(json.dumps(desc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return desc


def rewrite_single_channel(
    model: nn.Module,
    base_state: dict[str, torch.Tensor],
    family: str,
    channel_idx: int,
    feature_col: int,
) -> tuple[dict[str, torch.Tensor], float]:
    """In-place rewrite of one existing hidden channel to route nonnegative target load.

    Preserves exact model topology, parameter count (+0), and MAC count (+0).
    Returns (rewritten_state_dict, channel_scale).
    """
    model.load_state_dict(base_state, strict=True)
    params = dict(model.named_parameters())

    def _set_row(key: str, ix: int, values: Sequence[tuple[int | tuple[int, int] | None, float]] = ()) -> None:
        target = params[key][ix]
        target.zero_()
        for pos, val in values:
            if pos is None:
                target.fill_(val)
            else:
                target[pos] = val

    with torch.no_grad():
        if family == "MLP":
            hidden = [(f"net.{n}", m) for n, m in enumerate(model.net) if isinstance(m, nn.Linear)][:-1]
            for n, (path, _) in enumerate(hidden):
                src_col = 95 * 25 + feature_col if n == 0 else channel_idx
                _set_row(f"{path}.weight", channel_idx, [(src_col, 1.0)])
                _set_row(f"{path}.bias", channel_idx)
            scale = 1.0
        elif family == "TCN":
            for n, block in enumerate(model.blocks):
                prefix = f"blocks.{n}."
                _set_row(prefix + "conv2.weight", channel_idx)
                _set_row(prefix + "conv2.bias", channel_idx)
                src = feature_col if n == 0 else channel_idx
                if isinstance(block.res, nn.Conv1d):
                    _set_row(prefix + "res.weight", channel_idx, [((src, 0), 1.0)])
                    _set_row(prefix + "res.bias", channel_idx)
                else:
                    if not (isinstance(block.res, nn.Identity) and src == channel_idx):
                        raise ValueError("Identity residual block requires channel_idx == feature_col")
            scale = 1.0
        elif family == "GRU":
            gru = model.rnn
            d = gru.hidden_size
            for layer_idx in range(gru.num_layers):
                for gate in range(3):
                    ix = gate * d + channel_idx
                    iv = [((feature_col if layer_idx == 0 else channel_idx), (0.05 if layer_idx == 0 else 1.0))] if gate == 2 else []
                    _set_row(f"rnn.weight_ih_l{layer_idx}", ix, iv)
                    _set_row(f"rnn.weight_hh_l{layer_idx}", ix)
                    _set_row(f"rnn.bias_ih_l{layer_idx}", ix, [(None, -20.0)] if gate == 1 else [])
                    _set_row(f"rnn.bias_hh_l{layer_idx}", ix)
            scale = 0.05
        else:
            raise ValueError(f"Unsupported architecture family: {family}")

    return clone_state_dict(model), scale


def reconstruct_response(
    h_val: np.ndarray,
    u_val: np.ndarray,
    ref_pred: np.ndarray,
    ref_head: np.ndarray,
    routed_channel: int | None,
) -> np.ndarray:
    """Eq. (16): Reconstruct unperturbed response head on unlabeled validation hidden states."""
    initial = ref_head.copy()
    if routed_channel is not None:
        initial[:, routed_channel] = 0.0
    return initial + project_known(h_val, ref_pred - eval_head(h_val, u_val, initial))


def _proposal_set_r64(
    hs: np.ndarray,
    hv: np.ndarray,
    us: np.ndarray,
    uv: np.ndarray,
    y_support: np.ndarray,
    p_s: np.ndarray,
    p_v: np.ndarray,
    base: np.ndarray,
    levels: np.ndarray,
    tag: str = "r_",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the 928 multi-scale candidates (`base`, `mix`, `separate`, `reg`)."""
    H = base.shape[0]
    qv = eval_head(hv, uv, levels)
    qs = eval_head(hs, us, levels)
    families = [tag + "base"]
    labels = [tag + "base"]
    weights = [base]

    def _add(fam: str, lab: str, par: np.ndarray) -> None:
        families.append(tag + fam)
        labels.append(tag + lab)
        weights.append(par)

    blend_grid = tuple(FROZEN_HYPERPARAMETERS["scale_blend_grid"])
    mix_grid = tuple(FROZEN_HYPERPARAMETERS["calibration_mix_grid"])
    gamma_grid = tuple(FROZEN_HYPERPARAMETERS["residual_gamma_grid"])

    for slow in range(1, 4):
        for fast in range(slow):
            x = np.column_stack((qv[:, slow], qv[:, fast] - qv[:, slow]))
            xm = x.mean(axis=0)
            sd = _scales(x)
            zc = (x - xm) / sd
            levels2 = np.stack([levels[slow], levels[fast] - levels[slow]])
            beta = np.linalg.solve(
                zc.T @ zc / len(zc) + 1e-6 * np.eye(2),
                zc.T @ (p_v - p_v.mean(axis=0)) / len(zc),
            ) / sd[:, None]
            comp = beta.T @ levels2
            comp[:, -1] += p_v.mean(axis=0) - xm @ beta

            for n, e in enumerate(blend_grid):
                lev = levels[slow] + e * (levels[fast] - levels[slow])
                means = y_support.mean(axis=0) - (qs[:, slow] + e * (qs[:, fast] - qs[:, slow])).mean()
                gg = np.repeat(lev[None, :], H, axis=0)
                for shift in (0, 1):
                    g = gg.copy()
                    if shift:
                        g[:, -1] += means
                    for k, t in enumerate(mix_grid):
                        suffix = f"{fast}{slow}_{n}{shift}{k}"
                        _add("mix", f"mix_{suffix}", (1.0 - t) * base + t * g)
                        mixed = (1.0 - t) * comp + t * g
                        for q_idx, gamma in enumerate(gamma_grid):
                            _add("separate", f"sep{q_idx}_{suffix}", mixed + gamma * (base - comp))

            rr = y_support - p_s
            target = rr - rr.mean(axis=0)
            for lam in (0.01, 1.0, 100.0):
                corr = np.zeros_like(base)
                for k in range(H):
                    vv = np.column_stack((p_v[:, k], qv[:, slow], qv[:, fast] - qv[:, slow]))
                    ss = np.column_stack((p_s[:, k], qs[:, slow], qs[:, fast] - qs[:, slow]))
                    dur = _scales(vv)
                    ms = ss.mean(axis=0)
                    zz = (ss - ms) / dur
                    a = np.linalg.solve(zz.T @ zz / len(zz) + lam * np.eye(3), zz.T @ target[:, k] / len(zz)) / dur
                    corr[k] = a[0] * base[k] + a[1:] @ levels2
                    corr[k, -1] += rr[:, k].mean() - ms @ a
                for k_idx, t in enumerate((0.25, 0.5, 1.0)):
                    _add("reg", f"reg_{fast}{slow}_{lam}_{k_idx}", base + t * corr)

    for lam in (0.01, 1.0, 100.0):
        corr = np.zeros_like(base)
        residual = y_support - p_s
        for j in range(H):
            vv = np.column_stack((p_v[:, j], qv))
            ss = np.column_stack((p_s[:, j], qs))
            sc = _scales(vv)
            mu = ss.mean(axis=0)
            z = (ss - mu) / sc
            beta = np.linalg.solve(z.T @ z / len(z) + lam * np.eye(5), z.T @ (residual[:, j] - residual[:, j].mean()) / len(z)) / sc
            corr[j] = beta[0] * base[j] + beta[1:] @ levels
            corr[j, -1] += residual[:, j].mean() - mu @ beta
        for k_idx, t in enumerate((0.25, 0.5, 1.0)):
            _add("reg", f"allreg_{lam}_{k_idx}", base + t * corr)

    return np.asarray(families), np.asarray(labels), np.stack(weights).astype(np.float32)


def multi_scale_calibration(
    body: str,
    hs: np.ndarray,
    hv: np.ndarray,
    us: np.ndarray,
    uv: np.ndarray,
    ys: np.ndarray,
    yv: np.ndarray,
    teacher_s: np.ndarray,
    teacher_v: np.ndarray,
    measured_pv: np.ndarray,
    ref_head: np.ndarray,
    orig_head: np.ndarray,
    legacy_bank: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    routed_channel: int | None = None,
    channel_scale: float = 1.0,
    routed_width: int = 0,
) -> dict:
    """Evaluate exact frozen R64 calibration menu for one candidate body (`orig`, `low1`, `low5`).

    Returns dict with `W`, `b`, `val_mse`, `rows`, `families`, `actions`, and `n_candidates`.
    """
    d = hs.shape[1]
    H = orig_head.shape[0]
    original = body == "orig"

    levels = project_known(hv, uv)
    if routed_channel is not None:
        w_idx = [1, 5, 20, 96].index(int(routed_width))
        levels[w_idx] = 0.0
        levels[w_idx, int(routed_channel)] = 1.0 / float(channel_scale)

    base = ref_head if original else reconstruct_response(hv, uv, teacher_v, ref_head, routed_channel)
    sx = teacher_s if original else eval_head(hs, us, base)
    vx = teacher_v if original else eval_head(hv, uv, base)

    fam_r, lab_r, theta_r = _proposal_set_r64(hs, hv, us, uv, ys, sx, vx, base, levels, tag="r_")
    scores_r = per_horizon_mse(
        np.column_stack((hv, uv)),
        yv,
        measured_pv,
        orig_head[:, :-1],
        orig_head[:, -1],
        theta_r[:, :, :-1],
        theta_r[:, :, -1],
    )

    if original:
        if legacy_bank is None:
            raise ValueError("legacy_bank (R63 native proposals) is required when body == 'orig'")
        lf, ll, lp, l_scores = legacy_bank
        kind = np.concatenate((lf, fam_r))
        lab = np.concatenate((ll, lab_r))
        params = np.concatenate((lp, theta_r))
        scores = np.vstack((l_scores, scores_r))
    else:
        kind, lab, params, scores = fam_r, lab_r, theta_r, scores_r

    ix = np.concatenate([np.flatnonzero(kind == nm) for nm in SPA_ELIGIBLE_FAMILIES if np.any(kind == nm)])
    subset = scores[ix]
    chosen_ids = ix[(subset <= subset.min(axis=0, keepdims=True) + 1e-14).argmax(axis=0)]
    hr = np.arange(H)
    theta = params[chosen_ids, hr]
    val_mse = float(scores[chosen_ids, hr].mean())
    if np.any(theta[:, d:-1]):
        raise RuntimeError("Non-zero external bypass weights detected in native SPA head")

    return {
        "body": body,
        "W": theta[:, :d].astype(np.float32),
        "b": theta[:, -1].astype(np.float32),
        "val_mse": val_mse,
        "rows": [int(i) for i in chosen_ids],
        "families": [str(kind[i]) for i in chosen_ids],
        "actions": [str(lab[i]) for i in chosen_ids],
        "n_candidates": int(len(ix)),
    }


def adapt_model_spa(
    model: nn.Module,
    family: str,
    base_state: dict[str, torch.Tensor],
    xs_np: np.ndarray,
    ys_np: np.ndarray,
    xv_np: np.ndarray,
    yv_np: np.ndarray,
    device: torch.device | str | None = None,
    us_np: np.ndarray | None = None,
    uv_np: np.ndarray | None = None,
) -> dict:
    """Run the complete frozen R62+R63+R64 SPA pipeline starting from a `base02` adapted model.

    Evaluates `orig`, `low1`, and `low5` on the target validation set, folds the winning
    readout algebraically into the existing final `nn.Linear` layer, verifies exact
    SHA-256 topology invariance (+0 parameters, +0 MACs), and returns the deployed
    state_dict together with full verification diagnostics.
    """
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model.to(dev)
    model.load_state_dict(base_state, strict=True)
    topo_before = topology_signature(model)

    xs_t = torch.as_tensor(xs_np, dtype=torch.float32, device=dev)
    xv_t = torch.as_tensor(xv_np, dtype=torch.float32, device=dev)
    ys64 = np.asarray(ys_np, dtype=np.float64)
    yv64 = np.asarray(yv_np, dtype=np.float64)
    us64 = causal_load_summaries(xs_np) if us_np is None else np.asarray(us_np, dtype=np.float64)
    uv64 = causal_load_summaries(xv_np) if uv_np is None else np.asarray(uv_np, dtype=np.float64)

    head_name, hs32, ps32, W0, b0 = extract_hidden_and_pred(model, xs_t)
    _, hv32, pv32, _, _ = extract_hidden_and_pred(model, xv_t)
    hs64, hv64 = hs32.astype(np.float64), hv32.astype(np.float64)
    ps64, pv64 = ps32.astype(np.float64), pv32.astype(np.float64)
    W64, b64 = W0.astype(np.float64), b0.astype(np.float64)
    d = hv64.shape[1]

    # Step 1: Frozen R62 22-config initial readout p(x)
    W_resp, b_resp, r62_act, r62_mse = fit_initial_readout_r62(hs64, ps64, ys64, hv64, pv64, yv64, W64, b64)
    orig_aug = augment_head(W0, b0)
    resp_aug = augment_head(W_resp, b_resp)
    teacher_s = ps64 + eval_head(hs64, us64, resp_aug - orig_aug)
    teacher_v = pv64 + eval_head(hv64, uv64, resp_aug - orig_aug)

    # Step 2: Frozen R63 161-candidate native legacy bank for `orig`
    lf, ll, lp = r63_native_proposals(hs64, hv64, teacher_s, teacher_v, ys64, us64, uv64, resp_aug)
    l_scores = per_horizon_mse(
        np.column_stack((hv64, uv64)),
        yv64,
        pv64,
        orig_aug[:, :-1],
        orig_aug[:, -1],
        lp[:, :, :-1],
        lp[:, :, -1],
    )
    legacy_bank = (lf, ll, lp, l_scores)

    # Step 3: Optional single-channel causal load routing (`low1` col 3, `low5` col 10)
    low_sc, _ = channel_importance(hv32, W_resp, ridge=float(FROZEN_HYPERPARAMETERS["channel_importance_ridge"]))
    routed_states: dict[str, dict[str, torch.Tensor]] = {"orig": {k: v.detach().cpu().clone() for k, v in base_state.items()}}
    routed_info: dict[str, dict] = {}
    body_results: list[dict] = []

    # Candidate body 1: unmodified backbone `orig`
    body_results.append(
        multi_scale_calibration(
            "orig", hs64, hv64, us64, uv64, ys64, yv64,
            teacher_s, teacher_v, pv64, resp_aug, orig_aug,
            legacy_bank=legacy_bank,
        )
    )

    # Candidate bodies 2 & 3: `low1` and `low5`
    for width, feat_col in ((1, 3), (5, 10)):
        valid = routable_channels(model, family, d, feat_col)
        j_star = min(valid, key=lambda j: (float(low_sc[j]), j))
        sd_r, ch_scale = rewrite_single_channel(model, base_state, family, j_star, feat_col)
        model.load_state_dict(sd_r, strict=True)
        _, r_hs32, _, _, _ = extract_hidden_and_pred(model, xs_t)
        _, r_hv32, r_pv32, _, _ = extract_hidden_and_pred(model, xv_t)
        bname = f"low{width}"
        routed_states[bname] = sd_r
        routed_info[bname] = {"j": int(j_star), "width": int(width), "feature_col": int(feat_col), "scale": float(ch_scale)}
        body_results.append(
            multi_scale_calibration(
                bname,
                r_hs32.astype(np.float64),
                r_hv32.astype(np.float64),
                us64,
                uv64,
                ys64,
                yv64,
                teacher_s,
                teacher_v,
                r_pv32.astype(np.float64),
                resp_aug,
                orig_aug,
                routed_channel=j_star,
                channel_scale=ch_scale,
                routed_width=width,
            )
        )

    # Exact R64 body selection across ("orig", "low1", "low5") with 1e-14 tie tolerance
    best_val = min(r["val_mse"] for r in body_results)
    winner = next(r for r in body_results if r["val_mse"] <= best_val + 1e-14)
    win_body = winner["body"]

    # Fold calibrated linear head algebraically into the winning body's output layer
    deployed_sd = {k: v.detach().cpu().clone() for k, v in routed_states[win_body].items()}
    deployed_sd[f"{head_name}.weight"] = torch.from_numpy(winner["W"].copy()).to(deployed_sd[f"{head_name}.weight"].dtype)
    deployed_sd[f"{head_name}.bias"] = torch.from_numpy(winner["b"].copy()).to(deployed_sd[f"{head_name}.bias"].dtype)

    model.load_state_dict(deployed_sd, strict=True)
    topo_after = topology_signature(model)
    topo_invariant = (
        topo_before["sha"] == topo_after["sha"]
        and topo_before["numel"] == topo_after["numel"]
        and topo_before["denseMultiplyAccumulates"] == topo_after["denseMultiplyAccumulates"]
    )
    if not topo_invariant:
        raise RuntimeError("SPA topology invariance check failed!")

    return {
        "selected_body": win_body,
        "val_mse": winner["val_mse"],
        "W": winner["W"],
        "b": winner["b"],
        "winning_families": winner["families"],
        "winning_actions": winner["actions"],
        "n_candidates": winner["n_candidates"],
        "r62_action": r62_act,
        "r62_val_mse": r62_mse,
        "routed_info": routed_info,
        "body_val_mses": {r["body"]: r["val_mse"] for r in body_results},
        "deployed_state_dict": deployed_sd,
        "topology_before": topo_before,
        "topology_after": topo_after,
        "topology_invariant": topo_invariant,
    }
