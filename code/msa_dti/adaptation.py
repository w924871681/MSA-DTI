"""Structure-Preserving Adaptation (Section 4.4, Eqs. 15-17).

Implements:
1. Leave-one-channel-out ridge importance scoring I(j) on unlabeled validation features (Eq. 15).
2. In-place single-channel causal load routing inside existing MLP, TCN, and GRU weights
   with cryptographic topology verification (+0 parameters, +0 operations).
3. Unlabeled validation response reconstruction p'(x).
4. Multi-scale load-decomposed output calibration (Eqs. 16-17) folded algebraically into
   the existing linear output layer (W_out, b_out).
"""
from __future__ import annotations

import copy
import hashlib
import json
import numpy as np


def channel_importance(
    hidden_val: np.ndarray,
    output_weights: np.ndarray,
    ridge_frac: float = 0.01,
) -> tuple[np.ndarray, int]:
    """Score each hidden channel j by output weight squared norm times its leave-one-out
    ridge reconstruction residual variance on unlabeled validation features (Eq. 15).

    Exact ties prefer the smaller channel index.
    """
    x = np.asarray(hidden_val, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / len(x)
    d = cov.shape[0]
    w_sq = np.square(np.asarray(output_weights, dtype=np.float64)).sum(axis=0)
    if len(w_sq) != d:
        raise ValueError(f"Weight width {len(w_sq)} != hidden dimension {d}")

    if d == 1:
        imp = w_sq * np.square(x).mean(axis=0)
        return imp, 0

    residual_map = np.eye(d, dtype=np.float64)
    for j in range(d):
        keep = np.arange(d) != j
        sub = cov[np.ix_(keep, keep)].copy()
        lam = ridge_frac * max(float(np.trace(sub)) / (d - 1), 1e-12)
        sub.flat[::d] += lam
        beta = np.linalg.solve(sub, cov[keep, j])
        residual_map[keep, j] = -beta

    residual = x @ residual_map
    imp = w_sq * np.square(residual).mean(axis=0)
    j_star = int(np.argmin(imp))
    return imp, j_star


def topology_signature(model, sequence_length: int = 96) -> dict:
    """Compute exact parameter count, dense MACs, layer shapes, and SHA-256 topology digest."""
    import torch

    ops = []
    macs = 0
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            cost = layer.in_features * layer.out_features
            ops.append([name, "Linear", layer.in_features, layer.out_features, cost])
            macs += cost
        elif isinstance(layer, torch.nn.Conv1d):
            cost = (
                sequence_length * layer.out_channels
                * (layer.in_channels // layer.groups) * layer.kernel_size[0]
            )
            ops.append([
                name, "Conv1d", layer.in_channels, layer.out_channels,
                layer.kernel_size[0], layer.groups, layer.stride[0], layer.dilation[0], cost,
            ])
            macs += cost
        elif isinstance(layer, torch.nn.GRU):
            if layer.bidirectional:
                raise ValueError("Expected unidirectional causal GRU")
            cost = sum(
                sequence_length * 3 * layer.hidden_size
                * ((layer.input_size if l_idx == 0 else layer.hidden_size) + layer.hidden_size)
                for l_idx in range(layer.num_layers)
            )
            ops.append([name, "GRU", layer.input_size, layer.hidden_size, layer.num_layers, cost])
            macs += cost
        elif not list(layer.children()):
            ops.append([name, type(layer).__name__])

    shapes = [[n, list(t.shape), str(t.dtype)] for n, t in model.named_parameters()]
    desc = {
        "operations": ops,
        "parameters": shapes,
        "parameter_count": int(sum(t.numel() for t in model.parameters())),
        "dense_macs": int(macs),
        "sequence_length": int(sequence_length),
    }
    desc["sha256"] = hashlib.sha256(
        json.dumps(desc, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return desc


def routable_channels(model, family: str, hidden_dim: int, feature_col: int) -> list[int]:
    """Return existing hidden channels capable of in-place routing for the given architecture family."""
    import torch

    if family == "MLP":
        hidden_sizes = [m.out_features for m in model.net if isinstance(m, torch.nn.Linear)][:-1]
        return list(range(min(*hidden_sizes, hidden_dim))) if len(hidden_sizes) > 1 else list(range(min(hidden_sizes[0], hidden_dim)))
    if family == "GRU":
        return list(range(min(hidden_dim, model.rnn.hidden_size)))
    if family == "TCN":
        count = min([hidden_dim] + [blk.out_ch for blk in model.blocks])
        if isinstance(model.blocks[0].res, torch.nn.Identity):
            if feature_col >= count:
                raise ValueError("Identity residual block cannot route feature column out of range")
            return [feature_col]
        return list(range(count))
    raise ValueError(f"Unsupported architecture family: {family}")


def rewrite_single_channel(
    model,
    family: str,
    channel_idx: int,
    feature_col: int,
    sequence_length: int = 96,
    input_channels: int = 25,
) -> tuple[dict, dict]:
    """Rewrite a single existing hidden channel in place to carry a causal load feature
    strictly within Theta(a_q), verifying identical topology SHA-256 (+0 params, +0 ops).
    """
    import torch

    before_sig = topology_signature(model, sequence_length=sequence_length)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    params = dict(model.named_parameters())

    def set_row(key: str, row_ix: int, entries=()):
        target = params[key][row_ix]
        target.zero_()
        for pos, val in entries:
            if pos is None:
                target.fill_(val)
            else:
                target[pos] = val

    with torch.no_grad():
        if family == "MLP":
            hidden_layers = [(f"net.{n}", m) for n, m in enumerate(model.net) if isinstance(m, torch.nn.Linear)][:-1]
            last_step_offset = (sequence_length - 1) * input_channels + feature_col
            for layer_i, (path, _) in enumerate(hidden_layers):
                src_pos = last_step_offset if layer_i == 0 else channel_idx
                set_row(f"{path}.weight", channel_idx, [(src_pos, 1.0)])
                set_row(f"{path}.bias", channel_idx)
        elif family == "TCN":
            for blk_i, block in enumerate(model.blocks):
                prefix = f"blocks.{blk_i}."
                set_row(prefix + "conv2.weight", channel_idx)
                set_row(prefix + "conv2.bias", channel_idx)
                src = feature_col if blk_i == 0 else channel_idx
                if isinstance(block.res, torch.nn.Conv1d):
                    set_row(prefix + "res.weight", channel_idx, [((src, 0), 1.0)])
                    set_row(prefix + "res.bias", channel_idx)
                else:
                    if not (isinstance(block.res, torch.nn.Identity) and src == channel_idx):
                        raise ValueError("Incompatible identity skip for TCN routing")
        elif family == "GRU":
            gru = model.rnn
            d = gru.hidden_size
            for l_idx in range(gru.num_layers):
                for gate in range(3):
                    row_ix = gate * d + channel_idx
                    # Near-instantaneous update gate (bias = -20) passes linear scaled feature through gate 2
                    in_entries = [((feature_col if l_idx == 0 else channel_idx), (0.05 if l_idx == 0 else 1.0))] if gate == 2 else []
                    bias_entries = [(None, -20.0)] if gate == 1 else []
                    set_row(f"rnn.weight_ih_l{l_idx}", row_ix, in_entries)
                    set_row(f"rnn.weight_hh_l{l_idx}", row_ix)
                    set_row(f"rnn.bias_ih_l{l_idx}", row_ix, bias_entries)
                    set_row(f"rnn.bias_hh_l{l_idx}", row_ix)
        else:
            raise ValueError(f"Unsupported architecture family: {family}")

    after_sig = topology_signature(model, sequence_length=sequence_length)
    if before_sig["sha256"] != after_sig["sha256"]:
        model.load_state_dict(original, strict=True)
        raise RuntimeError("Topology invariant violated during single-channel rewrite")

    rewritten_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return rewritten_state, after_sig


def _affine_ridge(features: np.ndarray, targets: np.ndarray, ridge: float = 0.01) -> np.ndarray:
    """Fit an affine linear map [W, b] predicting targets from features via ridge regression."""
    mu = features.mean(axis=0)
    scale = max(float(np.square(features - mu).mean()) ** 0.5, 1e-12)
    z = (features - mu) / scale
    gram = (z.T @ z) / len(z) + ridge * np.eye(z.shape[1])
    y_mu = targets.mean(axis=0)
    coef = np.linalg.solve(gram, (z.T @ (targets - y_mu)) / len(z)) / scale
    bias = y_mu - mu @ coef
    return np.column_stack((coef.T, bias))


def _eval_affine(features: np.ndarray, head: np.ndarray) -> np.ndarray:
    return features @ head[:, :-1].T + head[:, -1]


def reconstruct_response(
    hidden_val_rewritten: np.ndarray,
    initial_response_val: np.ndarray,
    initial_head: np.ndarray,
    rewritten_channel: int | None,
    ridge: float = 0.01,
) -> np.ndarray:
    """Reconstruct initial model response p'(x) on rewritten hidden representations
    strictly from unlabeled validation inputs (zero future labels)."""
    base = np.asarray(initial_head, dtype=np.float64).copy()
    if rewritten_channel is not None:
        base[:, rewritten_channel] = 0.0
    residual = initial_response_val - _eval_affine(hidden_val_rewritten, base)
    return base + _affine_ridge(hidden_val_rewritten, residual, ridge=ridge)


def multi_scale_calibration(
    hidden_sup: np.ndarray,
    hidden_val: np.ndarray,
    load_scales_sup: np.ndarray,
    load_scales_val: np.ndarray,
    y_sup: np.ndarray,
    y_val: np.ndarray,
    initial_response_val: np.ndarray,
    initial_head: np.ndarray,
    rewritten_channel: int | None = None,
    routed_scale_idx: int | None = None,
    channel_scale: float = 1.0,
    blend_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25),
    mix_grid: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    gamma_grid: tuple[float, ...] = (0.5, 1.0),
) -> tuple[np.ndarray, float]:
    """Generate and select per-horizon output rows folded algebraically into (W_out, b_out)
    via multi-scale causal load decomposition (Eqs. 16-17).

    Returns:
        (selected_head [H, d+1], mean_validation_mse)
    """
    h_dim = hidden_val.shape[1]
    horizon = initial_head.shape[0]
    n_scales = load_scales_val.shape[1]

    # Express causal multi-scale load averages as affine functions of hidden representation
    levels = _affine_ridge(hidden_val, load_scales_val, ridge=0.01)
    if rewritten_channel is not None and routed_scale_idx is not None:
        levels[routed_scale_idx, :] = 0.0
        levels[routed_scale_idx, rewritten_channel] = 1.0 / channel_scale

    base = (
        initial_head.copy()
        if rewritten_channel is None
        else reconstruct_response(hidden_val, initial_response_val, initial_head, rewritten_channel)
    )
    p_val = _eval_affine(hidden_val, base)
    q_val = _eval_affine(hidden_val, levels)
    q_sup = _eval_affine(hidden_sup, levels)

    candidates = [base]
    for slow in range(1, n_scales):
        for fast in range(slow):
            # Project reconstructed response onto [q_s(x), q_r(x) - q_s(x)] on unlabeled validation inputs
            x_sub = np.column_stack((q_val[:, slow], q_val[:, fast] - q_val[:, slow]))
            x_mu = x_sub.mean(axis=0)
            x_sd = np.maximum(x_sub.std(axis=0), 1e-6)
            z_sub = (x_sub - x_mu) / x_sd
            levels_pair = np.stack([levels[slow], levels[fast] - levels[slow]])
            beta = np.linalg.solve(
                (z_sub.T @ z_sub) / len(z_sub) + 1e-6 * np.eye(2),
                (z_sub.T @ (p_val - p_val.mean(axis=0))) / len(z_sub),
            ) / x_sd[:, None]
            c_rs = beta.T @ levels_pair
            c_rs[:, -1] += p_val.mean(axis=0) - x_mu @ beta

            for omega in blend_grid:
                g_row = levels[slow] + omega * (levels[fast] - levels[slow])
                g_sup = q_sup[:, slow] + omega * (q_sup[:, fast] - q_sup[:, slow])
                support_shift = y_sup.mean(axis=0) - g_sup.mean()
                g_mat = np.repeat(g_row[None, :], horizon, axis=0)
                for use_shift in (False, True):
                    g_use = g_mat.copy()
                    if use_shift:
                        g_use[:, -1] += support_shift
                    for t_val in mix_grid:
                        candidates.append((1.0 - t_val) * base + t_val * g_use)
                        mixed = (1.0 - t_val) * c_rs + t_val * g_use
                        for gamma in gamma_grid:
                            candidates.append(mixed + gamma * (base - c_rs))

    cand_arr = np.stack(candidates, axis=0)  # [C, H, d+1]
    preds = np.einsum("nd,chd->cnh", hidden_val, cand_arr[:, :, :-1]) + cand_arr[:, None, :, -1]
    mses = np.square(preds - y_val[None, :, :]).mean(axis=1)  # [C, H]

    best_idx = np.argmin(mses, axis=0)
    selected_head = np.stack([cand_arr[best_idx[h], h] for h in range(horizon)], axis=0)
    val_mse = float(np.mean([mses[best_idx[h], h] for h in range(horizon)]))
    return selected_head, val_mse
