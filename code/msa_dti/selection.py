"""Response-Guided Model Selection (RGMS) for MSA-DTI.

Exact standalone implementation of the frozen two-stage RGMS algorithm
(Stage A: dynamic-anchor paired kNN calibration over 35-D anchor-relative
response features; Stage B: single-candidate reachable set enumeration,
three-group standardized pairwise direct-margin estimation, complete-graph
consensus ranking, and direct-margin guarded switching).
"""
from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np

from .config import CANDIDATE_BANK


def center_scale(
    lin_losses: np.ndarray,
    feasible_order: Sequence[int],
    floor: float = 1e-12,
) -> float:
    """Eq. (10): Robust center scale s_c = max(median_{q in Q^fea} J_c^lin(q), floor)."""
    feas = np.asarray(feasible_order, dtype=int)
    vals = np.asarray(lin_losses, dtype=np.float64)[feas]
    return max(float(np.median(vals)), float(floor))


def history_reference_scale(
    hist_true_losses: np.ndarray,
    hist_ref_scores: np.ndarray,
    feasible_order: Sequence[int],
    floor: float = 1e-8,
) -> float:
    """Budget-specific historical reference scale C_b over V_hist."""
    q = np.asarray(hist_true_losses, dtype=np.float64)
    ref = np.asarray(hist_ref_scores, dtype=np.float64)
    order = np.asarray(feasible_order, dtype=int)
    picks = order[np.argmin(ref[:, order], axis=1)]
    chosen_losses = q[np.arange(len(q)), picks]
    return max(float(floor), float(chosen_losses.mean()))


def anchor_pair_features(
    z: np.ndarray,
    lin_losses: np.ndarray,
    anchor_idx: int,
    c_b: float,
) -> np.ndarray:
    """Construct the 35-D anchor-relative paired representation U_{v,b}(q; a).

    U_{v,b}(q; a) = [(z_q + z_a)/2, z_q - z_a, (J^lin(q) - J^lin(a)) / C_b].
    """
    z_a = z[..., anchor_idx : anchor_idx + 1, :]
    b_a = lin_losses[..., anchor_idx : anchor_idx + 1]
    return np.concatenate(
        (
            (z + z_a) / 2.0,
            z - z_a,
            ((lin_losses - b_a) / c_b)[..., None],
        ),
        axis=-1,
    )


def calibrated_scores(
    target_feat17: np.ndarray,
    target_lin_losses: np.ndarray,
    hist_feat17: np.ndarray,
    hist_lin_losses: np.ndarray,
    hist_true_losses: np.ndarray,
    hist_ref_scores: np.ndarray,
    feasible_order: Sequence[int],
    k: int,
    alpha: float,
) -> tuple[np.ndarray, int, np.ndarray, float]:
    """Stage A: Historical-neighbor bias correction with dynamic target anchor.

    Parameters
    ----------
    target_feat17 : np.ndarray, shape (M, 17)
        17-D response probe features for all M bank candidates on target center.
    target_lin_losses : np.ndarray, shape (M,)
        Linearized post-adaptation response estimates J_c^lin(q).
    hist_feat17 : np.ndarray, shape (N_hist, M, 17)
        17-D response probe features across historical centers V_hist.
    hist_lin_losses : np.ndarray, shape (N_hist, M)
        Linearized response estimates across V_hist.
    hist_true_losses : np.ndarray, shape (N_hist, M)
        Archived post-adaptation assessment losses J_v^true(q) across V_hist.
    hist_ref_scores : np.ndarray, shape (N_hist, M)
        Historical baseline reference scores used to compute budget scale C_b.
    feasible_order : Sequence[int]
        Budget-feasible bank indices ordered deterministically by
        (parameters, operations, architecture_id).
    k : int
        Number of nearest historical neighbors.
    alpha : float
        Residual bias-correction damping coefficient.

    Returns
    -------
    calibrated : np.ndarray, shape (M,)
        Calibrated Stage A candidate loss estimates J_hat_c^pair(q).
    anchor_idx : int
        Dynamic target anchor index a = argmin_{q in Q^fea} J_c^lin(q).
    near_idx : np.ndarray, shape (k_eff, M)
        Indices of the k nearest historical centers per candidate.
    c_b : float
        Budget-specific historical reference scale C_b.
    """
    x_t = np.asarray(target_feat17, dtype=np.float64)
    b_t = np.asarray(target_lin_losses, dtype=np.float64)
    x_h = np.asarray(hist_feat17, dtype=np.float64)
    b_h = np.asarray(hist_lin_losses, dtype=np.float64)
    q_h = np.asarray(hist_true_losses, dtype=np.float64)
    order = np.asarray(feasible_order, dtype=int)

    n_hist = len(x_h)
    k_eff = min(int(k), n_hist)

    mu = x_h.mean(axis=(0, 1))
    sd = np.maximum(x_h.std(axis=(0, 1)), 1e-8)
    z_h = np.clip((x_h - mu) / sd, -5.0, 5.0)
    z_t = np.clip((x_t - mu) / sd, -5.0, 5.0)

    c_b = history_reference_scale(q_h, hist_ref_scores, order)
    anchor_idx = int(order[int(np.argmin(b_t[order]))])

    u_t = anchor_pair_features(z_t, b_t, anchor_idx, c_b)
    u_h = anchor_pair_features(z_h, b_h, anchor_idx, c_b)

    d0 = np.mean((u_t[None, :, :17] - u_h[:, :, :17]) ** 2, axis=-1)
    d1 = np.mean((u_t[None, :, 17:34] - u_h[:, :, 17:34]) ** 2, axis=-1)
    d2 = (u_t[None, :, 34] - u_h[:, :, 34]) ** 2
    dist = (17.0 * d0 + 17.0 * d1 + d2) / 35.0

    near_idx = np.argsort(dist, axis=0, kind="stable")[:k_eff]
    resid = q_h - b_h
    rel_resid = (resid - resid[:, anchor_idx : anchor_idx + 1]) / c_b
    corr = c_b * np.take_along_axis(rel_resid, near_idx, axis=0).mean(axis=0)
    calibrated = b_t + float(alpha) * corr
    return calibrated, anchor_idx, near_idx, c_b


def reachable_set(
    target_lin_losses: np.ndarray,
    base_scores: np.ndarray,
    hist_lin_losses: np.ndarray,
    hist_true_losses: np.ndarray,
    feasible_order: Sequence[int],
    anchor_idx: int,
    near_idx: np.ndarray,
    k: int,
    alpha: float,
) -> list[int]:
    """Enumerate reachable contenders R_c under single-candidate k-subset replacement.

    Returns candidate bank indices in deterministic order starting with the
    Stage A winner q_init followed by remaining reachable contenders in
    feasible_order.
    """
    b_t = np.asarray(target_lin_losses, dtype=np.float64)
    base = np.asarray(base_scores, dtype=np.float64)
    b_h = np.asarray(hist_lin_losses, dtype=np.float64)
    q_h = np.asarray(hist_true_losses, dtype=np.float64)
    order = np.asarray(feasible_order, dtype=int)
    n_hist = len(q_h)
    k_eff = min(int(k), n_hist)

    orig = int(order[int(np.argmin(base[order]))])
    possible = {orig}
    combos = np.array(list(itertools.combinations(range(n_hist), k_eff)), dtype=int)
    bias = q_h - b_h
    order_list = [int(i) for i in order]

    for pos, m in enumerate(order_list):
        if m == anchor_idx:
            continue
        available = order[order != m]
        rival = int(available[int(np.argmin(base[available]))])
        rival_pos = order_list.index(rival)
        vals = b_t[m] + float(alpha) * (bias[:, m] - bias[:, anchor_idx])[combos].mean(axis=1)
        match = np.all(combos == np.sort(near_idx[:, m]), axis=1)
        vals[match] = base[m]
        wins = (vals < base[rival]) | ((vals == base[rival]) & (pos < rival_pos))
        possible.update(int(x) for x in np.unique(np.where(wins, m, rival)))

    return [orig] + [m for m in order_list if m in possible and m != orig]


def build_node_features(
    lin_losses: np.ndarray,
    cal_scores: np.ndarray,
    block_features: np.ndarray,
    feasible_order: Sequence[int],
) -> tuple[np.ndarray, float]:
    """Build 9-D scaled node representation [-cal_score, -after_4blk, gain_4blk] / s_c."""
    s_c = center_scale(lin_losses, feasible_order)
    node = np.column_stack((-np.asarray(cal_scores, dtype=np.float64), np.asarray(block_features, dtype=np.float64))) / s_c
    return node, s_c


def pairwise_direct_margins(
    target_node: np.ndarray,
    hist_nodes: np.ndarray,
    hist_true_losses: np.ndarray,
    hist_scales: np.ndarray,
    sorted_feasible_idx: Sequence[int],
    k: int,
) -> np.ndarray:
    """Stage B: Direct pairwise margin matrix over feasible candidates.

    Uses standardized three-group L1 distance:
      group 1: calibrated-score difference (col 0)
      group 2: 4-block post-adaptation negative-loss difference (cols 1..4)
      group 3: 4-block adaptation-gain difference (cols 5..8)
    and k-nearest historical residual correction.

    Returns
    -------
    margin_matrix : np.ndarray, shape (L, L)
        Anti-symmetric matrix where entry [a, b] > 0 favors candidate a over b,
        indexed in the order of sorted_feasible_idx.
    """
    ids = np.asarray(sorted_feasible_idx, dtype=int)
    n_feas = len(ids)
    iu, iv = np.triu_indices(n_feas, 1)
    left, right = ids[iu], ids[iv]

    x_hist = np.asarray(hist_nodes, dtype=np.float64)
    q_hist = np.asarray(hist_true_losses, dtype=np.float64)
    s_hist = np.asarray(hist_scales, dtype=np.float64)
    x_t = np.asarray(target_node, dtype=np.float64)

    train = x_hist[:, left, :] - x_hist[:, right, :]  # (N_hist, P, 9)
    labels = (q_hist[:, right] - q_hist[:, left]) / s_hist[:, None]  # (N_hist, P)
    target_pair = x_t[left, :] - x_t[right, :]  # (P, 9)

    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd > 1e-10, sd, 1.0)
    z = (train - mu) / sd
    t = (target_pair - mu) / sd
    resid = labels - train[:, :, 0]

    delta = np.abs(z - t)
    dist = delta[:, :, 0] + delta[:, :, 1:5].mean(axis=2) + delta[:, :, 5:].mean(axis=2)

    k_eff = min(int(k), len(train))
    ix = np.argsort(dist, axis=0, kind="stable")[:k_eff]
    est = target_pair[:, 0] + np.take_along_axis(resid, ix, axis=0).mean(axis=0)

    matrix = np.zeros((n_feas, n_feas), dtype=np.float64)
    matrix[iu, iv] = est
    matrix[iv, iu] = -est
    return matrix


def consensus_scores(
    margin_matrix: np.ndarray,
    candidate_positions: Sequence[int],
) -> np.ndarray:
    """Complete-graph consensus score over the reachable subset R_c."""
    pos = np.asarray(candidate_positions, dtype=int)
    return np.asarray(margin_matrix, dtype=np.float64)[np.ix_(pos, pos)].mean(axis=1)


def guarded_replacement(
    initial_cand: int,
    consensus_cand: int,
    direct_margin: float,
) -> int:
    """Eq. (14): Replace q_init with q_top iff q_top != q_init and m_hat(q_top, q_init) > 0."""
    if consensus_cand != initial_cand and float(direct_margin) > 0.0:
        return int(consensus_cand)
    return int(initial_cand)


def select_candidate(
    target_feat17: np.ndarray,
    target_lin_losses: np.ndarray,
    target_block_features: np.ndarray,
    hist_feat17: np.ndarray,
    hist_lin_losses: np.ndarray,
    hist_true_losses: np.ndarray,
    hist_ref_scores: np.ndarray,
    hist_block_features: np.ndarray,
    feasible_order: Sequence[int],
    k: int,
    alpha: float,
    hist_loo_k: Sequence[int] | None = None,
    hist_loo_alpha: Sequence[float] | None = None,
    bank: Sequence[int] = CANDIDATE_BANK,
) -> dict:
    """Execute the complete frozen RGMS selection pipeline for one target deployment.

    Parameters
    ----------
    target_feat17 : np.ndarray, shape (M, 17)
        Target 17-D response features.
    target_lin_losses : np.ndarray, shape (M,)
        Target linearized response loss estimates J_c^lin.
    target_block_features : np.ndarray, shape (M, 8)
        Target 4-block prefix features [-after_4blk, before_4blk - after_4blk].
    hist_feat17 : np.ndarray, shape (N_hist, M, 17)
        Historical 17-D response features across V_hist.
    hist_lin_losses : np.ndarray, shape (N_hist, M)
        Historical linearized response loss estimates.
    hist_true_losses : np.ndarray, shape (N_hist, M)
        Historical post-adaptation assessment losses J_v^true.
    hist_ref_scores : np.ndarray, shape (N_hist, M)
        Historical reference scores (for budget scale C_b).
    hist_block_features : np.ndarray, shape (N_hist, M, 8)
        Historical 4-block prefix features across V_hist.
    feasible_order : Sequence[int]
        Budget-feasible bank indices ordered by (params, ops, arch_id).
    k : int
        Final neighborhood size for target center.
    alpha : float
        Final residual damping coefficient for target center.
    hist_loo_k : Sequence[int], optional
        Per-historical-center leave-one-out k values (defaults to k for all).
    hist_loo_alpha : Sequence[float], optional
        Per-historical-center leave-one-out alpha values (defaults to alpha for all).
    bank : Sequence[int]
        Candidate architecture IDs corresponding to the M bank indices.

    Returns
    -------
    dict
        Selection audit dictionary containing bank indices and architecture IDs
        for anchor, initial (PairKNN), reachable set R_c, consensus top,
        direct margin, final selected candidate, calibrated scores, and s_c.
    """
    order = [int(i) for i in feasible_order]
    n_hist = len(hist_feat17)
    loo_k = [int(k)] * n_hist if hist_loo_k is None else [int(v) for v in hist_loo_k]
    loo_a = [float(alpha)] * n_hist if hist_loo_alpha is None else [float(v) for v in hist_loo_alpha]

    # Stage A on target
    cal_t, anchor_idx, near_idx, c_b = calibrated_scores(
        target_feat17,
        target_lin_losses,
        hist_feat17,
        hist_lin_losses,
        hist_true_losses,
        hist_ref_scores,
        order,
        k=k,
        alpha=alpha,
    )
    orig_idx = int(order[int(np.argmin(cal_t[order]))])

    # Reachable set R_c under single-candidate historical subset replacement
    r_c = reachable_set(
        target_lin_losses,
        cal_t,
        hist_lin_losses,
        hist_true_losses,
        order,
        anchor_idx,
        near_idx,
        k=k,
        alpha=alpha,
    )

    # Leave-one-out Stage A scores for historical centers
    hist_nodes = []
    hist_scales = []
    for h in range(n_hist):
        rest = [i for i in range(n_hist) if i != h]
        cal_h, _, _, _ = calibrated_scores(
            hist_feat17[h],
            hist_lin_losses[h],
            hist_feat17[rest],
            hist_lin_losses[rest],
            hist_true_losses[rest],
            hist_ref_scores[rest],
            order,
            k=loo_k[h],
            alpha=loo_a[h],
        )
        node_h, s_h = build_node_features(hist_lin_losses[h], cal_h, hist_block_features[h], order)
        hist_nodes.append(node_h)
        hist_scales.append(s_h)

    node_t, s_t = build_node_features(target_lin_losses, cal_t, target_block_features, order)
    sorted_feas = sorted(order, key=lambda idx: int(bank[idx]))

    margin_matrix = pairwise_direct_margins(
        node_t,
        np.stack(hist_nodes),
        hist_true_losses,
        np.asarray(hist_scales, dtype=np.float64),
        sorted_feas,
        k=k,
    )

    r_pos = [sorted_feas.index(i) for i in r_c]
    c_scores = consensus_scores(margin_matrix, r_pos)
    top_idx = r_c[int(np.flatnonzero(c_scores >= c_scores.max() - 1e-12)[0])]

    if top_idx != orig_idx:
        direct_margin = float(margin_matrix[sorted_feas.index(top_idx), sorted_feas.index(orig_idx)])
    else:
        direct_margin = 0.0

    final_idx = guarded_replacement(orig_idx, top_idx, direct_margin)
    return {
        "anchor_idx": anchor_idx,
        "anchor_arch": int(bank[anchor_idx]),
        "initial_idx": orig_idx,
        "initial_arch": int(bank[orig_idx]),
        "reachable_idx": r_c,
        "reachable_archs": [int(bank[i]) for i in r_c],
        "consensus_idx": top_idx,
        "consensus_arch": int(bank[top_idx]),
        "direct_margin": direct_margin,
        "selected_idx": final_idx,
        "selected_arch": int(bank[final_idx]),
        "calibrated_scores": cal_t,
        "center_scale": s_t,
        "history_scale": c_b,
        "consensus_scores": c_scores,
        "margin_matrix": margin_matrix,
    }
