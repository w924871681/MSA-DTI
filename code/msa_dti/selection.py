"""Response-Guided Model Selection (Section 4.3, Eqs. 9-14).

Implements:
1. Scale normalization across heterogeneous centers (Eq. 10).
2. Historical bias correction and initial selection q_init (Eq. 11).
3. Direct pairwise margin estimation via standardized group-averaged L1 neighbor matching (Eq. 12).
4. Complete-graph least-squares consensus scoring (Eq. 13).
5. Guarded replacement rule requiring positive direct margin m_c(q_top, q_init) > 0 (Eq. 14).
"""
from __future__ import annotations

import numpy as np


def center_scale(linearized_losses: np.ndarray, floor: float = 1e-12) -> float:
    """Compute center normalization scale s_c (Eq. 10)."""
    return float(max(floor, np.median(np.asarray(linearized_losses, dtype=np.float64))))


def calibrated_scores(
    target_features: np.ndarray,
    target_linearized: np.ndarray,
    hist_features: np.ndarray,
    hist_linearized: np.ndarray,
    hist_real_losses: np.ndarray,
    feasible_indices: np.ndarray,
    k: int,
    shrinkage: float,
    scale_floor: float = 1e-12,
) -> tuple[np.ndarray, int]:
    """Estimate bias-corrected scores J_tilde_c(q) and initial choice q_init (Eq. 11).

    Args:
        target_features: [N, F] normalized response profile on target validation set.
        target_linearized: [N] linearized validation losses J_c^lin(q).
        hist_features: [V, N, F] historical center response profiles.
        hist_linearized: [V, N] historical linearized validation losses.
        hist_real_losses: [V, N] recorded historical post-adaptation assessment losses Y_v^real(q).
        feasible_indices: 1D integer array of feasible candidate indices (tie-break ordered).
        k: Number of nearest historical centers.
        shrinkage: Bias shrinkage weight in [0, 1].

    Returns:
        (calibrated_score_vector_over_feasible, initial_candidate_index_in_bank)
    """
    feas = np.asarray(feasible_indices, dtype=int)
    mu = hist_features.mean(axis=(0, 1))
    sd = np.maximum(hist_features.std(axis=(0, 1)), 1e-8)

    z_hist = np.clip((hist_features - mu) / sd, -5.0, 5.0)[:, feas]
    z_tar = np.clip((target_features - mu) / sd, -5.0, 5.0)[feas]
    dist = np.abs(z_hist - z_tar[None, :, :]).mean(axis=(1, 2))

    k_eff = min(max(int(k), 1), len(dist))
    nbrs = np.argsort(dist, kind="mergesort")[:k_eff]

    s_tar = center_scale(target_linearized[feas], floor=scale_floor)
    s_hist = np.maximum(np.median(hist_linearized[:, feas], axis=1), scale_floor)
    hist_bias = (hist_real_losses[:, feas] - hist_linearized[:, feas]) / s_hist[:, None]

    mean_bias = hist_bias[nbrs].mean(axis=0)
    norm_lin = target_linearized[feas] / s_tar
    j_tilde = s_tar * (norm_lin + shrinkage * mean_bias)
    q_init = int(feas[int(np.argmin(j_tilde))])
    return j_tilde, q_init


def pairwise_direct_margins(
    target_cal_scores: np.ndarray,
    target_block_src: np.ndarray,
    target_block_adapt: np.ndarray,
    hist_cal_scores: np.ndarray,
    hist_real_losses: np.ndarray,
    hist_block_src: np.ndarray,
    hist_block_adapt: np.ndarray,
    reachable_indices: np.ndarray,
    k: int,
    scale_floor: float = 1e-12,
) -> np.ndarray:
    """Compute antisymmetric direct pairwise margin matrix m_c(i, j) over reachable subset R_c (Eq. 12).

    Each ordered pair (i, j) constructs a three-group validation feature vector:
      1. Initial calibrated score difference (J_tilde(j) - J_tilde(i)) / s_c
      2. Chronological adapted block loss differences (e^(1)_j - e^(1)_i) / s_c
      3. Chronological block adaptation gain differences ((e^(0)_i - e^(1)_i) - (e^(0)_j - e^(1)_j)) / s_c
    """
    reach = np.asarray(reachable_indices, dtype=int)
    r_len = len(reach)
    margins = np.zeros((r_len, r_len), dtype=np.float64)
    if r_len <= 1:
        return margins

    s_tar = center_scale(target_cal_scores[reach], floor=scale_floor)
    s_hist = np.maximum(np.median(hist_cal_scores[:, reach], axis=1), scale_floor)
    k_eff = min(max(int(k), 1), len(s_hist))

    for a_pos in range(r_len):
        for b_pos in range(a_pos + 1, r_len):
            i, j = int(reach[a_pos]), int(reach[b_pos])

            # Group 1: calibrated score difference
            g1_t = (target_cal_scores[j] - target_cal_scores[i]) / s_tar
            g1_h = (hist_cal_scores[:, j] - hist_cal_scores[:, i]) / s_hist

            # Group 2: adapted block loss differences
            g2_t = (target_block_adapt[j] - target_block_adapt[i]) / s_tar
            g2_h = (hist_block_adapt[:, j] - hist_block_adapt[:, i]) / s_hist[:, None]

            # Group 3: block adaptation gain differences
            gain_t_i = target_block_src[i] - target_block_adapt[i]
            gain_t_j = target_block_src[j] - target_block_adapt[j]
            g3_t = (gain_t_i - gain_t_j) / s_tar

            gain_h_i = hist_block_src[:, i] - hist_block_adapt[:, i]
            gain_h_j = hist_block_src[:, j] - hist_block_adapt[:, j]
            g3_h = (gain_h_i - gain_h_j) / s_hist[:, None]

            # Standardize each coordinate by historical std and sum group-averaged L1 distances
            d1 = np.abs(g1_h - g1_t) / max(float(np.std(g1_h)), 1e-8)
            d2 = (np.abs(g2_h - g2_t[None, :]) / np.maximum(np.std(g2_h, axis=0), 1e-8)[None, :]).mean(axis=1)
            d3 = (np.abs(g3_h - g3_t[None, :]) / np.maximum(np.std(g3_h, axis=0), 1e-8)[None, :]).mean(axis=1)

            nbrs = np.argsort(d1 + d2 + d3, kind="mergesort")[:k_eff]
            real_diff_h = (hist_real_losses[nbrs, j] - hist_real_losses[nbrs, i]) / s_hist[nbrs]
            cal_diff_h = (hist_cal_scores[nbrs, j] - hist_cal_scores[nbrs, i]) / s_hist[nbrs]

            m_ij = float(g1_t + np.mean(real_diff_h - cal_diff_h))
            margins[a_pos, b_pos] = m_ij
            margins[b_pos, a_pos] = -m_ij

    return margins


def consensus_scores(margin_matrix: np.ndarray) -> np.ndarray:
    """Least-squares consistent consensus score m_bar_c(i) on complete comparison graph (Eq. 13)."""
    return margin_matrix.mean(axis=1)


def guarded_replacement(
    q_init: int,
    reachable_indices: np.ndarray,
    margin_matrix: np.ndarray,
) -> tuple[int, int, float]:
    """Apply consensus ranking and guarded replacement rule (Eq. 14).

    Tie-breaking retains q_init under ties, otherwise respects the deterministic
    resource ordering (smaller parameters, smaller operations, lower index) in reachable_indices.

    Returns:
        (selected_candidate_id, consensus_top_id, direct_margin_top_vs_init)
    """
    reach = list(map(int, reachable_indices))
    init_pos = reach.index(int(q_init))
    row_avg = consensus_scores(margin_matrix)
    best_val = float(np.max(row_avg))

    # Retain q_init on exact/numerical ties; otherwise first in deterministic resource order
    if row_avg[init_pos] >= best_val - 1e-12:
        top_pos = init_pos
    else:
        top_pos = int(np.flatnonzero(row_avg >= best_val - 1e-12)[0])

    q_top = reach[top_pos]
    direct_margin = float(margin_matrix[top_pos, init_pos])
    selected = q_top if (q_top != q_init and direct_margin > 0.0) else q_init
    return selected, q_top, direct_margin


def select_candidate(
    target_features: np.ndarray,
    target_linearized: np.ndarray,
    target_block_src: np.ndarray,
    target_block_adapt: np.ndarray,
    hist_features: np.ndarray,
    hist_linearized: np.ndarray,
    hist_real_losses: np.ndarray,
    hist_block_src: np.ndarray,
    hist_block_adapt: np.ndarray,
    feasible_indices: np.ndarray,
    k_bias: int,
    shrinkage: float,
    k_pair: int,
    reachable_indices: np.ndarray | None = None,
) -> dict:
    """Run complete Response-Guided Model Selection and return full decision diagnostics."""
    feas = np.asarray(feasible_indices, dtype=int)
    j_tilde_feas, q_init = calibrated_scores(
        target_features, target_linearized, hist_features, hist_linearized,
        hist_real_losses, feas, k=k_bias, shrinkage=shrinkage,
    )
    full_cal = np.full(len(target_linearized), np.inf, dtype=np.float64)
    full_cal[feas] = j_tilde_feas

    reach = feas if reachable_indices is None else np.asarray(reachable_indices, dtype=int)
    hist_cal = hist_linearized.copy()
    margins = pairwise_direct_margins(
        full_cal, target_block_src, target_block_adapt,
        hist_cal, hist_real_losses, hist_block_src, hist_block_adapt,
        reach, k=k_pair,
    )
    selected, q_top, direct_margin = guarded_replacement(q_init, reach, margins)
    return {
        "q_init": q_init,
        "q_top": q_top,
        "selected": selected,
        "direct_margin": direct_margin,
        "switched": selected != q_init,
        "calibrated_scores": j_tilde_feas,
        "margin_matrix": margins,
    }
