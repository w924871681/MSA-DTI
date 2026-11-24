"""MSA-DTI: Response-Guided Model Selection and Structure-Preserving Adaptation
for Few-Shot Digital Twin Instantiation.
"""
from .config import (
    CANDIDATE_BANK,
    ARCHITECTURE_FAMILY,
    DEPLOYMENT_BUDGETS,
    HORIZON_SUPPORT_GRID,
    FROZEN_HYPERPARAMETERS,
    feasible_candidates,
)
from .selection import (
    center_scale,
    calibrated_scores,
    pairwise_direct_margins,
    consensus_scores,
    guarded_replacement,
    select_candidate,
)
from .adaptation import (
    channel_importance,
    routable_channels,
    topology_signature,
    rewrite_single_channel,
    reconstruct_response,
    multi_scale_calibration,
)

__all__ = [
    "CANDIDATE_BANK",
    "ARCHITECTURE_FAMILY",
    "DEPLOYMENT_BUDGETS",
    "HORIZON_SUPPORT_GRID",
    "FROZEN_HYPERPARAMETERS",
    "feasible_candidates",
    "center_scale",
    "calibrated_scores",
    "pairwise_direct_margins",
    "consensus_scores",
    "guarded_replacement",
    "select_candidate",
    "channel_importance",
    "routable_channels",
    "topology_signature",
    "rewrite_single_channel",
    "reconstruct_response",
    "multi_scale_calibration",
]
