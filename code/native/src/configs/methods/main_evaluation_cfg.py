"""Frozen target adaptation and search settings for the comparison methods."""
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

@dataclass(frozen=True)
class MainEvaluationConfig:
    fixed_target_steps: int = 50
    fixed_target_lr: float = 0.01
    fixed_target_grad_clip: float = 1.0
    candidate_budget: int = 12
    meta_prior_weight: float = 0.6
    meta_proxy_weight: float = 0.4
    proxy_batch_size: int = 16
    proxy_support_points: int = 6
    search_target_steps: int = 50
    search_target_lr: float = 0.0003
    search_target_huber_delta: float = 0.2
    search_target_grad_clip: float = 5.0

CFG = MainEvaluationConfig()

def config_dict() -> Dict[str, object]:
    return asdict(CFG)
