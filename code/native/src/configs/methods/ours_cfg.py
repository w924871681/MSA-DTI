"""Pooled source initialization used for candidate preselection and search baselines."""
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

@dataclass
class OursCfg:
    prior_type: str = 'source_pooled_c1'
    source_prior_seed: int = 2026
    source_prior_force_retrain: bool = False
    source_prior_resume: bool = True
    source_pretrain_K: int = 10
    source_pretrain_epochs: int = 10
    source_pretrain_lr: float = 0.001
    source_pretrain_batch_size: int = 64
    source_pretrain_loss: str = 'mse'
    source_refine_updates: int = 180
    source_refine_lr: float = 0.001
    source_refine_batch_schedule: Tuple[int, int] = (10, 20)
    source_refine_loss: str = 'huber'
    source_weight_decay: float = 0.0
    huber_delta: float = 0.2
    max_grad_norm: float = 5.0

CFG = OursCfg()

def config_dict() -> Dict[str, object]:
    return asdict(CFG)
