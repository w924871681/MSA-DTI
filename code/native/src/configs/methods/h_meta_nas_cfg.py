"""Fixed source meta training and evolutionary target search for H-Meta-NAS."""
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

@dataclass(frozen=True)
class HMetaNasConfig:
    data_seed: int = 2904
    train_seed: int = 2904
    source_centers: int = 20
    H_list: Tuple[int, ...] = (1, 4)
    K_list: Tuple[int, ...] = (10, 20)
    architecture_count: int = 66
    source_outer_iterations: int = 240
    source_architectures_per_task: int = 12
    source_inner_steps: int = 5
    source_inner_lr: float = 0.01
    source_meta_lr: float = 0.001
    population_size: int = 6
    generations: int = 2
    parent_count: int = 3
    target_steps: int = 50
    target_lr: float = 0.01
    target_grad_clip: float = 1.0

CFG = HMetaNasConfig()

def config_dict() -> Dict[str, object]:
    return asdict(CFG)
