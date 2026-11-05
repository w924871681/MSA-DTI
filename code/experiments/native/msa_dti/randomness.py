from __future__ import annotations
import random,numpy as np,torch
from typing import Optional

def seed_all(seed: int, device: Optional[torch.device] = None) -> None:
    random.seed(int(seed)); np.random.seed(int(seed) % (2**32 - 1)); torch.manual_seed(int(seed))
    if device is not None and device.type == "cuda": torch.cuda.manual_seed_all(int(seed))
