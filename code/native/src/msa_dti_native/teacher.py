"""Native source teacher sampling and ranking; copied without historical data."""
from typing import Any, Dict, List, Sequence
import random
import torch

def _balanced_teacher_subset(*, feasible_indices: Sequence[int], A, n_arch: int, seed: int) -> List[int]:
    feasible = sorted({int(x) for x in feasible_indices})
    if int(n_arch) <= 0 or int(n_arch) >= len(feasible):
        return feasible
    by_family: Dict[str, List[int]] = {}
    for idx in feasible:
        by_family.setdefault(str(A[idx].family), []).append(idx)
    active = sorted(by_family)
    rng = random.Random(int(seed))
    for fam in active:
        rng.shuffle(by_family[fam])
    target = min(int(n_arch), len(feasible))
    selected: List[int] = []
    base = target // max(1, len(active))
    remainder = target % max(1, len(active))
    for pos, fam in enumerate(active):
        take = base + (1 if pos < remainder else 0)
        selected.extend(by_family[fam][:take])
    selected_set = set(selected)
    remaining = [idx for idx in feasible if idx not in selected_set]
    rng.shuffle(remaining)
    selected.extend(remaining[:max(0, target - len(selected))])
    return selected[:target]

def _compute_pi50(episodes: Sequence[Dict[str, Any]], num_arch: int) -> Dict[str, Any]:
    accum = torch.zeros(int(num_arch), dtype=torch.float64)
    counts = torch.zeros(int(num_arch), dtype=torch.float64)
    for ep in episodes:
        idxs = [int(x) for x in ep.get('candidate_indices', [])]
        losses = [float(x) for x in ep.get('validation_mse', [])]
        if not idxs or len(idxs) != len(losses):
            continue
        ordered = sorted(range(len(idxs)), key=lambda j: (losses[j], idxs[j]))
        denom = max(1, len(ordered) - 1)
        for rank, pos in enumerate(ordered):
            idx = idxs[pos]
            accum[idx] += 1.0 - float(rank) / float(denom)
            counts[idx] += 1.0
    raw = accum / counts.clamp_min(1.0)
    observed = counts > 0
    if bool(observed.any()):
        mean = raw[observed].mean()
        std = raw[observed].std(unbiased=False).clamp_min(1e-08)
        normalized = (raw - mean) / std
    else:
        normalized = raw
    return {'pi50': [float(x) for x in normalized.tolist()], 'coverage': [int(x) for x in counts.tolist()], 'coverage_min': int(counts.min().item()) if counts.numel() else 0, 'coverage_max': int(counts.max().item()) if counts.numel() else 0}
