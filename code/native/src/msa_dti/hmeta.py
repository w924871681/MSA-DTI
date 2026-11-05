from __future__ import annotations
import random,gc,copy
from typing import Any,Dict,List,Mapping,Optional,Sequence,Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from core.space import build_model,profile_arch,is_feasible
from core.methods.ours.adapt import adapt_steps
from core.methods.ours.stage2_runtime import candidate_device,candidate_backend_context,synchronize_if_cuda
from core.methods.ours.weight_bank import resolve_bank_key
from core.utils.metrics import eval_mae,eval_weighted_mse,eval_worst10

from configs.methods.h_meta_nas_cfg import CFG

def _state_key(H: int, idx: int) -> str:
    return f"h{int(H)}_a{int(idx)}"


def _mse(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((model(X) - y) ** 2).mean()


def _mutants(
    parent: int, feasible: Sequence[int], visited: set[int], A: Sequence[Any], rng: random.Random,
) -> List[int]:
    p = A[parent]
    def distance(idx: int) -> tuple[int, int]:
        spec = A[idx]
        family_penalty = 0 if spec.family == p.family else 10
        keys = set(spec.hparams) | set(p.hparams)
        changed = sum(spec.hparams.get(k) != p.hparams.get(k) for k in keys)
        return family_penalty + changed, idx
    candidates = [idx for idx in feasible if idx not in visited and idx != parent]
    candidates.sort(key=distance)
    if not candidates:
        return []
    best = [idx for idx in candidates if distance(idx)[0] == distance(candidates[0])[0]]
    rng.shuffle(best)
    return best + [idx for idx in candidates if idx not in best]


def _adapt_and_validate(
    *, spec: Any, state: Mapping[str, torch.Tensor], Xs: torch.Tensor, ys: torch.Tensor,
    Xv: torch.Tensor, yv: torch.Tensor, H: int, L: int, input_dim: int,
    device: torch.device, seed: int,
) -> tuple[torch.nn.Module, Dict[str, float]]:
    actual = candidate_device(spec, device, "gru-native")
    with candidate_backend_context(spec, actual, "gru-native"):
        seed_all(seed, actual)
        model = build_model(spec, input_dim=input_dim, H=H, L=L, device=str(actual))
        model.load_state_dict(state, strict=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=CFG.target_lr)
        xs, ys_dev = Xs.to(actual), ys.to(actual)
        for _ in range(CFG.target_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = _mse(model, xs, ys_dev)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite H-Meta-NAS target loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.target_grad_clip)
            optimizer.step()
        validation = eval_metrics(model, Xv, yv)
        synchronize_if_cuda(actual)
        return model, validation


def seed_all(seed: int, device: Optional[torch.device] = None) -> None:
    random.seed(int(seed)); np.random.seed(int(seed) % (2**32 - 1)); torch.manual_seed(int(seed))
    if device is not None and device.type == "cuda": torch.cuda.manual_seed_all(int(seed))


def eval_metrics(model, X, y) -> Dict[str, float]:
    dev = next(model.parameters()).device; model.eval()
    with torch.no_grad():
        pred = model(X.to(dev).contiguous()); target = y.to(dev).contiguous()
        return {"weighted_mse": float(eval_weighted_mse(pred, target)), "mae": float(eval_mae(pred, target)), "worst10": float(eval_worst10(pred, target, p=0.10))}

