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

from configs.methods.main_evaluation_cfg import CFG

def _fixed_target_adapt(
    model: nn.Module,
    Xs: torch.Tensor,
    ys: torch.Tensor,
    *,
    seed: int,
) -> None:
    dev = next(model.parameters()).device
    seed_all(seed, dev)
    model.train()
    opt = optim.SGD(model.parameters(), lr=CFG.fixed_target_lr)
    Xd, yd = Xs.to(dev).contiguous(), ys.to(dev).contiguous()
    for _ in range(CFG.fixed_target_steps):
        opt.zero_grad(set_to_none=True)
        loss = ((model(Xd) - yd) ** 2).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite SGD/MSE target loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.fixed_target_grad_clip
        )
        opt.step()


def _search_target_adapt(
    model: nn.Module, Xs: torch.Tensor, ys: torch.Tensor, *, seed: int
) -> None:
    dev = next(model.parameters()).device
    adapt_steps(
        model,
        Xs.to(dev).contiguous(),
        ys.to(dev).contiguous(),
        steps=CFG.search_target_steps,
        lr=CFG.search_target_lr,
        weight_decay=0.0,
        robust_loss_type="huber",
        huber_delta=CFG.search_target_huber_delta,
        cvar_lambda=0.0,
        prior_state=None,
        prior_group_lambdas=None,
        batch_size=0,
        use_amp=False,
        oom_to_cpu=False,
        seed=int(seed),
        max_grad_norm=CFG.search_target_grad_clip,
    )


def _sample_proxy_batch(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(X.shape[0])
    take = min(max(1, CFG.proxy_support_points), n)
    gen = torch.Generator(device=X.device)
    gen.manual_seed(int(seed))
    idx = torch.randperm(n, generator=gen, device=X.device)[:take]
    Xp, yp = X.index_select(0, idx), y.index_select(0, idx)
    if int(Xp.shape[0]) > CFG.proxy_batch_size:
        idx2 = torch.randperm(
            int(Xp.shape[0]), generator=gen, device=X.device
        )[: CFG.proxy_batch_size]
        Xp, yp = Xp.index_select(0, idx2), yp.index_select(0, idx2)
    return Xp, yp


def _entropy_proxy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.train()
    model.zero_grad(set_to_none=True)
    loss = ((model(X) - y) ** 2).mean()
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        loss, params, retain_graph=False, create_graph=False, allow_unused=True
    )
    chunks = [g.detach().reshape(-1).abs() for g in grads if g is not None]
    if not chunks:
        return float("-inf")
    flat = torch.cat(chunks)
    total = flat.sum()
    if not torch.isfinite(total) or float(total.item()) <= 0.0:
        return float("-inf")
    prob = (flat / (total + 1e-12)).clamp_min(1e-12)
    return float((-(prob * prob.log()).sum()).item())


def _rank_score(
    values: Mapping[int, float], *, larger_is_better: bool
) -> Dict[int, float]:
    items = sorted(
        values.items(), key=lambda kv: kv[1], reverse=larger_is_better
    )
    n = len(items)
    if n <= 1:
        return {int(k): 1.0 for k, _ in items}
    return {
        int(k): 1.0 - float(rank) / float(n - 1)
        for rank, (k, _value) in enumerate(items)
    }


def _run_search_case(
    method: str,
    frozen: Mapping[str, Any],
    cfg: Any,
    A: Sequence[Any],
    requested: torch.device,
    safe_mode: str,
    Xs: torch.Tensor,
    ys: torch.Tensor,
    Xv: torch.Tensor,
    yv: torch.Tensor,
    tier: str,
    H: int,
    input_dim: int,
    L: int,
    seed: int,
):
    feasible = feasible_indices(cfg, A, tier, L, input_dim, H)
    if not feasible:
        raise RuntimeError("No hard-feasible architecture")
    Xp, yp = _sample_proxy_batch(Xs.to(requested), ys.to(requested), seed=seed)
    proxy_raw: Dict[int, float] = {}
    for idx in feasible:
        spec = A[idx]
        actual = candidate_device(spec, requested, safe_mode)
        with candidate_backend_context(spec, actual, safe_mode):
            seed_all(seed + idx, actual)
            proxy_model = build_model(
                spec, input_dim=input_dim, H=H, L=L, device=str(actual)
            )
            proxy_raw[idx] = _entropy_proxy(
                proxy_model, Xp.to(actual), yp.to(actual)
            )
            del proxy_model
            synchronize_if_cuda(actual)
    proxy_rank = _rank_score(proxy_raw, larger_is_better=True)

    if method == "meta_nas_lite":
        source_rank = _rank_score(
            {i: float(frozen["pi50"][i]) for i in feasible},
            larger_is_better=True,
        )
        fused = {
            i: CFG.meta_prior_weight * source_rank[i]
            + CFG.meta_proxy_weight * proxy_rank[i]
            for i in feasible
        }
        shortlist = sorted(feasible, key=lambda i: (-fused[i], i))[
            : CFG.candidate_budget
        ]
        adapt = True
        selector_name = "frozen_meta_prior_plus_proxy_top12_valbest"
    elif method == "zero_nas":
        shortlist = sorted(feasible, key=lambda i: (-proxy_rank[i], i))[:1]
        adapt = False
        selector_name = "frozen_zero_proxy_top1_no_ft"
    elif method == "zero_nas_ft":
        shortlist = sorted(feasible, key=lambda i: (-proxy_rank[i], i))[
            : CFG.candidate_budget
        ]
        adapt = True
        selector_name = "frozen_zero_proxy_top12_ft50_valbest"
    else:
        raise ValueError(method)

    selected_model = None
    selected_spec = None
    best_row = None
    summaries: List[Dict[str, Any]] = []
    adapted_count = 0
    for idx in shortlist:
        spec = A[idx]
        actual = candidate_device(spec, requested, safe_mode)
        with candidate_backend_context(spec, actual, safe_mode):
            model, _prior = _load_prior_model(
                spec=spec,
                H=H,
                L=L,
                input_dim=input_dim,
                bank=frozen["bank"],
                device=actual,
            )
            if adapt:
                _search_target_adapt(model, Xs, ys, seed=seed + idx)
                adapted_count += 1
            val = eval_metrics(model, Xv, yv)
            params, flops = profile_arch(spec, L=L, input_dim=input_dim, H=H)
            row = {
                "arch_idx": int(idx),
                "arch_key": str(spec.arch_key),
                "family": str(spec.family),
                "params": float(params),
                "flops": float(flops),
                "validation": val,
            }
            summaries.append(row)
            lex = (
                float(val["weighted_mse"]),
                float(params),
                float(flops),
                int(idx),
            )
            if best_row is None or lex < best_row[0]:
                if selected_model is not None:
                    del selected_model
                selected_model = model
                selected_spec = spec
                best_row = (lex, row)
            else:
                del model
            synchronize_if_cuda(actual)
    if selected_model is None or selected_spec is None or best_row is None:
        raise RuntimeError("Search baseline failed to select a model")
    selector = {
        "selector": selector_name,
        "shortlist": [int(x) for x in shortlist],
        "selected_arch_idx": int(best_row[1]["arch_idx"]),
        "candidate_validations": summaries,
        "selection_uses_check": False,
        "selection_uses_test": False,
    }
    return selected_model, selected_spec, best_row[1], selector, len(shortlist), adapted_count


def seed_all(seed: int, device: Optional[torch.device] = None) -> None:
    random.seed(int(seed)); np.random.seed(int(seed) % (2**32 - 1)); torch.manual_seed(int(seed))
    if device is not None and device.type == "cuda": torch.cuda.manual_seed_all(int(seed))


def eval_metrics(model, X, y) -> Dict[str, float]:
    dev = next(model.parameters()).device; model.eval()
    with torch.no_grad():
        pred = model(X.to(dev).contiguous()); target = y.to(dev).contiguous()
        return {"weighted_mse": float(eval_weighted_mse(pred, target)), "mae": float(eval_mae(pred, target)), "worst10": float(eval_worst10(pred, target, p=0.10))}


def feasible_indices(cfg, A: Sequence[Any], tier: str, L: int, input_dim: int, H: int) -> List[int]:
    return [i for i, spec in enumerate(A) if bool(is_feasible(spec, cfg.main.budget, str(tier), int(L), int(input_dim), int(H)))]


def _load_prior_model(
    *,
    spec,
    H: int,
    L: int,
    input_dim: int,
    bank: Dict[str, Dict[str, torch.Tensor]],
    device: torch.device,
):
    model = build_model(spec, input_dim=int(input_dim), H=int(H), L=int(L), device=str(device))
    key, _hit = resolve_bank_key(
        bank,
        H=int(H),
        arch_key=str(spec.arch_key),
        input_dim=int(input_dim),
        L=int(L),
    )
    if key is not None:
        model.load_state_dict(bank[key], strict=True)
        prior_state = {k: v.detach().clone() for k, v in bank[key].items()}
    else:
        prior_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return model, prior_state

