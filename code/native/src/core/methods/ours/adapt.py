# core/methods/ours/adapt.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim


def _iter_minibatches(X: torch.Tensor, y: torch.Tensor, batch_size: int, seed: int):
    n = int(X.shape[0])
    if batch_size <= 0 or batch_size >= n:
        yield X, y
        return
    g = torch.Generator(device='cpu')
    g.manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size].to(X.device)
        yield X[idx], y[idx]


def _samplewise_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).reshape(pred.shape[0], -1).mean(dim=1)


def _samplewise_huber(pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
    err = (pred - target).abs()
    delta_t = torch.tensor(float(delta), device=err.device, dtype=err.dtype)
    quad = torch.minimum(err, delta_t)
    lin = err - quad
    huber = 0.5 * quad.pow(2) + delta_t * lin
    return huber.reshape(pred.shape[0], -1).mean(dim=1)


def _samplewise_trimmed_mse(pred: torch.Tensor, target: torch.Tensor, trimmed_ratio: float) -> torch.Tensor:
    err = ((pred - target) ** 2).reshape(pred.shape[0], -1)
    k = max(1, int(round((1.0 - float(trimmed_ratio)) * err.shape[1])))
    vals, _ = torch.sort(err, dim=1)
    return vals[:, :k].mean(dim=1)


def _base_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str = 'huber', huber_delta: float = 1.0, trimmed_ratio: float = 0.10):
    lt = str(loss_type).lower()
    if lt == 'mse':
        per = _samplewise_mse(pred, target)
    elif lt == 'trimmed':
        per = _samplewise_trimmed_mse(pred, target, trimmed_ratio=trimmed_ratio)
    else:
        per = _samplewise_huber(pred, target, delta=huber_delta)
    return per.mean(), per


def _cvar_tail_penalty(per_sample_loss: torch.Tensor, alpha: float) -> torch.Tensor:
    a = float(alpha)
    if a <= 0.0 or per_sample_loss.numel() == 0:
        return torch.tensor(0.0, device=per_sample_loss.device, dtype=per_sample_loss.dtype)
    k = max(1, int(math.ceil(a * per_sample_loss.numel())))
    tail = torch.topk(per_sample_loss, k=k, largest=True).values
    return tail.mean()


def _guess_param_group(name: str) -> str:
    s = str(name).lower()
    if 'norm' in s:
        return 'norm'
    if 'head' in s or 'proj_out' in s or 'fc_out' in s or s.endswith('bias'):
        return 'head'
    if 'gru' in s or 'rnn' in s:
        return 'recurrent'
    if 'conv' in s or 'tcn' in s:
        return 'temporal'
    return 'shared'


def _prior_penalty(model: nn.Module, prior_state: Optional[Dict[str, torch.Tensor]], prior_group_lambdas: Optional[Dict[str, float]]) -> torch.Tensor:
    dev = next(model.parameters()).device
    if not prior_state or not prior_group_lambdas:
        return torch.tensor(0.0, device=dev)
    total = torch.tensor(0.0, device=dev)
    for name, p in model.named_parameters():
        if (not p.requires_grad) or name not in prior_state:
            continue
        group = _guess_param_group(name)
        lam = float(prior_group_lambdas.get(group, prior_group_lambdas.get('__all__', 0.0)))
        if lam <= 0.0:
            continue
        p0 = prior_state[name].to(dev)
        total = total + lam * torch.mean((p - p0) ** 2)
    return total


def adapt_steps(
    model: nn.Module,
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    steps: int,
    lr: float,
    weight_decay: float = 0.0,
    *,
    robust_loss_type: str = 'huber',
    huber_delta: float = 1.0,
    trimmed_ratio: float = 0.10,
    cvar_alpha: float = 0.20,
    cvar_lambda: float = 0.0,
    prior_state: Optional[Dict[str, torch.Tensor]] = None,
    prior_group_lambdas: Optional[Dict[str, float]] = None,
    batch_size: int = 0,
    use_amp: bool = False,
    oom_to_cpu: bool = True,
    seed: int = 0,
    max_grad_norm: float = 5.0,
) -> Dict[str, float]:
    stats = {
        'adapt_loss_base': 0.0,
        'adapt_loss_tail': 0.0,
        'adapt_loss_prior': 0.0,
        'adapt_loss_total': 0.0,
        'adapt_steps_done': 0.0,
    }

    def _run(dev: torch.device):
        model.to(dev)
        X = X_tr.to(dev)
        y = y_tr.to(dev)
        model.train()
        opt = optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        amp_enabled = bool(use_amp and dev.type == 'cuda')
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        else:  # compatibility with older supported PyTorch releases
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        step_cnt = 0
        base_acc = 0.0
        tail_acc = 0.0
        prior_acc = 0.0
        total_acc = 0.0
        for t in range(int(steps)):
            for Xb, yb in _iter_minibatches(X, y, int(batch_size), seed=int(seed) + 1000 * t):
                opt.zero_grad(set_to_none=True)
                if dev.type == 'cuda' and bool(use_amp):
                    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                        autocast_context = torch.amp.autocast("cuda", enabled=amp_enabled)
                    else:
                        autocast_context = torch.cuda.amp.autocast(enabled=amp_enabled)
                    with autocast_context:
                        pred = model(Xb)
                        base_loss, per_sample = _base_loss(pred, yb, loss_type=str(robust_loss_type), huber_delta=float(huber_delta), trimmed_ratio=float(trimmed_ratio))
                        tail_pen = _cvar_tail_penalty(per_sample, alpha=float(cvar_alpha))
                        prior_pen = _prior_penalty(model, prior_state=prior_state, prior_group_lambdas=prior_group_lambdas)
                        loss = base_loss + float(cvar_lambda) * tail_pen + prior_pen
                    scaler.scale(loss).backward()
                    if float(max_grad_norm) > 0.0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
                    scaler.step(opt)
                    scaler.update()
                else:
                    pred = model(Xb)
                    base_loss, per_sample = _base_loss(pred, yb, loss_type=str(robust_loss_type), huber_delta=float(huber_delta), trimmed_ratio=float(trimmed_ratio))
                    tail_pen = _cvar_tail_penalty(per_sample, alpha=float(cvar_alpha))
                    prior_pen = _prior_penalty(model, prior_state=prior_state, prior_group_lambdas=prior_group_lambdas)
                    loss = base_loss + float(cvar_lambda) * tail_pen + prior_pen
                    loss.backward()
                    if float(max_grad_norm) > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
                    opt.step()
                step_cnt += 1
                base_acc += float(base_loss.detach().item())
                tail_acc += float(tail_pen.detach().item())
                prior_acc += float(prior_pen.detach().item())
                total_acc += float(loss.detach().item())
        denom = max(1, step_cnt)
        stats['adapt_loss_base'] = base_acc / denom
        stats['adapt_loss_tail'] = tail_acc / denom
        stats['adapt_loss_prior'] = prior_acc / denom
        stats['adapt_loss_total'] = total_acc / denom
        stats['adapt_steps_done'] = float(step_cnt)

    try:
        dev0 = next(model.parameters()).device
        _run(dev0)
    except RuntimeError as e:
        msg = str(e).lower()
        if ('out of memory' in msg or 'cuda error' in msg) and bool(oom_to_cpu):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            _run(torch.device('cpu'))
        else:
            raise
    return stats


def adapt_condition_modulated(
    model: nn.Module,
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    *,
    steps: int,
    lr: float,
    modulation: torch.Tensor,
    prior_lambda: float | torch.Tensor,
    prior_state: Optional[Dict[str, torch.Tensor]],
    weight_decay: float = 0.0,
    huber_delta: float = 0.2,
    batch_size: int = 0,
    seed: int = 0,
    max_grad_norm: float = 5.0,
) -> Dict[str, float]:
    """Paper-aligned target-side update.

    Implements the practical optimizer form of
      theta <- theta - alpha * Expand_a(m_c) odot grad L_tr
               - alpha * lambda_c * (theta - theta_0).

    Adam supplies the base optimizer step. The prediction-loss gradients are
    explicitly multiplied by the condition-generated parameter-group
    modulation, after which the unmodulated prior-preservation gradient is
    added. This preserves the two-term update structure used in the paper.
    """
    from core.methods.ours.paper_modules import GROUP_TO_ID, parameter_group

    dev = next(model.parameters()).device
    X = X_tr.to(dev)
    y = y_tr.to(dev)
    modulation = modulation.detach().to(dev).reshape(-1)
    if modulation.numel() != len(GROUP_TO_ID):
        raise ValueError(
            f"Expected {len(GROUP_TO_ID)} modulation values, got {modulation.numel()}"
        )
    lam = float(prior_lambda.detach().item()) if torch.is_tensor(prior_lambda) else float(prior_lambda)

    model.train()
    opt = optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    stats = {"adapt_loss": 0.0, "adapt_prior": 0.0, "steps": 0.0}
    total_loss = 0.0
    total_prior = 0.0
    count = 0

    for t in range(int(steps)):
        for Xb, yb in _iter_minibatches(X, y, int(batch_size), seed=int(seed) + 1000 * t):
            opt.zero_grad(set_to_none=True)
            pred = model(Xb)
            base_loss, _ = _base_loss(
                pred,
                yb,
                loss_type="huber",
                huber_delta=float(huber_delta),
                trimmed_ratio=0.0,
            )
            # First obtain the prediction-loss gradient and apply
            # Expand_a(m_c) only to this gradient, exactly as in the paper.
            base_loss.backward()
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                gid = GROUP_TO_ID[parameter_group(name)]
                p.grad.mul_(modulation[gid])

            # Add the prior-preservation gradient afterwards so it is not
            # multiplied by m_c. For mean((theta-theta_0)^2), the gradient is
            # 2*(theta-theta_0)/numel(theta).
            prior_pen = torch.tensor(0.0, device=dev, dtype=base_loss.dtype)
            if prior_state and lam > 0.0:
                for name, p in model.named_parameters():
                    if name not in prior_state:
                        continue
                    p0 = prior_state[name].to(device=dev, dtype=p.dtype)
                    diff = p - p0
                    prior_pen = prior_pen + torch.mean(diff ** 2)
                    prior_grad = (2.0 * lam / max(1, p.numel())) * diff.detach()
                    if p.grad is None:
                        p.grad = prior_grad.clone()
                    else:
                        p.grad.add_(prior_grad)

            if float(max_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
            opt.step()

            total_loss += float(base_loss.detach().item())
            total_prior += float(prior_pen.detach().item())
            count += 1

    denom = max(1, count)
    stats["adapt_loss"] = total_loss / denom
    stats["adapt_prior"] = total_prior / denom
    stats["steps"] = float(count)
    return stats
