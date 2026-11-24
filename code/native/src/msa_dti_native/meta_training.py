from __future__ import annotations
import copy,os,random
from pathlib import Path
from typing import Any,Dict,Mapping,Sequence
import torch
from configs.methods.h_meta_nas_cfg import CFG,config_dict
from core.space import build_model
from core.methods.ours.stage2_runtime import candidate_device,candidate_backend_context,synchronize_if_cuda
from .randomness import seed_all
from .io import sha as file_sha256

def get_support_validation_check(cfg,cache,cid,H,K):
    from .runtime import source_task
    return (*source_task(cid,H,K),None,None,None,None)

def _sha256(path: Path) -> str:
    return file_sha256(str(path))


def _save_torch(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _load_torch(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _state_key(H: int, idx: int) -> str:
    return f"h{int(H)}_a{int(idx)}"


def _mse(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((model(X) - y) ** 2).mean()


def _initial_state(spec: Any, *, H: int, input_dim: int, L: int, seed: int) -> Dict[str, torch.Tensor]:
    seed_all(seed, torch.device("cpu"))
    model = build_model(spec, input_dim=input_dim, H=H, L=L, device="cpu")
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _build_meta_bank(
    *, root: Path, cfg: Any, cache: Any, A: Sequence[Any], input_dim: int,
    L: int, device: torch.device, artifact: Path, smoke: bool,
) -> Mapping[str, Any]:
    target_iterations = 4 if smoke else CFG.source_outer_iterations
    archs_per_task = min(3 if smoke else CFG.source_architectures_per_task, len(A))
    if artifact.exists():
        saved = _load_torch(artifact)
        if saved.get("protocol") != config_dict():
            raise RuntimeError("H-Meta-NAS source artifact protocol mismatch")
        if int(saved.get("target_iterations", -1)) != int(target_iterations):
            raise RuntimeError("H-Meta-NAS smoke/formal source artifacts cannot be shared")
    else:
        states: Dict[str, Dict[str, torch.Tensor]] = {}
        for H in CFG.H_list:
            for idx, spec in enumerate(A):
                states[_state_key(H, idx)] = _initial_state(
                    spec, H=H, input_dim=input_dim, L=L,
                    seed=CFG.train_seed + 100000 * H + idx,
                )
        saved = {
            "study": "h_meta_nas_source_first_order_maml",
            "protocol": config_dict(),
            "target_iterations": target_iterations,
            "completed_iterations": 0,
            "states": states,
            "source_center_ids": list(range(CFG.source_centers)),
            "test_used": False,
            "target_pool_used": False,
        }
        _save_torch(saved, artifact)

    states = saved["states"]
    generator = random.Random(CFG.train_seed + 451)
    # Reconstruct the deterministic episode sequence, then skip completed ones.
    episodes = []
    for _ in range(target_iterations):
        episodes.append((
            generator.randrange(CFG.source_centers),
            generator.choice(CFG.H_list),
            generator.choice(CFG.K_list),
            generator.sample(range(len(A)), archs_per_task),
        ))
    begin = int(saved.get("completed_iterations", 0))
    for outer, (cid, H, K, indices) in enumerate(episodes[begin:], begin):
        Xs, ys, Xv, yv, _Xc, _yc, _tier, _ctype = get_support_validation_check(
            cfg, cache, cid, H, K
        )
        for idx in indices:
            spec = A[idx]
            actual = candidate_device(spec, device, "gru-native")
            with candidate_backend_context(spec, actual, "gru-native"):
                meta = build_model(spec, input_dim=input_dim, H=H, L=L, device=str(actual))
                meta.load_state_dict(states[_state_key(H, idx)], strict=True)
                inner = copy.deepcopy(meta)
                inner.train()
                optimizer = torch.optim.SGD(inner.parameters(), lr=CFG.source_inner_lr)
                xs, ys_dev = Xs.to(actual), ys.to(actual)
                xv, yv_dev = Xv.to(actual), yv.to(actual)
                for _ in range(CFG.source_inner_steps):
                    optimizer.zero_grad(set_to_none=True)
                    loss = _mse(inner, xs, ys_dev)
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite H-Meta-NAS source inner loss")
                    loss.backward(); optimizer.step()
                query = _mse(inner, xv, yv_dev)
                gradients = torch.autograd.grad(query, tuple(inner.parameters()))
                updated = meta.state_dict()
                for (name, _param), gradient in zip(inner.named_parameters(), gradients):
                    updated[name] = (updated[name] - CFG.source_meta_lr * gradient.detach()).cpu()
                states[_state_key(H, idx)] = {
                    name: value.detach().cpu().clone() for name, value in updated.items()
                }
                del meta, inner, optimizer
                synchronize_if_cuda(actual)
        saved["completed_iterations"] = outer + 1
        _save_torch(saved, artifact)
        print(f"[H-Meta-NAS] source episode {outer + 1}/{target_iterations}", flush=True)
    saved["complete"] = True
    _save_torch(saved, artifact)
    return saved

