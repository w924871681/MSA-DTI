# core/config.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from types import SimpleNamespace


# =============================
# FinalCfg: runner 统一只看 method_key（CLI key）
# =============================
@dataclass
class FinalCfg:
    main: Any                 # configs.main_cfg.MainCfg 实例
    method: Any               # 当前方法 cfg 实例（OursCfg/ZeroCostNASCfg/...）
    method_key: str           # "ours" / "zerocost_nas" / "medet" / "pretrain_ft" / "meta_nas_lite" / "ours_ablation"
    methods: Any = None       # configs.methods.<name> 统一容器
    runtime: Dict[str, Any] = field(default_factory=dict)


# =============================
# Helper: 从 cfg module 里“拿到一个配置对象”
# 兼容：CFG / get_cfg() / 唯一 dataclass 实例 / 唯一 dataclass 类（自动实例化）
# =============================
def _is_dataclass_instance(x: Any) -> bool:
    return dataclasses.is_dataclass(x) and not isinstance(x, type)


def _is_dataclass_class(x: Any) -> bool:
    return dataclasses.is_dataclass(x) and isinstance(x, type)


def _pick_cfg_object(mod, prefer_cfg_attr: str = "CFG") -> Any:
    # 1) get_cfg()
    if hasattr(mod, "get_cfg") and callable(getattr(mod, "get_cfg")):
        return mod.get_cfg()

    # 2) CFG (or other preferred attr)
    if hasattr(mod, prefer_cfg_attr):
        return getattr(mod, prefer_cfg_attr)

    # 3) 如果模块里恰好有“一个 dataclass 实例”，直接用它
    inst = []
    for k, v in vars(mod).items():
        if k.startswith("_"):
            continue
        if _is_dataclass_instance(v):
            inst.append(v)
    if len(inst) == 1:
        return inst[0]

    # 4) 如果模块里恰好有“一个 dataclass 类”，自动实例化
    cls = []
    for k, v in vars(mod).items():
        if k.startswith("_"):
            continue
        if _is_dataclass_class(v):
            cls.append(v)
    if len(cls) == 1:
        return cls[0]()

    # 5) 更稳的兜底：挑一个名字像 *Cfg 的 dataclass 类（method_cfg 常见）
    cfg_like = [c for c in cls if getattr(c, "__name__", "").endswith("Cfg")]
    if len(cfg_like) == 1:
        return cfg_like[0]()

    raise ValueError(
        f"{mod.__name__} must provide get_cfg() or CFG, or contain a single dataclass instance/class."
    )


# =============================
# Loaders
# =============================
def load_main_cfg(module_path: str = "configs.main_cfg") -> Any:
    mod = importlib.import_module(module_path)

    # main_cfg.py 里如果有 CFG = MainCfg()，优先用 CFG
    cfg = _pick_cfg_object(mod, prefer_cfg_attr="CFG")

    # 调用 main 的 sanity_check（若存在）
    if hasattr(cfg, "sanity_check") and callable(getattr(cfg, "sanity_check")):
        cfg.sanity_check()
    return cfg


def load_method_cfg(method_key: str, base_module: str = "configs.methods") -> Any:
    """
    约定方法配置文件名：{method_key}_cfg.py
    例如：cfg/methods/pretrain_ft_cfg.py
    """
    module_path = f"{base_module}.{method_key}_cfg"
    mod = importlib.import_module(module_path)

    cfg = _pick_cfg_object(mod, prefer_cfg_attr="CFG")

    if hasattr(cfg, "sanity_check") and callable(getattr(cfg, "sanity_check")):
        cfg.sanity_check()
    return cfg


def _try_load_method_cfg(method_key: str, base_module: str = "configs.methods") -> Optional[Any]:
    """
    不存在则返回 None；用于 methods 容器构造（允许你暂时没实现某个 cfg 文件）。
    """
    try:
        return load_method_cfg(method_key, base_module)
    except ModuleNotFoundError:
        return None


def load_methods_container(base_module: str = "configs.methods") -> SimpleNamespace:
    """
    统一 methods 容器，让 runner 可以使用 configs.methods.<name> 访问。
    若某个方法 cfg 文件不存在，会跳过，但建议你最终都补齐。
    """
    ns = SimpleNamespace()

    # 与 run.py choices / registry 保持一致
    keys: List[str] = ["ours"]

    for k in keys:
        cfg = _try_load_method_cfg(k, base_module)
        if cfg is not None:
            setattr(ns, k, cfg)

    return ns


# =============================
# Shared experimental protocol checks
# =============================
def sanity_check_protocol(main_cfg: Any) -> None:
    # Task protocol
    L = getattr(main_cfg.task, "L")
    H_list = tuple(getattr(main_cfg.task, "H_list"))
    K_list = tuple(getattr(main_cfg.task, "K_list"))
    if L != 96:
        raise ValueError(f"Protocol mismatch: task.L={L} (expect 96)")
    if set(H_list) != {1, 4}:
        raise ValueError(f"Protocol mismatch: task.H_list={H_list} (expect (1,4))")
    if set(K_list) != {10, 20}:
        raise ValueError(f"Protocol mismatch: task.K_list={K_list} (expect (10,20))")

    # Arch space size
    if not hasattr(main_cfg, "arch") or not hasattr(main_cfg.arch, "total_size"):
        raise ValueError("MainCfg must define arch.total_size()")
    A_size = int(main_cfg.arch.total_size())
    if A_size != 66:
        raise ValueError(f"Protocol mismatch: |A|={A_size} (expect 66)")

    # NAS fairness protocol
    s = main_cfg.search
    if getattr(s, "R_candidates") != 66:
        raise ValueError(f"Protocol mismatch: search.R_candidates={s.R_candidates} (expect 66)")
    if getattr(s, "K_arch") != 12:
        raise ValueError(f"Protocol mismatch: search.K_arch={s.K_arch} (expect 12)")
    if getattr(s, "K_proxy") != 6:
        raise ValueError(f"Protocol mismatch: search.K_proxy={s.K_proxy} (expect 6)")
    if getattr(s, "T_adapt_steps") != 50:
        raise ValueError(f"Protocol mismatch: search.T_adapt_steps={s.T_adapt_steps} (expect 50)")
    if not bool(getattr(s, "strict_test_isolation", False)):
        raise ValueError("Protocol mismatch: strict_test_isolation must be enabled")
    if str(getattr(main_cfg.task, "split_mode", "")) != "chronological":
        raise ValueError("Protocol mismatch: raw timeline must be split chronologically before windowing")

    # Budget tiers existence
    b = main_cfg.budget
    for tier in ("tight", "medium", "loose"):
        if not hasattr(b, tier):
            raise ValueError(f"MainCfg.budget must contain tier '{tier}'")
    if not isinstance(getattr(b, "hard_filter", True), bool):
        raise ValueError("MainCfg.budget.hard_filter must be bool")


def load_and_merge(
    method_key: str,
    main_module: str = "configs.main_cfg",
    methods_pkg: str = "configs.methods",
    smoke: bool = False,
) -> FinalCfg:
    main = load_main_cfg(main_module)

    # Robust execution on CPU-only machines while preserving CUDA use when it
    # is actually available. This fallback changes only the runtime device, not
    # the experimental protocol.
    try:
        import torch
        if str(getattr(main, "device", "cpu")).startswith("cuda") and not torch.cuda.is_available():
            main.device = "cpu"
    except Exception:
        if str(getattr(main, "device", "cpu")).startswith("cuda"):
            main.device = "cpu"

    sanity_check_protocol(main)

    # 构建 methods 容器（供 runner 统一访问）
    methods = load_methods_container(methods_pkg)

    # 当前方法 cfg
    method = load_method_cfg(method_key, methods_pkg)

    # 保险：确保 methods.<method_key> 一定存在（防止只实现了当前而容器没挂上）
    if not hasattr(methods, method_key):
        setattr(methods, method_key, method)

    final = FinalCfg(
        main=main,
        method=method,
        method_key=method_key,
        methods=methods,
        runtime={"smoke": bool(smoke)},
    )
    return final
