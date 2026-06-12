"""Utility functions to create optimizers dynamically."""
from __future__ import annotations

import importlib
import inspect
from typing import Any, Dict

import torch

# Cache of optimizer classes for quick lookup
_OPTIMIZER_CLASSES: Dict[str, type] | None = None
_TORCH_OPT = None
_TORCH_OPT_LOADED = False


def _get_torch_optimizer_module(force_reload: bool = False):
    """Load pytorch-optimizer lazily.

    This allows installation at runtime (e.g., from a notebook cell) without
    requiring a kernel restart before OptiRoulette can resolve advanced
    optimizers such as AdaBelief.
    """
    global _TORCH_OPT, _TORCH_OPT_LOADED
    if force_reload:
        _TORCH_OPT = None
        _TORCH_OPT_LOADED = False
    if not _TORCH_OPT_LOADED:
        try:
            _TORCH_OPT = importlib.import_module("pytorch_optimizer")
        except ImportError:  # pragma: no cover - exercised when optional dep missing locally
            _TORCH_OPT = None
        _TORCH_OPT_LOADED = True
    return _TORCH_OPT


def _build_optimizer_dict() -> Dict[str, type]:
    """Collect optimizer classes from torch and optional pytorch-optimizer.

    Returns:
        Mapping from lowercase optimizer name to optimizer class.
    """
    classes: Dict[str, type] = {}
    modules = [torch.optim]
    torch_opt = _get_torch_optimizer_module()
    if torch_opt is not None:
        modules.append(torch_opt)
    for module in modules:
        for name, cls in module.__dict__.items():
            if inspect.isclass(cls) and issubclass(cls, torch.optim.Optimizer):
                classes[name.lower()] = cls
    return classes


def _get_optimizer_class(name: str) -> type:
    """Resolve an optimizer class by name.

    Args:
        name: Optimizer name (case-insensitive).

    Returns:
        Optimizer class object.

    Raises:
        ValueError: If the optimizer name cannot be resolved.
    """
    global _OPTIMIZER_CLASSES
    if _OPTIMIZER_CLASSES is None:
        _OPTIMIZER_CLASSES = _build_optimizer_dict()
    key = name.lower()
    if key == "amsgrad":
        key = "adam"  # AMSGrad is Adam with amsgrad=True
    if key not in _OPTIMIZER_CLASSES:
        # If dependency was installed after initial import, refresh once.
        if _get_torch_optimizer_module(force_reload=True) is not None:
            _OPTIMIZER_CLASSES = _build_optimizer_dict()
        if key in _OPTIMIZER_CLASSES:
            return _OPTIMIZER_CLASSES[key]
        if _get_torch_optimizer_module() is None:
            raise ValueError(
                f"Unknown optimizer: {name}. Install 'pytorch-optimizer' for advanced optimizers."
            )
        raise ValueError(f"Unknown optimizer: {name}")
    return _OPTIMIZER_CLASSES[key]


def create_optimizer(name: str, params, cfg: Dict[str, Any]):
    """Instantiate optimizer by name using configuration dictionary.

    This supports both ``torch.optim`` and ``pytorch_optimizer`` optimizers.
    Any nested optimizer references such as ``base`` or ``base_optimizer``
    specified as strings will be resolved recursively.

    Args:
        name: Optimizer name.
        params: Model parameters (iterator/list/param groups).
        cfg: Optimizer configuration dictionary.

    Returns:
        Instantiated optimizer object.
    """
    cfg = cfg.copy() if cfg is not None else {}
    if "nus" in cfg and isinstance(cfg["nus"], list):
        cfg["nus"] = tuple(cfg["nus"])
    name_lower = name.lower()
    if name_lower in {"opti_roulette", "optiroulette", "roulette"}:
        from .opti_roulette import OptiRoulette

        optimizer_specs = cfg.pop("optimizer_specs", None)
        optimizers = cfg.pop("optimizers", None)
        if optimizer_specs is None and optimizers is not None:
            if isinstance(optimizers, dict):
                if not all(
                    isinstance(value, torch.optim.Optimizer)
                    for value in optimizers.values()
                ):
                    optimizer_specs = optimizers
                    optimizers = None
            elif isinstance(optimizers, (list, tuple)):
                optimizer_specs = optimizers
                optimizers = None
        if "optimizer_pool" in cfg and "pool_config" not in cfg:
            cfg["pool_config"] = cfg.pop("optimizer_pool")
        if "pool" in cfg and "pool_config" not in cfg:
            cfg["pool_config"] = cfg.pop("pool")
        # Deprecated legacy flag; defaults are now automatic when manual specs
        # are not provided.
        cfg.pop("use_optimized_defaults", None)

        return OptiRoulette(
            params,
            optimizer_specs=optimizer_specs,
            optimizers=optimizers,
            **cfg,
        )
    base_cls = None
    # Resolve nested optimizer names
    for key in ["base_optimizer", "base_opt", "optimizer", "optim", "base"]:
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = _get_optimizer_class(cfg[key])
        if key in cfg and isinstance(cfg[key], type):
            base_cls = cfg.pop(key)
    if name.lower() == "amsgrad":
        cfg["amsgrad"] = True
        cls = _get_optimizer_class("adam")
    else:
        cls = _get_optimizer_class(name)
    #print(_OPTIMIZER_CLASSES)
    if base_cls is not None:
        return cls(params, base_cls, **cfg)
    return cls(params, **cfg)
