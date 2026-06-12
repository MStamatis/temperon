"""Default OptiRoulette settings loaded from the bundled optimized profile."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple

import yaml
import importlib.resources as importlib_resources

try:
    from importlib.resources import files as resource_files
except ImportError:  # Python 3.8
    resource_files = None

_DEFAULT_CACHE: Dict[str, Any] | None = None


def _load_default_config() -> Dict[str, Any]:
    """Load and cache bundled `optimized.yaml`, then return a deep copy.

    The cache avoids repeated disk/package-resource reads, while returning a
    deep copy prevents callers from mutating shared state.
    """
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        if resource_files is not None:
            path = resource_files("optiroulette.resources").joinpath("optimized.yaml")
            with path.open("r", encoding="utf-8") as handle:
                _DEFAULT_CACHE = yaml.safe_load(handle) or {}
        else:
            with importlib_resources.open_text(
                "optiroulette.resources",
                "optimized.yaml",
                encoding="utf-8",
            ) as handle:
                _DEFAULT_CACHE = yaml.safe_load(handle) or {}
    return deepcopy(_DEFAULT_CACHE)


def get_default_config() -> Dict[str, Any]:
    """Return the full bundled default configuration."""
    return _load_default_config()


def get_default_seed() -> int:
    """Return the default deterministic seed from bundled configuration."""
    system_cfg = _load_default_config().get("system", {}) or {}
    return int(system_cfg.get("seed", 42))


def get_default_optimizer_specs() -> Dict[str, Dict[str, Any]]:
    """Return default optimizer specifications keyed by optimizer name."""
    optimizers = _load_default_config().get("optimizers", {})
    return {str(name): (cfg or {}) for name, cfg in optimizers.items()}


def get_default_roulette_config() -> Dict[str, Any]:
    """Return merged roulette defaults including warmup-related fields.

    Warmup fields are sourced from the training section for backward
    compatibility with profile layout.
    """
    cfg = _load_default_config()
    roulette_cfg = dict(cfg.get("roulette", {}) or {})
    training_cfg = cfg.get("training", {}) or {}
    phase_cfg = training_cfg.get("phases", {}) or {}
    warmup_phase_cfg = phase_cfg.get("warmup", {}) or {}

    roulette_cfg.setdefault("warmup_optimizer", training_cfg.get("warmup_optimizer"))
    roulette_cfg.setdefault("warmup_epochs", int(training_cfg.get("warmup_epochs", 0) or 0))
    roulette_cfg.setdefault(
        "drop_after_warmup",
        bool(training_cfg.get("dropafter_warmup", False)),
    )
    if warmup_phase_cfg:
        roulette_cfg.setdefault("warmup_config", dict(warmup_phase_cfg))
    return roulette_cfg


def get_default_lr_scaling_rules() -> Dict[str, Any]:
    """Return LR compatibility scaling rules from pool defaults."""
    pool_cfg = _load_default_config().get("optimizer_pool", {}) or {}
    return dict(pool_cfg.get("lr_scaling_rules", {}) or {})


def get_default_pool_setup() -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Return default pool config plus active/backup optimizer name lists.

    Returns:
        Tuple of `(pool_config_dict, active_names, backup_names)`.
    """
    cfg = _load_default_config()
    pool_cfg = dict(cfg.get("optimizer_pool", {}) or {})
    active_names = list(pool_cfg.pop("active_optimizers", []) or [])
    backup_names = list(pool_cfg.pop("backup_optimizers", []) or [])

    if not backup_names:
        all_optimizer_names = list((cfg.get("optimizers", {}) or {}).keys())
        backup_names = [name for name in all_optimizer_names if name not in active_names]

    return pool_cfg, active_names, backup_names
