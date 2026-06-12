"""Random optimizer-switching wrapper for OptiRoulette."""
from __future__ import annotations

import logging
import random
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from .compatibility import OptimizerCompatibility
from .defaults import (
    get_default_seed,
    get_default_lr_scaling_rules,
    get_default_optimizer_specs,
    get_default_pool_setup,
    get_default_roulette_config,
)
from .optimizer_pool import OptimizerPoolManager, PoolConfig


def _normalize_optimizer_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize one optimizer config dict to runtime-friendly types."""
    normalized = dict(cfg or {})
    if "betas" in normalized and isinstance(normalized["betas"], list):
        normalized["betas"] = tuple(normalized["betas"])
    if "nus" in normalized and isinstance(normalized["nus"], list):
        normalized["nus"] = tuple(normalized["nus"])
    return normalized


def _normalize_optimizer_specs(specs: Any) -> Dict[str, Dict[str, Any]]:
    """Normalize user optimizer specs into a `{name: config}` mapping."""
    if specs is None:
        return {}
    if isinstance(specs, dict):
        return {str(name): _normalize_optimizer_cfg(cfg) for name, cfg in specs.items()}
    if isinstance(specs, list):
        normalized: Dict[str, Dict[str, Any]] = {}
        for item in specs:
            if isinstance(item, str):
                normalized[item] = {}
                continue
            if isinstance(item, tuple) and len(item) == 2:
                name, cfg = item
                normalized[str(name)] = _normalize_optimizer_cfg(cfg)
                continue
            if isinstance(item, dict):
                if "name" not in item:
                    raise ValueError("Optimizer spec dict requires a 'name' key.")
                name = str(item["name"])
                cfg = {k: v for k, v in item.items() if k != "name"}
                normalized[name] = _normalize_optimizer_cfg(cfg)
                continue
            raise TypeError(f"Unsupported optimizer spec format: {type(item)}")
        return normalized
    raise TypeError("optimizer_specs must be a dict, list, or None.")


class OptiRouletteOptimizer(torch.optim.Optimizer):
    """Wrap multiple optimizers and switch between them randomly.

    Notes:
        External training loops are expected to call the lifecycle hooks
        (`on_epoch_start`, `on_batch_start`, `on_epoch_end`) so switching and
        warmup logic is evaluated at the intended times.
    """

    def __init__(
        self,
        optimizers: Dict[str, torch.optim.Optimizer],
        *,
        switch_granularity: str = "epoch",
        switch_every_steps: int = 1,
        switch_probability: float = 1.0,
        avoid_repeat: bool = True,
        warmup_optimizer: Optional[str] = None,
        drop_after_warmup: bool = False,
        warmup_epochs: int = 0,
        warmup_config: Optional[Dict[str, Any]] = None,
        lr_scaling_rules: Optional[Dict[str, Any]] = None,
        pool_config: Optional[PoolConfig] = None,
        active_names: Optional[List[str]] = None,
        backup_names: Optional[List[str]] = None,
        seed: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize roulette state around pre-built optimizer instances."""
        if not optimizers:
            raise ValueError("OptiRouletteOptimizer requires at least one optimizer.")

        first_optimizer = next(iter(optimizers.values()))
        defaults = getattr(first_optimizer, "defaults", {})
        super().__init__(first_optimizer.param_groups, defaults)

        self.optimizers = optimizers
        self.logger = logger or logging.getLogger(__name__)
        self.random = random.Random(seed)
        self.switch_granularity = switch_granularity
        self.switch_every_steps = max(1, int(switch_every_steps))
        self.switch_probability = float(switch_probability)
        self.avoid_repeat = avoid_repeat
        self.warmup_optimizer = warmup_optimizer
        self.drop_after_warmup = drop_after_warmup
        self.warmup_epochs = max(0, int(warmup_epochs))
        warmup_config = warmup_config or {}
        self.warmup_config = warmup_config
        # Warmup has two modes:
        # 1) fixed: leave warmup after `warmup_epochs`
        # 2) plateau: leave warmup when validation plateaus (via on_epoch_end)
        self._warmup_enabled = self.warmup_epochs > 0 or bool(warmup_config)
        self._warmup_mode = "fixed" if self.warmup_epochs > 0 else "plateau"
        if warmup_config:
            self._warmup_min_val_acc = float(warmup_config.get("min_val_acc", 0.0))
            self._warmup_plateau_threshold = float(
                warmup_config.get("plateau_threshold", 0.001)
            )
            self._warmup_min_epochs = int(warmup_config.get("min_epochs", 0))
        else:
            self._warmup_min_val_acc = 0.0
            self._warmup_plateau_threshold = 0.0
            self._warmup_min_epochs = 0
        self._warmup_val_acc_hist = deque(
            maxlen=max(5, self._warmup_min_epochs)
        )
        self._warmup_completed = False
        self._warmup_lock_name: Optional[str] = None
        self.phase = "warmup" if self._warmup_enabled else "roulette"

        self.compatibility: Optional[OptimizerCompatibility] = None
        # LR compatibility adjusts learning rate when moving between optimizer
        # families (e.g., adaptive -> momentum methods).
        if lr_scaling_rules:
            self.compatibility = OptimizerCompatibility(
                {"lr_scaling_rules": lr_scaling_rules}
            )

        self.pool: Optional[OptimizerPoolManager] = None
        # Pool manager can replace failing optimizers with backups while
        # keeping a limited active set.
        if pool_config is not None:
            active = active_names or list(optimizers.keys())[: pool_config.num_active]
            remaining = [n for n in optimizers.keys() if n not in active]
            backups = backup_names or remaining
            self.pool = OptimizerPoolManager(
                optimizers, active, backups, pool_config, logger=self.logger
            )
            self.active_names = self.pool.get_active_names()
        else:
            self.active_names = list(optimizers.keys())

        self.blocked_names: set[str] = set()
        self.locked_optimizer: Optional[str] = None
        # Start from warmup optimizer (if valid), otherwise first active entry.
        self.current_name = (
            warmup_optimizer if warmup_optimizer in optimizers else self.active_names[0]
        )
        self.current_optimizer = optimizers[self.current_name]
        self.param_groups = self.current_optimizer.param_groups
        self.state = self.current_optimizer.state
        self.defaults = getattr(self.current_optimizer, "defaults", {})

        self.selection_counts = {name: 0 for name in optimizers.keys()}
        self.switch_history: List[Tuple[Optional[str], str, str]] = []
        self.phase_history: List[Tuple[int, str]] = [(0, self.phase)]
        self.epoch = 0
        self.step_count = 0
        self._apply_phase_state(initial=True)

    @property
    def active_optimizer(self) -> torch.optim.Optimizer:
        """Return the currently active underlying optimizer object."""
        return self.current_optimizer

    @property
    def active_optimizer_name(self) -> str:
        """Return the name of the currently active optimizer."""
        return self.current_name

    def set_phase(self, phase: Optional[str]) -> None:
        """Set execution phase (`warmup` or `roulette`) when provided."""
        if phase:
            self.phase = phase

    def _lock_for_warmup(self) -> None:
        """Lock optimizer selection to the configured warmup optimizer."""
        target = (
            self.warmup_optimizer
            if self.warmup_optimizer in self.optimizers
            else self.current_name
        )
        self._warmup_lock_name = target
        self.lock_optimizer(target)

    def _apply_phase_state(self, *, initial: bool = False) -> None:
        """Apply phase side effects such as warmup lock and post-warmup blocking."""
        if self.phase == "warmup":
            # During warmup we force optimizer selection through a lock so random
            # switching cannot bypass the warmup choice.
            self._lock_for_warmup()
        else:
            if self._warmup_completed and self.drop_after_warmup and self.warmup_optimizer:
                self.block_optimizer(self.warmup_optimizer)

    def _transition_phase(self, new_phase: str, *, reason: str, initial: bool = False) -> None:
        """Transition phase and apply lock/unlock/block rules for that phase."""
        if not initial and new_phase == self.phase:
            return
        old_phase = self.phase
        self.phase = new_phase
        if not initial:
            self.phase_history.append((self.epoch, new_phase))
            if self.logger:
                self.logger.info(
                    "Phase transition: %s -> %s (%s)", old_phase, new_phase, reason
                )
        if new_phase == "warmup":
            self._lock_for_warmup()
        else:
            if self._warmup_enabled:
                self._warmup_completed = True
            # Leaving warmup should release the warmup lock before roulette
            # switching resumes.
            if self.locked_optimizer and self.locked_optimizer == self._warmup_lock_name:
                self.unlock_optimizer()
            if self._warmup_completed and self.drop_after_warmup and self.warmup_optimizer:
                self.block_optimizer(self.warmup_optimizer)

    @staticmethod
    def _check_plateau(history: Iterable[float], threshold: float) -> bool:
        """Return True when metric trend slope magnitude is below threshold."""
        if threshold <= 0:
            return False
        history_list = list(history)
        if len(history_list) < 5:
            return False
        x = np.arange(len(history_list), dtype=np.float32)
        y = np.array(history_list, dtype=np.float32)
        slope = np.polyfit(x, y, 1)[0]
        return abs(float(slope)) < threshold

    def reset_epoch_stats(self) -> None:
        """Reset per-epoch selection counters and switch trace."""
        for name in self.selection_counts:
            self.selection_counts[name] = 0
        self.switch_history.clear()

    def get_epoch_selection_counts(self) -> Dict[str, int]:
        """Return a snapshot of current epoch optimizer selection counts."""
        return dict(self.selection_counts)

    def lock_optimizer(self, name: Optional[str]) -> None:
        """Force optimizer selection to a specific optimizer name."""
        if name is None:
            self.locked_optimizer = None
            return
        if name not in self.optimizers:
            return
        self.locked_optimizer = name
        self._switch_to(name, reason="lock")

    def unlock_optimizer(self) -> None:
        """Clear optimizer lock and re-enable roulette selection."""
        self.locked_optimizer = None

    def block_optimizer(self, name: str) -> None:
        """Exclude one optimizer from random candidate selection."""
        if name in self.optimizers:
            self.blocked_names.add(name)

    def unblock_optimizer(self, name: str) -> None:
        """Re-include one optimizer in random candidate selection."""
        self.blocked_names.discard(name)

    def on_epoch_start(self, epoch: int, *, phase: Optional[str] = None) -> None:
        """Epoch lifecycle hook for fixed warmup and epoch-level switching."""
        self.epoch = epoch
        if phase is not None:
            self.set_phase(phase)
        # Fixed warmup transition is evaluated at epoch boundaries.
        if self.phase == "warmup" and self._warmup_mode == "fixed":
            if self.warmup_epochs > 0 and epoch >= self.warmup_epochs:
                self._transition_phase("roulette", reason="warmup_epochs")
        self._apply_phase_state()
        if self.locked_optimizer:
            self._switch_to(self.locked_optimizer, reason="locked_epoch")
            return
        # Epoch-level roulette selection happens once per epoch.
        if self.switch_granularity == "epoch":
            self._maybe_switch(reason="epoch_start")

    def on_epoch_end(self, *, val_acc: Optional[float] = None) -> None:
        """Epoch-end hook for plateau warmup transition checks."""
        if not self._warmup_enabled or self.phase != "warmup":
            return
        # Plateau-driven warmup transition requires validation accuracy history.
        if self._warmup_mode != "plateau":
            return
        if val_acc is None:
            return
        val_acc = float(val_acc)
        self._warmup_val_acc_hist.append(val_acc)
        if self._warmup_min_epochs > 0 and self.epoch + 1 < self._warmup_min_epochs:
            return
        if self._warmup_min_val_acc > 0 and val_acc < self._warmup_min_val_acc:
            return
        if self._check_plateau(self._warmup_val_acc_hist, self._warmup_plateau_threshold):
            self._transition_phase("roulette", reason="warmup_plateau")

    def on_batch_start(self, batch_idx: int) -> None:
        """Batch lifecycle hook for batch-level switching and counters."""
        if self.locked_optimizer:
            self.selection_counts[self.current_name] += 1
            return
        # Batch-level roulette only runs when granularity is explicitly "batch".
        if self.switch_granularity == "batch":
            if (
                self.step_count % self.switch_every_steps == 0
                and self.random.random() <= self.switch_probability
            ):
                self._maybe_switch(reason="batch_start")
        self.selection_counts[self.current_name] += 1

    def _available_names(self) -> List[str]:
        """Return currently eligible optimizer names for random choice."""
        if self.locked_optimizer:
            return [self.locked_optimizer]
        if self.pool:
            self.active_names = self.pool.get_active_names()
        # Blocked optimizers are temporarily excluded from random choice.
        candidates = [n for n in self.active_names if n not in self.blocked_names]
        if not candidates:
            # Fallback to all known optimizers if active pool becomes empty.
            candidates = [n for n in self.optimizers.keys() if n not in self.blocked_names]
        return candidates

    def _choose_random(self) -> str:
        """Choose next optimizer name according to roulette constraints."""
        candidates = self._available_names()
        if len(candidates) == 1:
            return candidates[0]
        if self.avoid_repeat and self.current_name in candidates:
            candidates = [n for n in candidates if n != self.current_name] or candidates
        return self.random.choice(candidates)

    def _maybe_switch(self, *, reason: str) -> None:
        """Sample and apply a switch attempt with a reason tag."""
        target = self._choose_random()
        self._switch_to(target, reason=reason)

    def _switch_to(self, name: str, *, reason: str) -> None:
        """Switch active optimizer and keep wrapper state synchronized."""
        if name == self.current_name:
            return
        if name not in self.optimizers:
            return
        old_name = self.current_name
        self.current_name = name
        self.current_optimizer = self.optimizers[name]
        if self.compatibility:
            for group in self.current_optimizer.param_groups:
                base_lr = group.get("lr", 0.0)
                group["lr"] = self.compatibility.adjust_lr(old_name, name, base_lr)
        # Keep Optimizer interface attributes in sync with active optimizer so
        # external loops/checkpoint tools can treat this as a normal optimizer.
        self.param_groups = self.current_optimizer.param_groups
        self.state = self.current_optimizer.state
        self.defaults = getattr(self.current_optimizer, "defaults", {})
        self.switch_history.append((old_name, name, reason))
        if self.logger:
            self.logger.info("Optimizer switch: %s -> %s (%s)", old_name, name, reason)

    def update_performance(
        self,
        optimizer_name: str,
        reward: float,
        val_acc: float,
        best_val_acc: float,
    ) -> Optional[str]:
        """Forward reward/accuracy feedback to pool manager and apply swaps."""
        if not self.pool:
            return None
        swapped = self.pool.record_performance(
            optimizer_name,
            reward,
            val_acc,
            best_val_acc,
        )
        if swapped:
            self.active_names = self.pool.get_active_names()
            if optimizer_name in self.selection_counts:
                self.selection_counts.pop(optimizer_name, None)
            self.selection_counts.setdefault(swapped, 0)
            if self.current_name == optimizer_name:
                self._switch_to(swapped, reason="pool_swap")
        return swapped

    def step(self, closure=None):
        """Run one optimizer step on the currently active optimizer."""
        # `step()` delegates to the currently active optimizer; switching is
        # controlled by lifecycle hooks, not by step itself.
        result = self.current_optimizer.step(closure)
        self.step_count += 1
        return result

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients on the currently active optimizer."""
        self.current_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize both torch optimizer keys and roulette-specific state."""
        # Keep torch-optimizer compatible keys ("state", "param_groups") so
        # external trainers (e.g., Ultralytics checkpoint utilities) can
        # inspect/serialize optimizer state without KeyError.
        active_state = self.current_optimizer.state_dict()
        if not isinstance(active_state, dict):
            active_state = {}

        return {
            "state": active_state.get("state", {}),
            "param_groups": active_state.get("param_groups", []),
            "roulette_state": {
                "current_name": self.current_name,
                "active_names": list(self.active_names),
                "blocked_names": list(self.blocked_names),
                "locked_optimizer": self.locked_optimizer,
                "switch_granularity": self.switch_granularity,
                "switch_every_steps": self.switch_every_steps,
                "switch_probability": self.switch_probability,
                "avoid_repeat": self.avoid_repeat,
                "epoch": self.epoch,
                "step_count": self.step_count,
                "selection_counts": dict(self.selection_counts),
                "switch_history": list(self.switch_history),
                "random_state": self.random.getstate(),
                "phase": self.phase,
                "phase_history": list(self.phase_history),
                "warmup_mode": self._warmup_mode,
                "warmup_val_acc_hist": list(self._warmup_val_acc_hist),
                "warmup_completed": self._warmup_completed,
            },
            "optimizers": {
                name: optimizer.state_dict() for name, optimizer in self.optimizers.items()
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore wrapper and underlying optimizer states from checkpoint."""
        roulette_state = state_dict.get("roulette_state", {})
        # Restore individual optimizer internals first, then wrapper state.
        for name, opt_state in state_dict.get("optimizers", {}).items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(opt_state)

        self.current_name = roulette_state.get("current_name", self.current_name)
        if self.current_name in self.optimizers:
            self.current_optimizer = self.optimizers[self.current_name]
            self.param_groups = self.current_optimizer.param_groups
            self.state = self.current_optimizer.state
            self.defaults = getattr(self.current_optimizer, "defaults", {})
        self.active_names = roulette_state.get("active_names", self.active_names)
        self.blocked_names = set(roulette_state.get("blocked_names", []))
        self.locked_optimizer = roulette_state.get("locked_optimizer")
        self.switch_granularity = roulette_state.get(
            "switch_granularity", self.switch_granularity
        )
        self.switch_every_steps = roulette_state.get(
            "switch_every_steps", self.switch_every_steps
        )
        self.switch_probability = roulette_state.get(
            "switch_probability", self.switch_probability
        )
        self.avoid_repeat = roulette_state.get("avoid_repeat", self.avoid_repeat)
        self.epoch = roulette_state.get("epoch", self.epoch)
        self.step_count = roulette_state.get("step_count", self.step_count)
        self.selection_counts = roulette_state.get(
            "selection_counts", self.selection_counts
        )
        self.switch_history = roulette_state.get("switch_history", self.switch_history)
        random_state = roulette_state.get("random_state")
        if random_state is not None:
            self.random.setstate(random_state)
        self.phase = roulette_state.get("phase", self.phase)
        self.phase_history = roulette_state.get("phase_history", self.phase_history)
        warmup_hist = roulette_state.get("warmup_val_acc_hist")
        if warmup_hist is not None:
            self._warmup_val_acc_hist = deque(
                warmup_hist,
                maxlen=max(5, self._warmup_min_epochs),
            )
        self._warmup_mode = roulette_state.get("warmup_mode", self._warmup_mode)
        self._warmup_completed = roulette_state.get("warmup_completed", self._warmup_completed)
        self._apply_phase_state()


class OptiRoulette(OptiRouletteOptimizer):
    """Drop-in optimizer wrapper that builds underlying optimizers from specs."""

    def __init__(
        self,
        params: Iterable,
        *,
        optimizer_specs: Optional[Any] = None,
        optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
        roulette: Optional[Dict[str, Any]] = None,
        pool_config: Optional[Any] = None,
        lr_scaling_rules: Optional[Dict[str, Any]] = None,
        active_names: Optional[List[str]] = None,
        backup_names: Optional[List[str]] = None,
        switch_granularity: Optional[str] = None,
        switch_every_steps: Optional[int] = None,
        switch_probability: Optional[float] = None,
        avoid_repeat: Optional[bool] = None,
        warmup_optimizer: Optional[str] = None,
        drop_after_warmup: bool = False,
        warmup_epochs: int = 0,
        warmup_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        """Build optimizer instances from specs/defaults and initialize wrapper."""
        using_default_specs = optimizers is None and optimizer_specs is None

        default_roulette_cfg: Dict[str, Any] = get_default_roulette_config()
        roulette_cfg = dict(default_roulette_cfg)
        # Explicit `roulette=` values override defaults field-by-field.
        roulette_cfg.update(roulette or {})

        if switch_granularity is None:
            switch_granularity = roulette_cfg.get("switch_granularity", "epoch")
        if switch_every_steps is None:
            switch_every_steps = roulette_cfg.get("switch_every_steps", 1)
        if switch_probability is None:
            switch_probability = roulette_cfg.get("switch_probability", 1.0)
        if avoid_repeat is None:
            avoid_repeat = roulette_cfg.get("avoid_repeat", True)
        if warmup_optimizer is None:
            warmup_optimizer = roulette_cfg.get("warmup_optimizer")
        if not drop_after_warmup:
            drop_after_warmup = bool(roulette_cfg.get("drop_after_warmup", False))
        if warmup_epochs <= 0:
            warmup_epochs = int(roulette_cfg.get("warmup_epochs", warmup_epochs))
        if warmup_config is None:
            warmup_config = roulette_cfg.get("warmup_config")
        if seed is None:
            seed = get_default_seed()

        if optimizers is None:
            if not isinstance(params, (list, tuple)):
                # Materialize parameter iterator once so each optimizer receives
                # the same parameter references.
                params = list(params)
            specs = optimizer_specs
            if specs is None:
                specs = get_default_optimizer_specs()
            normalized = _normalize_optimizer_specs(specs)
            if not normalized:
                raise ValueError("OptiRoulette requires at least one optimizer spec.")
            if any(name.lower() in {"opti_roulette", "optiroulette", "roulette"} for name in normalized):
                raise ValueError("Nested OptiRoulette specs are not supported.")
            from .optimizer_factory import create_optimizer

            built: Dict[str, torch.optim.Optimizer] = {}
            for name, cfg in normalized.items():
                built[name] = create_optimizer(name, params, _normalize_optimizer_cfg(cfg))
            optimizers = built

        if lr_scaling_rules is None and using_default_specs:
            lr_scaling_rules = get_default_lr_scaling_rules()

        if pool_config is None and using_default_specs:
            # Default pool setup is only injected when user does not pass a
            # custom optimizer set/spec list.
            pool_cfg_dict, default_active, default_backup = get_default_pool_setup()
            pool_config = pool_cfg_dict
            if active_names is None:
                active_names = default_active
            if backup_names is None:
                backup_names = default_backup

        pool_cfg_obj = None
        if pool_config is not None:
            if isinstance(pool_config, PoolConfig):
                pool_cfg_obj = pool_config
            elif isinstance(pool_config, dict):
                pool_cfg_obj = PoolConfig(**pool_config)
            else:
                raise TypeError("pool_config must be a dict or PoolConfig instance.")

        super().__init__(
            optimizers,
            switch_granularity=switch_granularity,
            switch_every_steps=switch_every_steps,
            switch_probability=switch_probability,
            avoid_repeat=avoid_repeat,
            warmup_optimizer=warmup_optimizer,
            drop_after_warmup=drop_after_warmup,
            warmup_epochs=warmup_epochs,
            warmup_config=warmup_config,
            lr_scaling_rules=lr_scaling_rules,
            pool_config=pool_cfg_obj,
            active_names=active_names,
            backup_names=backup_names,
            seed=seed,
            logger=logger,
        )
