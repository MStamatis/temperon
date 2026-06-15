from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import logging
import torch

from .compatibility import OptimizerCompatibility


@dataclass
class OptimizerStats:
    """Statistics for an optimizer in the pool."""
    name: str
    total_epochs_used: int = 0
    consecutive_failures: int = 0
    catastrophic_failures: int = 0
    instability_events: int = 0
    average_reward: float = 0.0
    reward_history: deque = field(default_factory=lambda: deque(maxlen=20))
    last_val_acc: float = 0.0
    best_val_acc: float = 0.0
    swap_count: int = 0
    is_active: bool = True
    is_blacklisted: bool = False
    epochs_since_activation: int = 0
    in_grace_period: bool = True


@dataclass
class PoolConfig:
    """Configuration for the optimizer pool."""
    num_active: int = 5
    num_backup: int = 3
    enable_failure_swaps: bool = True
    failure_threshold: float = -0.3
    consecutive_failure_limit: int = 3
    catastrophic_drop: float = 0.2
    catastrophic_failure_limit: int = 1
    blacklist_threshold: int = 3
    swap_recovery: Dict[str, Any] = field(default_factory=dict)
    compatibility_groups: Dict[str, Any] = field(default_factory=dict)
    compatibility_rules: Dict[str, Any] = field(default_factory=dict)
    group_interactions: Dict[str, Any] = field(default_factory=dict)
    lr_scaling_rules: Dict[str, Any] = field(default_factory=dict)


class OptimizerPoolManager:
    """Maintain active and backup optimizers with hot-swapping."""

    def __init__(self,
                 all_optimizers: Dict[str, torch.optim.Optimizer],
                 active_names: List[str],
                 backup_names: List[str],
                 config: PoolConfig,
                 logger: Optional[logging.Logger] = None):
        """Initialize active/backup pools and per-optimizer tracking stats."""
        assert len(active_names) == config.num_active, "Active optimizer count mismatch"
        assert len(backup_names) >= config.num_backup, "Not enough backup optimizers"

        self.all_optimizers = all_optimizers
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.swap_cfg = config.swap_recovery
        self.compatibility = None
        if (
            config.compatibility_groups
            or config.compatibility_rules
            or config.group_interactions
            or config.lr_scaling_rules
        ):
            self.compatibility = OptimizerCompatibility({
                'groups': config.compatibility_groups,
                'rules': config.compatibility_rules,
                'group_interactions': config.group_interactions,
                'lr_scaling_rules': config.lr_scaling_rules,
            })

        # Pools:
        # - active_pool: candidates that can be selected now
        # - backup_pool: immediate replacements
        # - reserve_pool: extra fallbacks if backups are exhausted
        self.active_pool = active_names.copy()
        self.backup_pool = backup_names[:config.num_backup]
        self.reserve_pool = backup_names[config.num_backup:]

        self.stats: Dict[str, OptimizerStats] = {
            name: OptimizerStats(name=name, is_active=(name in self.active_pool))
            for name in all_optimizers
        }

    def get_active_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """Return optimizer objects that are currently active in the pool."""
        return {name: self.all_optimizers[name] for name in self.active_pool}

    def get_active_names(self) -> List[str]:
        """Return a copy of active optimizer names."""
        return self.active_pool.copy()

    def record_performance(
        self,
        name: str,
        reward: float,
        val_acc: float,
        global_best_val_acc: Optional[float] = None,
    ) -> Optional[str]:
        """Record an optimizer's performance and decide on potential swapping.

        Args:
            name: Name of the optimizer being evaluated.
            reward: Reward signal from training.
            val_acc: Validation accuracy achieved this epoch.
            global_best_val_acc: Best validation accuracy achieved by any optimizer
                so far. If provided, catastrophic drops are measured against this
                global best; otherwise the optimizer's own best validation accuracy
                is used.

        Returns:
            Name of the optimizer swapped in if a swap occurred, otherwise ``None``.
        """

        stats = self.stats[name]
        stats.total_epochs_used += 1
        stats.reward_history.append(reward)
        stats.average_reward = sum(stats.reward_history) / len(stats.reward_history)
        stats.last_val_acc = val_acc
        if stats.instability_events > 0:
            stats.instability_events = max(0, stats.instability_events - 1)

        # Grace period allows new optimizers to stabilize before strict failure
        # thresholds are applied.
        grace_epochs = self.swap_cfg.get('grace_period_epochs', 0)
        if stats.in_grace_period and stats.epochs_since_activation >= grace_epochs:
            stats.in_grace_period = False

        failure_threshold = self.config.failure_threshold
        if stats.in_grace_period:
            failure_threshold *= self.swap_cfg.get('relaxed_threshold_factor', 1.0)

        # Failure conditions based on reward
        if reward < failure_threshold:
            stats.consecutive_failures += 1
        else:
            stats.consecutive_failures = 0

        # Determine reference best accuracy for catastrophic drop detection
        reference_best = (
            global_best_val_acc if global_best_val_acc is not None else stats.best_val_acc
        )
        catastrophic = reference_best - val_acc > self.config.catastrophic_drop
        if catastrophic:
            stats.catastrophic_failures += 1
        else:
            stats.catastrophic_failures = 0

        # Update the optimizer's own best accuracy after computing catastrophic drop
        stats.best_val_acc = max(stats.best_val_acc, val_acc)

        # Keep statistics up to date even when swap actions are disabled.
        if not self.config.enable_failure_swaps:
            return None

        is_failure = stats.consecutive_failures >= self.config.consecutive_failure_limit
        catastrophic_limit = (
            stats.catastrophic_failures >= self.config.catastrophic_failure_limit
        )

        if (not stats.in_grace_period and is_failure) or catastrophic_limit:
            new_name = self._swap_optimizer(name)
            if new_name:
                if catastrophic_limit:
                    self.logger.info(
                        f"Catastrophic drop detected: val_acc fell from {reference_best:.4f} to {val_acc:.4f}. "
                        f"Dropping {name}, adding {new_name} to the game."
                    )
                    stats.catastrophic_failures = 0
                else:
                    self.logger.info(
                        f"Failure threshold triggered: reward {reward:.4f} < {self.config.failure_threshold:.4f} "
                        f"for {stats.consecutive_failures} epochs. Dropping {name}, adding {new_name} to the game."
                    )
            return new_name
        return None

    def increment_epochs(self) -> None:
        """Increment activation-age counters for active optimizers."""
        for name in self.active_pool:
            stats = self.stats[name]
            stats.epochs_since_activation += 1
            if stats.in_grace_period and stats.epochs_since_activation >= self.swap_cfg.get('grace_period_epochs', 0):
                stats.in_grace_period = False

    def _get_compatible_candidates(self, failed_name: str) -> List[str]:
        """Collect candidate optimizers not active and not blacklisted."""
        return [
            name
            for name in self.all_optimizers
            if name not in self.active_pool
            and name != failed_name
            and not self.stats[name].is_blacklisted
        ]

    def _swap_optimizer(self, failed_name: str) -> Optional[str]:
        """Swap one failed active optimizer with a compatible backup candidate."""
        # Prefer backups first. Reserve candidates are considered when
        # compatibility rules are enabled.
        backup_candidates = [
            name for name in self.backup_pool if not self.stats[name].is_blacklisted
        ]
        reserve_candidates = [
            name for name in self.reserve_pool if not self.stats[name].is_blacklisted
        ]

        candidates = backup_candidates
        if self.compatibility:
            candidates = backup_candidates + reserve_candidates

        if not candidates:
            if self.compatibility:
                candidates = self._get_compatible_candidates(failed_name)
            else:
                self.logger.warning("No backup optimizers available for swap")
                return None

        if not candidates:
            self.logger.warning("No compatible optimizers available for swap")
            return None

        new_name = (
            self.compatibility.get_compatible_backup(failed_name, candidates)
            if self.compatibility
            else candidates[0]
        )
        if not new_name:
            self.logger.warning("No compatible optimizers available for swap")
            return None

        # Remove promoted optimizer from non-active pools.
        if new_name in self.backup_pool:
            self.backup_pool.remove(new_name)
        if new_name in self.reserve_pool:
            self.reserve_pool.remove(new_name)

        # Atomic replacement in active set.
        self.active_pool.remove(failed_name)
        self.active_pool.append(new_name)

        failed_stats = self.stats[failed_name]
        failed_stats.is_active = False
        failed_stats.catastrophic_failures = 0
        failed_stats.instability_events = 0
        failed_stats.swap_count += 1
        # After repeated failures, failed optimizers are blacklisted to avoid
        # immediate reintroduction loops.
        if failed_stats.swap_count >= self.config.blacklist_threshold:
            failed_stats.is_blacklisted = True
        else:
            self.reserve_pool.append(failed_name)

        new_stats = self.stats[new_name]
        new_stats.is_active = True
        new_stats.consecutive_failures = 0
        new_stats.catastrophic_failures = 0
        new_stats.instability_events = 0
        new_stats.total_epochs_used = 0
        new_stats.epochs_since_activation = 0
        new_stats.in_grace_period = True

        self._apply_lr_scaling(failed_name, new_name)
        self._transfer_optimizer_state(failed_name, new_name)

        return new_name

    def _transfer_optimizer_state(self, old_name: str, new_name: str) -> None:
        """Optionally transfer momentum/EMA state from old optimizer to new one."""
        # Optional state transfer keeps momentum/EMA continuity across swaps,
        # but stays conservative to avoid corrupting third-party optimizer state.
        if not self.swap_cfg.get('transfer_momentum', False):
            return
        if self.compatibility and not self.compatibility.can_transfer_state(old_name, new_name):
            return
        mode = self.swap_cfg.get('transfer_mode', 'none')
        if mode == 'none':
            return
        old_opt = self.all_optimizers[old_name]
        new_opt = self.all_optimizers[new_name]


        old_state = old_opt.state_dict()
        new_state = new_opt.state_dict()
        # Some third-party optimizers expose non-standard state_dict formats.
        # In that case, skip transfer instead of crashing training.
        if not isinstance(old_state, dict) or not isinstance(new_state, dict):
            return
        if 'state' not in new_state or 'param_groups' not in new_state:
            if self.logger:
                self.logger.warning(
                    "Skipping state transfer %s -> %s: target optimizer returned non-standard state_dict keys=%s",
                    old_name,
                    new_name,
                    list(new_state.keys()),
                )
            return
        new_state_dict = new_state.get('state', {})
        for param_id, state in old_state.get('state', {}).items():
            # Only transfer to parameters that already have state to avoid
            # creating incomplete entries for optimizers with specialized
            # requirements (e.g., Novograd's "grads_ema"/"moments").
            if param_id not in new_state_dict:
                continue
            new_param_state = new_state_dict[param_id]
            if 'momentum_buffer' in state:
                scale = self.swap_cfg.get('momentum_scaling', 1.0)
                new_param_state['momentum_buffer'] = state['momentum_buffer'] * scale
            if mode in ('direct', 'adaptive'):
                if 'exp_avg' in state:
                    new_param_state['exp_avg'] = state['exp_avg']
                if 'exp_avg_sq' in state:
                    new_param_state['exp_avg_sq'] = state['exp_avg_sq']
        try:
            new_opt.load_state_dict(new_state)
        except Exception as exc:  # noqa: BLE001
            if self.logger:
                self.logger.warning(
                    "State transfer failed for %s -> %s (%s). Continuing without transfer.",
                    old_name,
                    new_name,
                    exc,
                )

    def _apply_lr_scaling(self, old_name: str, new_name: str) -> None:
        """Apply compatibility LR scaling when an optimizer swap occurs."""
        if not self.compatibility:
            return
        new_opt = self.all_optimizers[new_name]
        for group in new_opt.param_groups:
            base_lr = group.get('lr', 0.0)
            group['lr'] = self.compatibility.adjust_lr(old_name, new_name, base_lr)

    def report_instability(self, name: str, reason: str) -> None:
        """React to catastrophic optimizer behaviour (NaNs, explosions)."""
        optimizer = self.all_optimizers.get(name)
        stats = self.stats.get(name)
        if optimizer is None or stats is None:
            return

        stats.instability_events += 1
        lr_factor = self.swap_cfg.get('instability_lr_factor', 0.5)
        min_lr = self.swap_cfg.get('instability_min_lr', 1e-7)
        self._scale_optimizer_lr(name, lr_factor, min_lr)

        threshold = self.swap_cfg.get('instability_swap_threshold', 2)
        if stats.instability_events >= threshold:
            stats.consecutive_failures = max(
                stats.consecutive_failures,
                self.config.consecutive_failure_limit,
            )
            stats.catastrophic_failures = max(
                stats.catastrophic_failures,
                self.config.catastrophic_failure_limit,
            )

        self.logger.warning(
            "Optimizer %s reported instability (%s). "
            "Scaled lr by %.2f and marked failure count=%d/%d.",
            name,
            reason,
            lr_factor,
            stats.instability_events,
            threshold,
        )

    def _scale_optimizer_lr(self, name: str, factor: float, min_lr: float) -> None:
        """Scale an optimizer learning rate in-place with a lower bound."""
        optimizer = self.all_optimizers.get(name)
        if optimizer is None or factor <= 0:
            return
        for group in optimizer.param_groups:
            old_lr = group.get('lr', 0.0)
            new_lr = max(old_lr * factor, min_lr)
            group['lr'] = new_lr
