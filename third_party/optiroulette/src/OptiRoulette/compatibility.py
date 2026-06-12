from typing import Dict, List, Any, Optional


class OptimizerCompatibility:
    """Advanced compatibility manager for optimizer swaps and LR scaling."""

    def __init__(self, compatibility_config: Dict[str, Any]):
        self.groups = compatibility_config.get('groups', {})
        self.rules = compatibility_config.get('rules', {})
        gi_cfg = compatibility_config.get('group_interactions', {})
        self.group_interactions_enabled = gi_cfg.get('enabled', False)
        self.group_interactions = {k: v for k, v in gi_cfg.items() if k != 'enabled'}
        self.lr_scaling = compatibility_config.get('lr_scaling_rules', {})
        self.opt_to_group: Dict[str, str] = {}
        for group_name, group in self.groups.items():
            for opt in group.get('members', []):
                self.opt_to_group[opt] = group_name

    # ---------------------- Group helpers ----------------------
    def get_group_name(self, optimizer: str) -> Optional[str]:
        return self.opt_to_group.get(optimizer)

    def get_group(self, optimizer: str) -> Optional[Dict[str, Any]]:
        name = self.get_group_name(optimizer)
        if name is None:
            return None
        return {'name': name, **self.groups.get(name, {})}

    def _name_or_group_in(self, name: str, values: List[str]) -> bool:
        return name in values or self.get_group_name(name) in values

    def are_groups_compatible(self, group_a: str, group_b: str) -> bool:
        if group_a == group_b:
            return True
        if not self.group_interactions_enabled:
            return self.rules.get('allow_cross_group_swap', False)
        info_a = self.group_interactions.get(group_a, {})
        info_b = self.group_interactions.get(group_b, {})
        if group_b in info_a.get('incompatible_with', []):
            return False
        if group_a in info_b.get('incompatible_with', []):
            return False
        if group_b in info_a.get('compatible_with', []):
            return True
        if group_a in info_b.get('compatible_with', []):
            return True
        return self.rules.get('allow_cross_group_swap', False)

    def _is_swap_allowed(self, old_opt: str, new_opt: str) -> bool:
        old_group = self.get_group_name(old_opt)
        new_group = self.get_group_name(new_opt)
        if old_group is None or new_group is None:
            return True
        if old_group == new_group:
            return True
        if self.groups.get(old_group, {}).get('internal_swap_only', False):
            return False
        if self.groups.get(new_group, {}).get('internal_swap_only', False):
            return False
        return self.are_groups_compatible(old_group, new_group)

    # ------------------- Backup selection -------------------
    def get_compatible_backup(self, failed_optimizer: str, available_backups: List[str]) -> Optional[str]:
        if not available_backups:
            return None

        prefs = self.rules.get('preferred_swaps', {}).get(failed_optimizer, {})
        avoid = prefs.get('avoid', [])
        failed_group = self.get_group_name(failed_optimizer)
        group_cfg = self.groups.get(failed_group, {}) if failed_group else {}
        group_enabled = group_cfg.get('enabled', False)
        prefer_same = self.rules.get('prefer_same_group', False) and group_enabled

        def is_avoided(c: str) -> bool:
            return c in avoid or self.get_group_name(c) in avoid

        def valid(c: str) -> bool:
            return (
                c in available_backups
                and not is_avoided(c)
                and self._is_swap_allowed(failed_optimizer, c)
            )

        # When group-based selection is enabled, prioritize members of that group
        if group_enabled and failed_group:
            group_members = [m for m in group_cfg.get('members', []) if m != failed_optimizer]
            for cand in group_members:
                if valid(cand):
                    return cand

        # Same-group preference when explicitly allowed and group is enabled
        if prefer_same and failed_group:
            same_group = [b for b in available_backups if self.get_group_name(b) == failed_group]
            for cand in same_group:
                if valid(cand):
                    return cand

        # Preferred swaps in tiers (used when group not enabled or no valid member found)
        for tier in ['first_choice', 'second_choice', 'third_choice']:
            for pref in prefs.get(tier, []):
                if pref in available_backups:
                    if valid(pref):
                        return pref
                else:
                    group_cands = [b for b in available_backups if self.get_group_name(b) == pref]
                    for cand in group_cands:
                        if valid(cand):
                            return cand

        # Fallback: choose by group swap priority
        sorted_backups = sorted(
            available_backups,
            key=lambda opt: self.groups.get(self.get_group_name(opt), {}).get('swap_priority', 999),
        )
        for cand in sorted_backups:
            if valid(cand):
                return cand
        return None

    # -------------------- State transfer --------------------
    def can_transfer_state(self, old_name: str, new_name: str) -> bool:
        matrix = self.rules.get('state_transfer_compatible', {})
        return new_name in matrix.get(old_name, [])

    # -------------------- LR scaling --------------------
    def _matches_lr_rule(self, old_name: str, new_name: str, rule_cfg: Dict[str, Any]) -> bool:
        for case in rule_cfg.get('applies_to', []):
            if (
                self._name_or_group_in(old_name, case.get('from', []))
                and self._name_or_group_in(new_name, case.get('to', []))
            ):
                return True
        return False

    def adjust_lr(self, old_name: str, new_name: str, current_lr: float) -> float:
        if not self.lr_scaling:
            return current_lr

        special = self.lr_scaling.get('special_cases', {})
        case = special.get(new_name)
        new_lr = current_lr
        no_scaling = False

        if case:
            if 'override_lr' in case:
                new_lr = case['override_lr']
            no_scaling = case.get('no_scaling', False)

        if not no_scaling:
            high_to_low = self.lr_scaling.get('from_high_to_low', {})
            low_to_high = self.lr_scaling.get('from_low_to_high', {})
            if self._matches_lr_rule(old_name, new_name, high_to_low):
                new_lr *= high_to_low.get('scale_factor', 1.0)
            elif self._matches_lr_rule(old_name, new_name, low_to_high):
                new_lr *= low_to_high.get('scale_factor', 1.0)

        if case and case.get('force_clip') and 'max_lr' in case:
            new_lr = min(new_lr, case['max_lr'])

        return new_lr

