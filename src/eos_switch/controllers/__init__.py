from eos_switch.controllers.base import Controller, SwitchEvent


def build_controller(cfg: dict) -> Controller:
    """Factory: controller config dict -> Controller instance."""
    kind = cfg["type"].lower()
    if kind == "fixed":
        from eos_switch.controllers.fixed import FixedController

        return FixedController(cfg)
    if kind == "sequential":
        from eos_switch.controllers.fixed import SequentialController

        return SequentialController(cfg)
    if kind == "optiroulette":
        from eos_switch.controllers.roulette import OptiRouletteController

        return OptiRouletteController(cfg)
    if kind == "eos_switch":
        from eos_switch.controllers.eos_switch import EosSwitchController  # Phase 3

        return EosSwitchController(cfg)
    raise ValueError(f"unknown controller type {cfg['type']!r}")


__all__ = ["Controller", "SwitchEvent", "build_controller"]
