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
    if kind == "perlayer":
        from eos_switch.controllers.per_layer import PerLayerController  # Phase 4

        return PerLayerController(cfg)
    if kind == "edge_lr":
        from eos_switch.controllers.edge_lr import EdgeLRController  # arm H

        return EdgeLRController(cfg)
    if kind == "adam_sgd_hybrid":
        from eos_switch.controllers.hybrid import AdamSgdHybridController  # arm I

        return AdamSgdHybridController(cfg)
    if kind == "cyclic_catapult":
        from eos_switch.controllers.cyclic import CyclicCatapultController  # arm J

        return CyclicCatapultController(cfg)
    if kind == "eos_restart":
        from eos_switch.controllers.eos_restart import EosRestartController  # arm K

        return EosRestartController(cfg)
    if kind == "sam_catapult":
        from eos_switch.controllers.sam_catapult import SamCatapultController  # arm L

        return SamCatapultController(cfg)
    if kind == "sharp_muon_catapult":
        from eos_switch.controllers.sharp_muon import SharpMuonCatapultController  # arm M

        return SharpMuonCatapultController(cfg)
    if kind == "sam_eos_catapult":
        from eos_switch.controllers.sam_eos_catapult import SamEosCatapultController  # arm N

        return SamEosCatapultController(cfg)
    raise ValueError(f"unknown controller type {cfg['type']!r}")


__all__ = ["Controller", "SwitchEvent", "build_controller"]
