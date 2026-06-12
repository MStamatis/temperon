from optiroulette import (
    get_default_config,
    get_default_optimizer_specs,
    get_default_pool_setup,
    get_default_roulette_config,
)


def test_default_sections_exist():
    cfg = get_default_config()
    assert "optimizers" in cfg
    assert "optimizer_pool" in cfg
    assert "roulette" in cfg


def test_pool_setup_consistent():
    pool_cfg, active, backup = get_default_pool_setup()
    assert len(active) == int(pool_cfg["num_active"])
    assert len(backup) >= int(pool_cfg["num_backup"])


def test_optimizer_specs_include_default_active_pool():
    specs = get_default_optimizer_specs()
    pool_cfg, active, _ = get_default_pool_setup()
    assert int(pool_cfg["num_active"]) > 0
    for name in active:
        assert name in specs


def test_roulette_defaults_have_switching():
    roulette_cfg = get_default_roulette_config()
    assert roulette_cfg.get("switch_granularity") in {"epoch", "batch"}
    assert float(roulette_cfg.get("switch_probability", 0.0)) > 0.0
