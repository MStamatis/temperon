import torch
import pytest

from optiroulette import OptiRoulette


def test_custom_smoke_step():
    model = torch.nn.Linear(8, 2)
    optimizer = OptiRoulette(
        model.parameters(),
        optimizer_specs={
            "adam": {"lr": 1e-3},
            "sgd": {"lr": 1e-2, "momentum": 0.9},
        },
        switch_granularity="batch",
        switch_every_steps=1,
        switch_probability=1.0,
        avoid_repeat=True,
    )

    x = torch.randn(16, 8)
    y = torch.randint(0, 2, (16,))

    optimizer.on_epoch_start(0)
    optimizer.on_batch_start(0)
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()

    assert optimizer.step_count == 1
    assert optimizer.active_optimizer_name in {"adam", "sgd"}


def test_default_profile_instantiates():
    pytest.importorskip("pytorch_optimizer", reason="requires pytorch-optimizer package")
    model = torch.nn.Linear(4, 2)
    optimizer = OptiRoulette(model.parameters())
    assert optimizer.active_optimizer_name in optimizer.optimizers
