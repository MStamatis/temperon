import torch
import torch.nn as nn

from eos_switch.probes import lambda_max
from eos_switch.probes.precision import probe_precision


def test_tf32_flags_disabled_inside_and_restored():
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with probe_precision():
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def test_autocast_disabled_inside_even_when_nested():
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        with probe_precision():
            assert not torch.is_autocast_enabled("cpu")
            x = torch.randn(4, 4) @ torch.randn(4, 4)
            assert x.dtype == torch.float32
        assert torch.is_autocast_enabled("cpu")


def test_probe_restores_flags_on_exception():
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with probe_precision():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert torch.backends.cuda.matmul.allow_tf32 is True
    torch.backends.cuda.matmul.allow_tf32 = False


def test_bn_stats_preserved_during_probe():
    model = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8))
    model.train()
    # Warm the BN stats, snapshot, then probe.
    for _ in range(3):
        model(torch.randn(16, 8))
    bn = model[1]
    mean_before = bn.running_mean.clone()
    count_before = bn.num_batches_tracked.clone()

    batch = (torch.randn(16, 8), torch.randint(0, 8, (16,)))
    lambda_max(model, nn.functional.cross_entropy, batch, iters=5)

    assert torch.equal(bn.running_mean, mean_before)
    assert torch.equal(bn.num_batches_tracked, count_before)
