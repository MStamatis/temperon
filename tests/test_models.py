import pytest
import torch

from eos_switch.train.models import build_model


@pytest.mark.parametrize("name,num_classes", [("smallcnn", 10), ("resnet18", 100), ("resnet110", 100)])
def test_forward_shape(name, num_classes):
    model = build_model(name, num_classes)
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, num_classes)


def test_resnet110_param_count():
    model = build_model("resnet110", 100)
    n = sum(p.numel() for p in model.parameters())
    # He et al. CIFAR ResNet-110 is ~1.73M params; allow slack for the
    # projection-shortcut variant and the 100-class head.
    assert 1.5e6 < n < 2.0e6, n


def test_resnet18_param_count():
    model = build_model("resnet18", 100)
    n = sum(p.numel() for p in model.parameters())
    assert 10e6 < n < 12e6, n


def test_invalid_depth_rejected():
    with pytest.raises(ValueError):
        build_model("resnet111", 10)
