"""Model zoo: CIFAR-style ResNets and a tiny CNN for smoke runs.

- ``resnet18``: the common CIFAR adaptation of ResNet-18 (3x3 stem, no
  maxpool, widths 64-512, blocks [2,2,2,2]); ~11M params.
- ``resnet110``: the original CIFAR ResNet of He et al. (2015): 3 stages of
  n=(depth-2)/6 BasicBlocks at widths 16/32/64; depth 110 -> n=18, ~1.7M
  params. This is the paper-comparable architecture.
- ``smallcnn``: 3-conv network used by --smoke runs and CPU tests.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, bn_momentum: float = 0.1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, momentum=bn_momentum),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class CifarResNet(nn.Module):
    """Generic CIFAR ResNet over a list of (width, num_blocks, stride) stages."""

    def __init__(self, stages: list[tuple[int, int, int]], stem_width: int, num_classes: int,
                 bn_momentum: float = 0.1, stem_stride: int = 1) -> None:
        super().__init__()
        # stem_stride=2 downsamples the input once (e.g. 64x64 Tiny ImageNet ->
        # 32x32) so the rest of the CIFAR ResNet runs at its native resolution.
        self.conv1 = _conv3x3(3, stem_width, stride=stem_stride)
        self.bn1 = nn.BatchNorm2d(stem_width, momentum=bn_momentum)
        layers: list[nn.Module] = []
        in_planes = stem_width
        for width, num_blocks, stride in stages:
            strides = [stride] + [1] * (num_blocks - 1)
            for s in strides:
                layers.append(BasicBlock(in_planes, width, s, bn_momentum=bn_momentum))
                in_planes = width
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(in_planes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.head(out)


class SmallCNN(nn.Module):
    """Tiny 3-conv net for --smoke runs and CPU unit tests."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.head(out)


def build_model(
    name: str,
    num_classes: int,
    initial_channels: int | None = None,
    bn_momentum: float = 0.1,
    stem_stride: int = 1,
) -> nn.Module:
    name = name.lower()
    if name == "smallcnn":
        return SmallCNN(num_classes)
    if name == "resnet18":
        stages = [(64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2)]
        return CifarResNet(stages, stem_width=64, num_classes=num_classes,
                           bn_momentum=bn_momentum, stem_stride=stem_stride)
    if name.startswith("resnet"):
        depth = int(name.removeprefix("resnet"))
        if (depth - 2) % 6 != 0:
            raise ValueError(f"CIFAR ResNet depth must satisfy depth=6n+2, got {depth}")
        n = (depth - 2) // 6
        # CIFAR ResNet stem width: 16 (He et al. standard) unless overridden.
        # The OptiRoulette framework uses initial_channels=64 (a much wider net).
        c = initial_channels or 16
        stages = [(c, n, 1), (2 * c, n, 2), (4 * c, n, 2)]
        return CifarResNet(stages, stem_width=c, num_classes=num_classes,
                           bn_momentum=bn_momentum, stem_stride=stem_stride)
    raise ValueError(f"unknown model {name!r}")
