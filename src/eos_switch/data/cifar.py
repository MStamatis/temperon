"""GPU-resident CIFAR datasets with on-GPU augmentation.

CIFAR-10/100 fit comfortably in VRAM (~0.2 GB as uint8), so the full dataset
is uploaded once as a uint8 tensor and batches are sliced and augmented with
torch ops directly on the device. For small models the dataloader, not the
compute, is the usual bottleneck; this removes it entirely.

All randomness (shuffling, crop offsets, flips, cutout positions) goes
through an explicit ``torch.Generator`` living on the dataset device, so an
epoch's batch stream is fully reproducible given the seed.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import torchvision

_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "tinyimagenet": ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
}

_NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "tinyimagenet": 200}


def _random_crop(x: torch.Tensor, pad: int, gen: torch.Generator) -> torch.Tensor:
    """Per-sample random crop after zero padding (vectorized, no loops)."""
    b, c, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad))
    dy = torch.randint(0, 2 * pad + 1, (b,), device=x.device, generator=gen)
    dx = torch.randint(0, 2 * pad + 1, (b,), device=x.device, generator=gen)
    rows = dy[:, None] + torch.arange(h, device=x.device)[None, :]
    cols = dx[:, None] + torch.arange(w, device=x.device)[None, :]
    bi = torch.arange(b, device=x.device)[:, None, None, None]
    ci = torch.arange(c, device=x.device)[None, :, None, None]
    return xp[bi, ci, rows[:, None, :, None], cols[:, None, None, :]]


def _random_flip(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    mask = torch.rand(x.shape[0], device=x.device, generator=gen) < 0.5
    return torch.where(mask[:, None, None, None], x.flip(-1), x)


def _cutout(x: torch.Tensor, size: int, gen: torch.Generator) -> torch.Tensor:
    """Zero out a random (size x size) square per sample (applied post-norm)."""
    b, _, h, w = x.shape
    cy = torch.randint(0, h, (b,), device=x.device, generator=gen)
    cx = torch.randint(0, w, (b,), device=x.device, generator=gen)
    half = size // 2
    ys = torch.arange(h, device=x.device)[None, :]
    xs = torch.arange(w, device=x.device)[None, :]
    ymask = (ys >= cy[:, None] - half) & (ys < cy[:, None] + half)
    xmask = (xs >= cx[:, None] - half) & (xs < cx[:, None] + half)
    mask = ymask[:, None, :, None] & xmask[:, None, None, :]
    return x.masked_fill(mask, 0.0)


def _color_jitter(x01: torch.Tensor, b: float, c: float, s: float, gen: torch.Generator) -> torch.Tensor:
    """Per-sample brightness/contrast/saturation jitter on a [0,1] float image
    (matches torchvision ColorJitter strengths). Factors ~ U(1-strength, 1+strength)."""
    n = x01.shape[0]

    def fac(strength: float) -> torch.Tensor:
        return (1.0 - strength) + 2.0 * strength * torch.rand(
            n, 1, 1, 1, device=x01.device, generator=gen
        )

    if b > 0:
        x01 = x01 * fac(b)
    if c > 0:
        m = x01.mean(dim=(1, 2, 3), keepdim=True)
        x01 = (x01 - m) * fac(c) + m
    if s > 0:
        w = torch.tensor([0.299, 0.587, 0.114], device=x01.device).view(1, 3, 1, 1)
        gray = (x01 * w).sum(dim=1, keepdim=True)
        x01 = (x01 - gray) * fac(s) + gray
    return x01.clamp_(0.0, 1.0)


class GPUCifar:
    """CIFAR-10/100 held entirely on the target device as uint8 tensors."""

    def __init__(
        self,
        name: str,
        device: torch.device | str = "cpu",
        root: str | None = None,
        smoke_subset: int | None = None,
        val_split: float | None = None,
        val_seed: int = 42,
        color_jitter: float = 0.0,
    ) -> None:
        name = name.lower()
        if name not in _STATS:
            raise ValueError(f"unknown dataset {name!r}; expected cifar10/cifar100")
        self.name = name
        self.num_classes = _NUM_CLASSES[name]
        self.device = torch.device(device)
        self.color_jitter = float(color_jitter)
        root = root or os.environ.get("EOS_DATA_DIR", "./data")

        if name == "tinyimagenet":
            from eos_switch.data.tiny_imagenet import load_tiny_imagenet

            x_train, y_train, x_test, y_test = load_tiny_imagenet(root)
        else:
            cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
            train_ds = cls(root, train=True, download=True)
            test_ds = cls(root, train=False, download=True)
            x_train = torch.from_numpy(train_ds.data).permute(0, 3, 1, 2).contiguous()
            y_train = torch.tensor(train_ds.targets, dtype=torch.long)
            x_test = torch.from_numpy(test_ds.data).permute(0, 3, 1, 2).contiguous()
            y_test = torch.tensor(test_ds.targets, dtype=torch.long)

        if smoke_subset is not None:
            # Fixed permutation (seed 0) so every smoke run sees the same subset.
            idx = torch.randperm(len(x_train), generator=torch.Generator().manual_seed(0))
            idx = idx[:smoke_subset]
            x_train, y_train = x_train[idx], y_train[idx]

        # Optional held-out validation split from the train set (matches the
        # OptiRoulette framework: train on 90%, evaluate on the held-out 10%).
        x_val = y_val = None
        if val_split:
            g = torch.Generator().manual_seed(val_seed)
            perm = torch.randperm(len(x_train), generator=g)
            nv = int(len(x_train) * val_split)
            x_val, y_val = x_train[perm[:nv]], y_train[perm[:nv]]
            x_train, y_train = x_train[perm[nv:]], y_train[perm[nv:]]

        self.x_train = x_train.to(self.device)
        self.y_train = y_train.to(self.device)
        # Random-crop padding scales with the image size (4 px for 32x32 CIFAR,
        # 8 px for 64x64 Tiny ImageNet).
        self.crop_pad = self.x_train.shape[-1] // 8
        self.x_test = x_test.to(self.device)
        self.y_test = y_test.to(self.device)
        if x_val is not None:
            self.x_eval = x_val.to(self.device)
            self.y_eval = y_val.to(self.device)
            self.eval_name = f"val(held-out train {val_split:.0%})"
        else:
            self.x_eval, self.y_eval = self.x_test, self.y_test
            self.eval_name = "test"

        mean, std = _STATS[name]
        self._mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)

    def __len__(self) -> int:
        return len(self.x_train)

    def _standardize(self, x01: torch.Tensor) -> torch.Tensor:
        return (x01 - self._mean) / self._std

    def normalize(self, x_uint8: torch.Tensor) -> torch.Tensor:
        return self._standardize(x_uint8.float().div_(255.0))

    def train_batches(
        self,
        batch_size: int,
        generator: torch.Generator,
        augment: bool = True,
        cutout: int = 0,
        drop_last: bool = True,
    ):
        """Yield one epoch of shuffled, augmented (x, y) batches."""
        n = len(self.x_train)
        perm = torch.randperm(n, device=self.device, generator=generator)
        end = (n // batch_size) * batch_size if drop_last else n
        for start in range(0, end, batch_size):
            idx = perm[start : start + batch_size]
            x = self.x_train[idx]
            y = self.y_train[idx]
            if augment:
                x = _random_crop(x, pad=self.crop_pad, gen=generator)
                x = _random_flip(x, gen=generator)
            x01 = x.float().div_(255.0)
            if augment and self.color_jitter > 0:
                cj = self.color_jitter
                x01 = _color_jitter(x01, cj, cj, cj, generator)
            x = self._standardize(x01)
            if augment and cutout > 0:
                x = _cutout(x, cutout, gen=generator)
            yield x, y

    def eval_batches(self, batch_size: int = 1000):
        """Iterate the evaluation set: the held-out val split if configured,
        otherwise the official test set."""
        for start in range(0, len(self.x_eval), batch_size):
            x = self.normalize(self.x_eval[start : start + batch_size])
            y = self.y_eval[start : start + batch_size]
            yield x, y

    def sample_probe_batch(
        self, batch_size: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Random un-augmented training micro-batch for curvature probes."""
        idx = torch.randint(0, len(self.x_train), (batch_size,), device=self.device, generator=generator)
        return self.normalize(self.x_train[idx]), self.y_train[idx]
