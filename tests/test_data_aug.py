"""Augmentation ops are tested standalone (no dataset download needed)."""

import torch

from eos_switch.data.cifar import _cutout, _random_crop, _random_flip


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_random_crop_shape_and_content():
    x = torch.arange(2 * 3 * 32 * 32, dtype=torch.uint8).reshape(2, 3, 32, 32)
    out = _random_crop(x, pad=4, gen=_gen())
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_random_crop_deterministic_given_seed():
    x = torch.randint(0, 255, (4, 3, 32, 32), dtype=torch.uint8)
    a = _random_crop(x, pad=4, gen=_gen(7))
    b = _random_crop(x, pad=4, gen=_gen(7))
    assert torch.equal(a, b)


def test_random_flip_preserves_pixels():
    x = torch.randint(0, 255, (8, 3, 32, 32), dtype=torch.uint8)
    out = _random_flip(x.float(), gen=_gen())
    # Each image is either identical or exactly mirrored.
    for i in range(8):
        same = torch.equal(out[i], x[i].float())
        flipped = torch.equal(out[i], x[i].float().flip(-1))
        assert same or flipped


def test_cutout_zeroes_a_square():
    x = torch.ones(4, 3, 32, 32)
    out = _cutout(x, size=8, gen=_gen())
    zeros_per_img = (out == 0).flatten(1).sum(dim=1)
    # Square may be clipped at the border: between 4*4 and 8*8 pixels x 3 ch.
    assert ((zeros_per_img >= 3 * 16) & (zeros_per_img <= 3 * 64)).all()
