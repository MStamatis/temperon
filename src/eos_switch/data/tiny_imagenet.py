"""Tiny ImageNet loader -> uint8 CHW tensors (for the GPU-resident pipeline).

Tiny ImageNet: 200 classes, 64x64x3, 500 train images/class (100k train) + 10k
labelled validation images. The official *test* split is unlabelled, so the
standard protocol uses the val split as the held-out test set; we additionally
carve a small held-out val from train for milestone tracking (done by the
caller). Directory layout (tiny-imagenet-200/):

    wnids.txt                         200 class ids (wnids)
    train/<wnid>/images/*.JPEG        500 each
    val/images/*.JPEG                 10000
    val/val_annotations.txt           <file>\t<wnid>\t<bbox...>

Decoding 110k JPEGs is slow, so the decoded tensors are cached to a single .pt
file after the first load; subsequent runs (e.g. a multi-seed grid) load instantly.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile

import torch

_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _maybe_download(root: str) -> str:
    base = os.path.join(root, "tiny-imagenet-200")
    if os.path.isdir(os.path.join(base, "train")):
        return base
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "tiny-imagenet-200.zip")
    if not os.path.exists(zip_path):
        print(f"[tiny-imagenet] downloading {_URL} ...", flush=True)
        urllib.request.urlretrieve(_URL, zip_path)
    print("[tiny-imagenet] extracting ...", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    if not os.path.isdir(os.path.join(base, "train")):
        raise RuntimeError(f"tiny-imagenet not found/extracted under {base!r}")
    return base


def _decode(path: str) -> torch.Tensor:
    # read_image -> uint8 (C,H,W); force 3 channels (some images are grayscale).
    from torchvision.io import ImageReadMode, read_image

    img = read_image(path, mode=ImageReadMode.RGB)
    if img.shape[1:] != (64, 64):  # a few images are not exactly 64x64
        img = torch.nn.functional.interpolate(
            img[None].float(), size=(64, 64), mode="bilinear", align_corners=False
        )[0].round().clamp(0, 255).to(torch.uint8)
    return img


def load_tiny_imagenet(root: str):
    """Return (x_train, y_train, x_test, y_test) as uint8 (N,3,64,64) / long (N,).

    x_test/y_test is the labelled validation split (the standard Tiny ImageNet
    test protocol). Cached to <base>/cache.pt after the first decode.
    """
    base = _maybe_download(root)
    cache = os.path.join(base, "cache.pt")
    if os.path.exists(cache):
        d = torch.load(cache)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]

    with open(os.path.join(base, "wnids.txt")) as f:
        wnids = sorted(line.strip() for line in f if line.strip())
    cls_to_idx = {w: i for i, w in enumerate(wnids)}

    # --- train ---
    xtr, ytr = [], []
    for w in wnids:
        d = os.path.join(base, "train", w, "images")
        for fn in sorted(os.listdir(d)):
            xtr.append(_decode(os.path.join(d, fn)))
            ytr.append(cls_to_idx[w])
    # --- val (= test) ---
    xte, yte = [], []
    val_dir = os.path.join(base, "val")
    ann = {}
    with open(os.path.join(val_dir, "val_annotations.txt")) as f:
        for line in f:
            parts = line.split("\t")
            ann[parts[0]] = parts[1]
    for fn in sorted(ann):
        xte.append(_decode(os.path.join(val_dir, "images", fn)))
        yte.append(cls_to_idx[ann[fn]])

    xtr = torch.stack(xtr); ytr = torch.tensor(ytr, dtype=torch.long)
    xte = torch.stack(xte); yte = torch.tensor(yte, dtype=torch.long)
    torch.save({"xtr": xtr, "ytr": ytr, "xte": xte, "yte": yte}, cache)
    print(f"[tiny-imagenet] cached {len(xtr)} train / {len(xte)} test -> {cache}", flush=True)
    return xtr, ytr, xte, yte
