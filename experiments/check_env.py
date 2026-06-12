"""Verify the container sees the RTX 5090 correctly (Blackwell, sm_120).

Run: python experiments/check_env.py [--allow-cpu]

Fails loudly if CUDA is unavailable or a kernel-image error occurs (the
classic symptom of wheels built without sm_120 support).
"""

from __future__ import annotations

import argparse
import sys
import time

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu", action="store_true", help="do not fail when CUDA is missing")
    args = parser.parse_args()

    print(f"torch.__version__   : {torch.__version__}")
    print(f"torch.version.cuda  : {torch.version.cuda}")
    print(f"cudnn version       : {torch.backends.cudnn.version()}")

    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE")
        return 0 if args.allow_cpu else 1

    props = torch.cuda.get_device_properties(0)
    cap = f"{props.major}.{props.minor}"
    print(f"GPU                 : {props.name}")
    print(f"compute capability  : {cap}")
    print(f"total VRAM          : {props.total_memory / 2**30:.1f} GiB")

    try:
        a = torch.randn(4096, 4096, device="cuda")
        b = torch.randn(4096, 4096, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        c = a @ b
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        assert torch.isfinite(c).all()
        print(f"fp32 matmul 4096^2  : OK ({dt * 1e3:.1f} ms)")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            d = a @ b
        torch.cuda.synchronize()
        assert torch.isfinite(d.float()).all()
        print("bf16 autocast matmul: OK")
    except RuntimeError as exc:
        if "no kernel image" in str(exc):
            print(
                "FATAL: 'no kernel image is available' -- the installed torch "
                "wheels lack sm_120 support. Reinstall torch>=2.7 from "
                "https://download.pytorch.org/whl/cu128"
            )
        raise

    print("environment check: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
