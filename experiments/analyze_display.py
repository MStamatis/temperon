"""Does the display attachment recorded at run start explain the cost spread?

Every run snapshots nvidia-smi before training. That snapshot carries a
`Disp.A` column: whether the GPU was driving a monitor at the time. A GPU that
is compositing a desktop, decoding video and running a browser is not the same
GPU as an idle one, and this is the only per-run record we have of it.

If the flag separates the fast runs from the slow ones, then contamination is
retroactively detectable in every grid we have ever run -- no re-runs needed to
know WHICH numbers to trust.

    python experiments/analyze_display.py results
"""

from __future__ import annotations

import csv
import glob
import os
import re
import statistics as st
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"

SUBS = [
    ("c100", "phase6/arm_P_handoff"), ("c100", "phase6/arm_P_handoff_e80"),
    ("c100", "latesam/c100_latesam"), ("c100", "bench/c100/c100_sammuon"),
    ("c100", "bench/c100/c100_samsgd"), ("c100", "bench/c100/c100_strongsgd"),
    ("c100", "bench/c100/c100_muon"), ("c100", "bench/c100/c100_cyclicj"),
    ("tiny", "phase6/arm_P_tiny"), ("tiny", "bench/tiny/tiny_sammuon"),
    ("tiny", "bench/tiny/tiny_samsgd"), ("tiny", "bench/tiny/tiny_strongsgd"),
    ("c10", "phase6/arm_P_c10"), ("c10", "bench/c10/c10_sammuon"),
    ("c10", "bench/c10/c10_samsgd"), ("c10", "bench/c10/c10_strongsgd"),
    ("svhn", "phase6/arm_P_svhn"), ("svhn", "bench/svhn/svhn_sammuon"),
    ("svhn", "bench/svhn/svhn_samsgd"), ("svhn", "bench/svhn/svhn_strongsgd"),
]

# "| 0  NVIDIA GeForce RTX 5090  On | 00000000:02:00.0  On |" -- the SECOND
# On/Off on the bus-id line is Disp.A; the first is persistence mode.
BUS = re.compile(r"\|\s+00000000:[0-9A-Fa-f:.]+\s+(On|Off)\s+\|")


def display_flag(path):
    try:
        txt = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    m = BUS.search(txt)
    return m.group(1) if m else None


def main():
    rows = []
    for ds, sub in SUBS:
        pat = os.path.join(glob.escape(os.path.join(ROOT, *sub.split("/"))), "seed*")
        for d in sorted(glob.glob(pat)):
            f = os.path.join(d, "epochs.csv")
            smi = os.path.join(d, "nvidia_smi.txt")
            if not (os.path.exists(f) and os.path.exists(smi)):
                continue
            per = {}
            for r in csv.DictReader(open(f, newline="")):
                if int(r["epoch"]):
                    per.setdefault(r["optimizer"], []).append(float(r["epoch_time_s"]))
            flag = display_flag(smi)
            for opt, v in per.items():
                rows.append((ds, opt, flag, st.median(v), sub,
                             os.path.basename(d)))

    print(f"{'dataset':>7} {'optimizer':<18} | "
          f"{'display OFF':>22} | {'display ON':>22} | penalty")
    print("-" * 88)
    for ds in ["c100", "tiny", "c10", "svhn"]:
        opts = sorted({r[1] for r in rows if r[0] == ds})
        for opt in opts:
            off = [r[3] for r in rows if r[0] == ds and r[1] == opt and r[2] == "Off"]
            on = [r[3] for r in rows if r[0] == ds and r[1] == opt and r[2] == "On"]
            def cell(v):
                if not v:
                    return f"{'--':>22}"
                return f"n={len(v):2d} med {st.median(v):6.1f}s".rjust(22)
            pen = (f"{100*(st.median(on)/st.median(off)-1):+5.1f}%"
                   if off and on else "   --")
            print(f"{ds:>7} {opt:<18} | {cell(off)} | {cell(on)} | {pen}")

    unknown = [r for r in rows if r[2] is None]
    if unknown:
        print(f"\n!! {len(unknown)} runs had no parsable Disp.A field")


if __name__ == "__main__":
    main()
