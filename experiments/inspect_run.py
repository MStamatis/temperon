"""Quick inspection of run outputs: summary, probe records, switch events.

    python experiments/inspect_run.py results/<run_name>/seed<seed> [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def inspect(run_dir: str) -> None:
    d = Path(run_dir)
    s = pd.read_csv(d / "summary.csv").iloc[0]
    probes = [json.loads(line) for line in (d / "probes.jsonl").read_text().splitlines()]
    records = [p for p in probes if p.get("type") == "probe"]
    switches = [p for p in probes if p.get("type") == "switch"]
    pre = [p for p in records if p.get("tag") == "pre_switch"]
    post = [p for p in records if p.get("tag") == "post_switch"]
    meta = json.loads((d / "meta.json").read_text())

    print(f"{d}:")
    print(
        f"  total {s.total_time_s:.1f}s  final_acc {s.final_val_acc:.3f}  "
        f"switches {s.n_switches}  probe_overhead {s.probe_overhead_frac:.3f}"
    )
    print(
        f"  probes: {len(records)} records (pre_switch {len(pre)}, "
        f"post_switch {len(post)}), switch markers {len(switches)}"
    )
    if records:
        p = records[-1]
        print(
            f"  last probe: opt={p['optimizer']} batch_sharp={p['batch_sharpness']:.2f} "
            f"precond={p['precond_sharpness']:.2f} thr={p['threshold']:.0f} "
            f"margin={p['stability_margin']:.3f} geom={p['geometry']} "
            f"fallback={p['fallback_raw']} t={p['probe_time_s']:.3f}s"
        )
    evs = [(e["from_name"], e["to_name"], e["step"]) for e in meta["switch_events"]]
    print(f"  switch_events: {evs}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        inspect(arg)
