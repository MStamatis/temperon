"""Run logging: per-step JSONL, per-epoch CSV, summary CSV, nvidia-smi snapshot."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

MILESTONE_TARGETS = (0.50, 0.60, 0.65, 0.70)


class MilestoneTracker:
    """First-hit tracking of validation-accuracy targets (epoch + wall-clock)."""

    def __init__(self, targets: tuple[float, ...] = MILESTONE_TARGETS) -> None:
        self.targets = targets
        self.hits: dict[float, dict] = {}

    def update(self, epoch: int, wall_clock_s: float, val_acc: float) -> None:
        for t in self.targets:
            if t not in self.hits and val_acc >= t:
                self.hits[t] = {"epoch": epoch, "wall_clock_s": round(wall_clock_s, 2)}

    def as_flat_dict(self) -> dict:
        out: dict = {}
        for t in self.targets:
            hit = self.hits.get(t)
            out[f"hit_{t:.2f}_epoch"] = hit["epoch"] if hit else None
            out[f"hit_{t:.2f}_s"] = hit["wall_clock_s"] if hit else None
        return out


class RunLogger:
    """Writes steps.jsonl / probes.jsonl / epochs.csv / summary.csv / meta.json."""

    FLUSH_EVERY = 200

    def __init__(self, out_dir: str | Path) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._steps = open(self.dir / "steps.jsonl", "w", encoding="utf-8")
        self._probes = open(self.dir / "probes.jsonl", "w", encoding="utf-8")
        self._step_buf: list[str] = []
        self._epoch_rows: list[dict] = []

    def log_step(self, record: dict) -> None:
        self._step_buf.append(json.dumps(record))
        if len(self._step_buf) >= self.FLUSH_EVERY:
            self._flush_steps()

    def _flush_steps(self) -> None:
        if self._step_buf:
            self._steps.write("\n".join(self._step_buf) + "\n")
            self._step_buf.clear()

    def log_probe(self, record: dict) -> None:
        self._probes.write(json.dumps(record) + "\n")
        self._probes.flush()

    def log_epoch(self, record: dict) -> None:
        self._epoch_rows.append(record)
        msg = (
            f"epoch {record['epoch']:>3}  loss {record.get('train_loss', float('nan')):.4f}  "
            f"val_acc {record.get('val_acc', float('nan')):.4f}  "
            f"opt {record.get('optimizer', '?')}  {record.get('epoch_time_s', 0):.1f}s"
        )
        print(msg, flush=True)

    def snapshot_nvidia_smi(self) -> None:
        try:
            out = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=30
            ).stdout
            (self.dir / "nvidia_smi.txt").write_text(out, encoding="utf-8")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            (self.dir / "nvidia_smi.txt").write_text("nvidia-smi unavailable\n", encoding="utf-8")

    def write_meta(self, meta: dict) -> None:
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def finalize(self, summary: dict) -> None:
        import pandas as pd

        self._flush_steps()
        self._steps.close()
        self._probes.close()
        if self._epoch_rows:
            pd.DataFrame(self._epoch_rows).to_csv(self.dir / "epochs.csv", index=False)
        pd.DataFrame([summary]).to_csv(self.dir / "summary.csv", index=False)
