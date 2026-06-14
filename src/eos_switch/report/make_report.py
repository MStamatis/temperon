"""Aggregate results/ into a markdown report.

    python -m eos_switch.report.make_report results/

Produces results/report.md plus PNG figures: first-hit milestone table per
arm (mean +/- std over seeds), accuracy/loss curves, sharpness-around-switch
analysis, measured probe overhead, and an explicit Limitations section.
Only quantities actually present in results/ are reported; everything else
is listed as "not yet run".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from eos_switch.report.logging import MILESTONE_TARGETS
from eos_switch.report.plots import (
    plot_accuracy_curves,
    plot_sharpness_around_switches,
    plot_train_loss_curves,
)

CANONICAL_ARMS = [
    "arm_A_paper_baseline",
    "arm_B_tuned_baseline",
    "arm_C_warmup_only",
    "arm_D_optiroulette",
    "arm_E_optiroulette_no_warmup",
    "arm_F_eos_switch",
    "arm_G_perlayer",
    "arm_H_mpc",
]


def _fmt(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def discover_runs(results_dir: Path) -> dict[str, list[Path]]:
    runs: dict[str, list[Path]] = {}
    for summary in sorted(results_dir.glob("*/seed*/summary.csv")):
        runs.setdefault(summary.parent.parent.name, []).append(summary.parent)
    return runs


def load_probes(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "probes.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    df = pd.DataFrame([r for r in records if r.get("type") == "probe"])
    return df


def main(results_dir: str) -> int:
    results = Path(results_dir)
    runs = discover_runs(results)
    if not runs:
        print(f"no runs found under {results}")
        return 1

    lines: list[str] = ["# eos-switch experiment report", ""]
    lines.append(f"Source: `{results}` — {sum(len(v) for v in runs.values())} runs, "
                 f"{len(runs)} arms/configs.")
    lines.append("")

    summaries: dict[str, pd.DataFrame] = {}
    arm_epochs: dict[str, list[pd.DataFrame]] = {}
    arm_probes: dict[str, pd.DataFrame] = {}
    for arm, run_dirs in runs.items():
        summaries[arm] = pd.concat(
            [pd.read_csv(d / "summary.csv") for d in run_dirs], ignore_index=True
        )
        frames = [pd.read_csv(d / "epochs.csv") for d in run_dirs if (d / "epochs.csv").exists()]
        if frames:
            arm_epochs[arm] = frames
        probe_frames = [load_probes(d) for d in run_dirs]
        probe_frames = [f for f in probe_frames if not f.empty]
        if probe_frames:
            arm_probes[arm] = pd.concat(probe_frames, ignore_index=True)

    # ---- headline table -------------------------------------------------
    lines.append("## Final / best validation accuracy (mean ± std over seeds)\n")
    lines.append("| arm | seeds | final val acc | best val acc | total time (s) | switches |")
    lines.append("|---|---|---|---|---|---|")
    for arm, df in sorted(summaries.items()):
        lines.append(
            f"| {arm} | {len(df)} | "
            f"{_fmt(df['final_val_acc'].mean(), df['final_val_acc'].std(ddof=0))} | "
            f"{_fmt(df['best_val_acc'].mean(), df['best_val_acc'].std(ddof=0))} | "
            f"{df['total_time_s'].mean():.1f} | {df['n_switches'].mean():.1f} |"
        )
    lines.append("")

    # ---- milestones ------------------------------------------------------
    lines.append("## First-hit milestones (epoch / wall-clock seconds)\n")
    header = "| arm |" + "".join(f" acc≥{t:.2f} |" for t in MILESTONE_TARGETS)
    lines.append(header)
    lines.append("|---|" + "---|" * len(MILESTONE_TARGETS))
    for arm, df in sorted(summaries.items()):
        cells = []
        for t in MILESTONE_TARGETS:
            ep_col, s_col = f"hit_{t:.2f}_epoch", f"hit_{t:.2f}_s"
            hit = df[df[ep_col].notna()] if ep_col in df.columns else df.iloc[0:0]
            if hit.empty:
                cells.append(" not reached |")
            else:
                cells.append(
                    f" ep {hit[ep_col].mean():.1f}±{hit[ep_col].std(ddof=0):.1f} / "
                    f"{hit[s_col].mean():.0f}s ({len(hit)}/{len(df)} seeds) |"
                )
        lines.append(f"| {arm} |" + "".join(cells))
    lines.append("")

    # ---- probe overhead --------------------------------------------------
    lines.append("## Probe overhead (fraction of total run wall-clock, target < 0.10)\n")
    for arm, df in sorted(summaries.items()):
        if "probe_overhead_frac" in df.columns and df["probe_overhead_frac"].max() > 0:
            lines.append(
                f"- {arm}: {_fmt(df['probe_overhead_frac'].mean(), df['probe_overhead_frac'].std(ddof=0))}"
            )
    lines.append("")

    # ---- figures -----------------------------------------------------------
    if arm_epochs:
        plot_accuracy_curves(arm_epochs, results / "fig_accuracy.png")
        plot_train_loss_curves(arm_epochs, results / "fig_train_loss.png")
        lines += ["## Curves\n", "![accuracy](fig_accuracy.png)", "", "![loss](fig_train_loss.png)", ""]
    if arm_probes:
        plot_sharpness_around_switches(arm_probes, results / "fig_sharpness_switch.png")
        lines += ["## Sharpness around switches (catapult check)\n",
                  "![sharpness](fig_sharpness_switch.png)", ""]

    # ---- limitations -------------------------------------------------------
    lines.append("## Limitations\n")
    missing = [a for a in CANONICAL_ARMS if a not in runs]
    if missing:
        lines.append(f"- Not yet run: {', '.join(missing)}.")
    for arm, df in sorted(summaries.items()):
        if len(df) < 3:
            lines.append(f"- {arm}: only {len(df)} seed(s); project standard is 3 (paper uses 10).")
    models = {arm: df["model"].iloc[0] for arm, df in summaries.items() if "model" in df.columns}
    if any(m != "resnet110" for m in models.values()):
        lines.append(
            "- Fast-track arms use ResNet-18 (CIFAR variant), NOT the paper's "
            "ResNet-110; only `paper_*` runs are architecture-comparable."
        )
    lines.append("- Adam-family stability threshold 38/lr is an empirical reference, "
                 "not a closed-form bound; spectral (Muon) threshold is a research "
                 "approximation with no literature value.")
    lines.append("")

    out = results / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results"))
