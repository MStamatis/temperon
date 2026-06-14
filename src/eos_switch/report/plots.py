"""Matplotlib figures for the experiment report (Agg backend, PNG output)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_accuracy_curves(arm_epochs: dict[str, list[pd.DataFrame]], out_path: Path) -> None:
    """val_acc vs epoch (left) and vs wall-clock (right); mean across seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for arm, frames in sorted(arm_epochs.items()):
        merged = pd.concat(frames).groupby("epoch").agg(
            val_acc=("val_acc", "mean"), wall=("wall_clock_s", "mean")
        )
        axes[0].plot(merged.index, merged["val_acc"], label=arm)
        axes[1].plot(merged["wall"] / 60.0, merged["val_acc"], label=arm)
    axes[0].set_xlabel("epoch")
    axes[1].set_xlabel("wall-clock (min)")
    for ax in axes:
        ax.set_ylabel("val accuracy")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_train_loss_curves(arm_epochs: dict[str, list[pd.DataFrame]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm, frames in sorted(arm_epochs.items()):
        merged = pd.concat(frames).groupby("epoch")["train_loss"].mean()
        ax.plot(merged.index, merged.values, label=arm)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_sharpness_around_switches(arm_probe_frames: dict[str, pd.DataFrame], out_path: Path) -> None:
    """Batch sharpness and stability margin vs step offset from each switch.

    Uses probe records tagged pre_switch/post_switch; offset 0 = switch step.
    Tests the 'catapult' hypothesis: switches should knock sharpness down.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plotted = False
    for arm, df in sorted(arm_probe_frames.items()):
        tagged = df[df.get("tag").notna()] if "tag" in df.columns else df.iloc[0:0]
        if tagged.empty:
            continue
        tagged = tagged.assign(offset=tagged["step"] - tagged["switch_step"])
        grouped = tagged.groupby("offset").agg(
            bs=("batch_sharpness", "median"), margin=("stability_margin", "median")
        )
        axes[0].plot(grouped.index, grouped["bs"], marker="o", ms=3, label=arm)
        axes[1].plot(grouped.index, grouped["margin"], marker="o", ms=3, label=arm)
        plotted = True
    axes[0].set_ylabel("batch sharpness (median)")
    axes[0].set_yscale("symlog")
    axes[1].set_ylabel("stability margin (median)")
    for ax in axes:
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("steps from switch")
        ax.grid(alpha=0.3)
    if plotted:
        axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
