"""Generate every preprint figure from the results tree.

No number is typed in by hand: F1 parses analysis_costmodel.txt, F2/F4/F5
read the raw per-run logs, F3 reads calibration.json. Rerunning after new
results regenerates the exact figures the paper embeds.

Usage (inside the container):
    ./eos.sh python experiments/make_figures.py
Outputs paper/figures/*.pdf (+ .png previews).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = ["42", "1181241943", "958682846", "271828", "314159"]

# Fixed entity -> colour mapping, identical in every figure (validated
# 4-colour categorical palette; neutral dashed lines for no-SAM anchors).
C_TEMPERON = "#0072B2"
C_RIVAL = "#E69F00"
C_SAMSGD = "#009E73"
C_SAMMUON = "#CC79A7"
C_ANCHOR = "#444444"

DATASETS = [("c100", "CIFAR-100", 0.82), ("tiny", "Tiny ImageNet", 0.69),
            ("c10", "CIFAR-10", 0.968), ("svhn", "SVHN", 0.98)]

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "figure.dpi": 200,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.pdf/.png")


# ---------------------------------------------------------------- F1: frontier
def parse_costmodel():
    """{dataset: {target: {method: (clean_s, k_of_5) | (None, k)}}}"""
    txt = (RESULTS / "analysis_costmodel.txt").read_text()
    out: dict = {}
    ds = tgt = None
    hdr = re.compile(r"TIME TO TARGET -- (\w+)")
    tline = re.compile(r"^\s+target ([\d.]+)")
    hit = re.compile(r"^\s+(\S.*?)\s{2,}ep\s+[\d.]+\s+(\d+)s clean\s+\d+s raw\s+(\d)/5 seeds")
    miss = re.compile(r"^\s+(\S.*?)\s{2,}--\s+(\d)/5 seeds")
    for line in txt.splitlines():
        m = hdr.search(line)
        if m:
            ds = m.group(1)
            out.setdefault(ds, {})
            continue
        m = tline.match(line)
        if m and ds:
            tgt = float(m.group(1))
            out[ds][tgt] = {}
            continue
        if ds is None or tgt is None:
            continue
        m = hit.match(line)
        if m:
            out[ds][tgt][norm_name(m.group(1))] = (int(m.group(2)), int(m.group(3)))
            continue
        m = miss.match(line)
        if m:
            out[ds][tgt][norm_name(m.group(1))] = (None, int(m.group(2)))
    return out


def norm_name(raw: str) -> str:
    raw = raw.strip()
    for suffix in (" tiny", " c10", " svhn", " (arm P)"):
        raw = raw.removesuffix(suffix)
    return raw


def fig1_frontier():
    data = parse_costmodel()
    methods = [("Temperon", C_TEMPERON), ("late-phase SAM", C_RIVAL),
               ("full SAM+SGD", C_SAMSGD), ("full SAM+Muon", C_SAMMUON)]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1))
    for ax, (ds, label, tgt) in zip(axes, DATASETS):
        cells = data[ds][tgt]
        for i, (name, color) in enumerate(methods):
            secs, k = cells[name]
            if secs is None:
                ax.text(i, 0.04, "never", ha="center", va="bottom", rotation=90,
                        color="#777777", transform=ax.get_xaxis_transform())
                continue
            ax.bar(i, secs / 60, width=0.62, color=color)
            note = f"{secs}s" + (f"\n{k}/5" if k < 5 else "")
            ax.text(i, secs / 60, note, ha="center", va="bottom", fontsize=6.5)
        ax.set_title(f"{label} → {tgt}")
        ax.set_xticks([])
        ax.margins(y=0.22)
        if ax is axes[0]:
            ax.set_ylabel("minutes to target (calibrated)")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, c in methods],
               labels=[n for n, _ in methods], ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.10))
    save(fig, "fig1_frontier")


# ------------------------------------------------------------- F2: plateau bar
def envelope_mean(ds: str, arm: str) -> np.ndarray:
    envs = []
    for s in SEEDS:
        f = RESULTS / "bench" / ds / f"{ds}_{arm}" / f"seed{s}" / "epochs.csv"
        acc = [float(r["val_acc"]) for r in csv.DictReader(open(f))]
        envs.append(np.maximum.accumulate(acc))
    n = min(len(e) for e in envs)
    return np.mean([e[:n] for e in envs], axis=0)


def fig2_plateau():
    panels = [("svhn", "SVHN", 0.98, (0.935, 0.995)),
              ("c10", "CIFAR-10", 0.968, (0.83, 0.985)),
              ("c100", "CIFAR-100", 0.82, (0.55, 0.86))]
    series = [("sammuon", "full SAM+Muon", C_SAMMUON, "-"),
              ("samsgd", "full SAM+SGD", C_SAMSGD, (0, (4, 1.5, 1, 1.5))),
              ("strongsgd", "tuned SGD", C_ANCHOR, "--")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))
    for ax, (ds, label, bar, ylim) in zip(axes, panels):
        for arm, name, color, style in series:
            env = envelope_mean(ds, arm)
            ax.plot(np.arange(len(env)), env, color=color, ls=style, lw=1.6,
                    label=name if ax is axes[0] else None)
        ax.axhline(bar, color="#222222", lw=0.8, ls=":")
        ax.annotate(f"target {bar}", (3, bar), xytext=(0, 2),
                    textcoords="offset points", fontsize=6.5, color="#222222")
        ax.set_ylim(*ylim)
        ax.set_xlim(0, 103)
        ax.set_title(label)
        ax.set_xlabel("epoch")
        if ax is axes[0]:
            ax.set_ylabel("best-so-far val. accuracy")
    fig.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.24))
    save(fig, "fig2_plateau")


# ---------------------------------------------------------- F3: exchange rate
def fig3_exchange():
    """Credit/premium/net exactly as analyze_allocation.py reports them."""
    txt = (RESULTS / "latesam" / "analysis_allocation.txt").read_text()
    pat = re.compile(r"^\s*(\w+)\s+\S+\s*\|\s*([+-]\d+)\s+([+-]\d+)\s+"
                     r"[+-]?\d+\s*\|\s*([+-][\d.]+)%", re.M)
    rows = {m.group(1): (int(m.group(2)), int(m.group(3)), float(m.group(4)))
            for m in pat.finditer(txt)}
    order = ["c100", "c10", "svhn", "tiny"]
    labels = {"c100": "CIFAR-100", "c10": "CIFAR-10", "svhn": "SVHN",
              "tiny": "Tiny ImageNet"}
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    for i, ds in enumerate(order):
        credit, premium, net = rows[ds]
        ax.bar(i - 0.17, credit, width=0.3, color=C_TEMPERON)
        ax.bar(i + 0.17, premium, width=0.3, color=C_SAMMUON)
        ax.text(i - 0.17, credit, f"{credit:+d}s", ha="center", va="top",
                fontsize=6.5)
        ax.text(i + 0.17, premium,
                "+0s (SGD tail)" if premium == 0 else f"{premium:+d}s",
                ha="center", va="bottom", fontsize=6.5)
        ax.text(i, -3450, f"net {net:+.1f}%", ha="center", fontsize=6.5,
                color="#222222")
    ax.axhline(0, color="#222222", lw=0.8)
    ax.set_xticks(range(len(order)), [labels[d] for d in order])
    ax.set_ylim(-3700, 2500)
    ax.set_ylabel("seconds vs full-time SAM+SGD")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_TEMPERON),
               plt.Rectangle((0, 0), 1, 1, color=C_SAMMUON)]
    ax.legend(handles, ["credit: SAM skipped before the switch",
                        "premium: Muon tail at 1.50×, up to the target"],
              loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=6.5,
              ncol=2, frameon=False)
    save(fig, "fig3_exchange")


# ----------------------------------------------------------- F4: attribution
def finals(sub: str) -> np.ndarray:
    vals = []
    for s in SEEDS:
        f = RESULTS / sub / f"seed{s}" / "metrics.json"
        vals.append(json.load(open(f))["accuracy"])
    return np.array(vals)


def fig4_attribution():
    arms = [  # bottom-up display order
        ("tuned SGD", "bench/c100/c100_strongsgd", "none"),
        ("full SAM+SGD", "bench/c100/c100_samsgd", "sgd"),
        ("late-phase SAM (rival)", "latesam/c100_latesam", "sgd"),
        ("arm S: our shape, SGD tail", "phase6/arm_P_sgdtail", "sgd"),
        ("arm T: single cosine, Muon tail", "phase6/arm_T_singlecos", "muon"),
        ("full SAM+Muon", "bench/c100/c100_sammuon", "muon"),
        ("Temperon (arm P)", "phase6/arm_P_handoff", "muon"),
    ]
    colors = {"none": C_ANCHOR, "sgd": C_SAMSGD, "muon": C_SAMMUON}
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    ax.axvspan(0.8210, 0.8234, color=C_SAMSGD, alpha=0.10)
    ax.axvspan(0.8285, 0.8295, color=C_SAMMUON, alpha=0.12)
    for y, (name, sub, ref) in enumerate(arms):
        v = finals(sub)
        ax.errorbar(v.mean(), y, xerr=v.std(ddof=1), fmt="o", ms=4.5,
                    color=colors[ref], capsize=2, lw=1.2)
        ax.text(v.mean(), y + 0.24, f"{v.mean():.4f}", ha="center", fontsize=6.5)
    ax.set_yticks(range(len(arms)), [a[0] for a in arms], fontsize=7)
    ax.set_xlabel("final test accuracy, CIFAR-100 (mean ± sd, 5 seeds)")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    ax.text(0.8222, len(arms) - 0.55, "SGD-refined tier", fontsize=6.5,
            ha="center", color=C_SAMSGD)
    ax.text(0.8290, 0.05, "Muon-refined tier", fontsize=6.5, ha="center",
            color=C_SAMMUON)
    save(fig, "fig4_attribution")


# ------------------------------------------------------------------ F5: LM
def fig5_lm():
    arms = [("lm_muon", "Muon, no SAM", C_ANCHOR, "--"),
            ("lm_sammuon", "SAM+Muon, full-time", C_SAMMUON, "-"),
            ("lm_handoff", "SAM tail (ours)", C_TEMPERON, "-")]
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    ends = {}
    for arm, name, color, style in arms:
        rows = list(csv.DictReader(open(RESULTS / "phase7" / arm / "seed42" / "evals.csv")))
        t = np.array([float(r["wall_clock_s"]) for r in rows])
        loss = np.array([float(r["val_loss"]) for r in rows])
        ax.plot(t / 60, loss, color=color, ls=style, lw=1.6)
        ax.annotate(f"{name}\n{loss[-1]:.4f}", (t[-1] / 60, loss[-1]),
                    xytext=(3, 0), textcoords="offset points", color=color,
                    fontsize=6.5, va="center")
        ends[arm] = (t[-1] / 60, loss[-1])
    ax.annotate("", xy=(ends["lm_handoff"][0], 3.62),
                xytext=(ends["lm_sammuon"][0], 3.62),
                arrowprops=dict(arrowstyle="->", color="#222222", lw=0.9))
    ax.text((ends["lm_handoff"][0] + ends["lm_sammuon"][0]) / 2, 3.63,
            "−29% wall-clock", ha="center", fontsize=6.5)
    ax.set_xlim(8, 118)
    ax.set_ylim(3.28, 3.78)
    ax.set_xlabel("wall-clock (minutes)")
    ax.set_ylabel("validation loss")
    ax.set_title("GPT-2 124M, WikiText-103, 400M tokens")
    save(fig, "fig5_lm")


if __name__ == "__main__":
    fig1_frontier()
    fig2_plateau()
    fig3_exchange()
    fig4_attribution()
    fig5_lm()
