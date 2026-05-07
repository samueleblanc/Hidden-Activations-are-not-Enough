"""Generate paper-specific matplotlib figures.

Outputs to docs/Final-twist/paper/figures/:
  - teleportation_drift.pdf       — Study 1b per-arch drift box-plot
  - amplification_per_attack.pdf  — Study 2 d_M/d_f & d_h/d_f per attack/arch

Run:  python scripts/paper_figures.py
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("docs/Final-twist/paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARCHS = [("resnet152", "ResNet-152"),
         ("densenet121", "DenseNet-121"),
         ("googlenet", "GoogLeNet")]
ATTACKS = ["FGSM", "PGD", "CW", "DeepFool", "APGD", "Square"]


# ---------------------------------------------------------------------------
# Figure 1: teleportation drift box-plot
# ---------------------------------------------------------------------------
def fig_teleportation_drift():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    width = 0.25
    x = np.arange(len(ARCHS))
    splits = [("train", "C0"), ("test", "C1"), ("random", "C2")]
    handles = []
    for i, (split, color) in enumerate(splits):
        positions = x + (i - 1) * width
        data = []
        for arch, _ in ARCHS:
            with open(f"results/teleportation/{arch}_imagenet_teleportation.json") as f:
                d = json.load(f)
            # per_teleportation is a list of dicts; each has per-split distance summaries.
            # We extract the per-teleport mean for boxplot.
            per_tp = d.get("per_teleportation", [])
            vals = []
            for tp in per_tp:
                stats = tp.get(split, {}) if isinstance(tp.get(split), dict) else {}
                if "mean" in stats:
                    vals.append(stats["mean"])
            data.append(vals)
        bp = ax.boxplot(data, positions=positions, widths=width * 0.85,
                        patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for whisk in bp["whiskers"] + bp["caps"]:
            whisk.set_color(color)
        for med in bp["medians"]:
            med.set_color("k")
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.6, label=split))
    ax.set_xticks(x)
    ax.set_xticklabels([disp for _, disp in ARCHS], fontsize=10)
    ax.set_ylabel(r"penultimate drift  $\|h-h'\|_2 / \sqrt{D}$", fontsize=10)
    ax.set_title("Penultimate-feature drift under neural teleportation\n"
                 "(100 teleports per arch, 500 samples per split)", fontsize=11)
    ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "teleportation_drift.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'teleportation_drift.pdf'}")


# ---------------------------------------------------------------------------
# Figure 2: amplification per attack
# ---------------------------------------------------------------------------
def fig_amplification():
    fig, axes = plt.subplots(1, len(ARCHS), figsize=(13, 4), sharey=True)
    for ax, (arch, disp) in zip(axes, ARCHS):
        amp_M = []
        amp_h = []
        labels = []
        for atk in ATTACKS:
            p = Path(f"experiments/{arch}_imagenet/theorem45/per_attack/{atk}.json")
            if not p.exists():
                continue
            with open(p) as f:
                d = json.load(f)
            r = d.get("result", {})
            if r.get("skipped"):
                continue
            amp_M.append(r["amplification_M_median"])
            amp_h.append(r["amplification_h_median"])
            labels.append(atk)
        x = np.arange(len(labels))
        ax.bar(x - 0.2, amp_M, width=0.4, label=r"$d_M / d_f$", color="C0")
        ax.bar(x + 0.2, amp_h, width=0.4, label=r"$d_h / d_f$", color="C3")
        ax.axhline(1.0, color="k", linestyle="--", alpha=0.4, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(disp, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("amplification (median ratio)", fontsize=10)
            ax.legend(fontsize=9, frameon=False, loc="upper left")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle("Per-attack amplification of logit distance\n"
                 "(values > 1 mean the representation distance EXCEEDS logit distance)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "amplification_per_attack.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'amplification_per_attack.pdf'}")


if __name__ == "__main__":
    print("Generating teleportation drift figure...")
    fig_teleportation_drift()
    print("Generating amplification per-attack figure...")
    fig_amplification()
    print("Done.")
