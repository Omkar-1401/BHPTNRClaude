"""Learning-curve figures from summary.json / results/ (run summarize.py first).

  plots/learning_curve.{pdf,png}   V300 median / mean / max vs number of training points, one line per model (chosen
                                   start per grid = lowest training mean); faint dots = the other starts.
  plots/learning_ratio.{pdf,png}   the same, each model normalised by its own G96 refit (same protocol, so 1 = T96).
Palette: reference categorical slots 1-7 in fixed order; marker + line style per model as the secondary encoding.

    python plot_curves.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"; OUT.mkdir(exist_ok=True)
S = json.loads((HERE / "summary.json").read_text())
LAB = {"pn13": "pn_anchored_spin", "flux_orbit": "flux_anc_spin_with_orbit", "splitA": "split A", "splitB": "split B",
       "p40": "pass 40", "lean": "lean (pass 42 K1)", "rink": "Rink et al. form (refit)"}
# pass 39 L4 (results p39L4) is in SUMMARY.md but not plotted: pass 40 tracks it within ~2% at every grid
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MK = ["o", "s", "^", "v", "D", "X", "P"]; LS = ["-", "--", "-.", ":", "-", ":", "--"]
GRIDS = [24, 48, 96, 192]
SHAPE = {24: "6x4", 48: "8x6", 96: "12x8 (T96)", 192: "16x12"}
STATS = [("median", "median"), ("mean", "mean"), ("max", "max")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e6e6"


def style(ax):
    ax.set_xscale("log", base=2); ax.set_xticks(GRIDS); ax.set_xticklabels([f"{g}\n{SHAPE[g]}" for g in GRIDS])
    ax.minorticks_off(); ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.axvline(96, color=MUTED, lw=0.8, ls=(0, (2, 3)), zorder=0)


# second figure (2026-10-28): the lean pruning ladder, 18c -> 16c -> 12c -> 9c; its own slots 1-4 (adjacent-validated order)
LAB_LEAN = {"p40": "pass 40 (18c)", "lean": "lean (16c)", "lean12": "lean-12 (12c)", "lean9": "lean-9 (9c)"}


def draw(norm, series=None, tag=""):
    series = LAB if series is None else series
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.0), constrained_layout=True)
    for j, (st, lab) in enumerate(STATS):
        ax = axs[j]; style(ax)
        for i, (m, name) in enumerate(series.items()):
            cur = S[m]["curve"]; ref = cur["96"]["V300"][st] if norm else 1.0
            y = [cur[str(g)]["V300"][st] / ref for g in GRIDS]
            ax.plot(GRIDS, y, color=COL[i], ls=LS[i], lw=2, marker=MK[i], ms=8, mec="white", mew=1.2, label=name, zorder=3)
            for g in GRIDS:
                for s_, v in cur[str(g)]["starts"].items():
                    if s_ != cur[str(g)]["chosen"]:
                        ax.plot(g, v["V300"][st] / ref, ls="none", marker=MK[i], ms=4, color=COL[i], alpha=0.35, zorder=2)
        if norm:
            ax.axhline(1.0, color=MUTED, lw=0.8)
        else:
            ax.set_yscale("log")
        ax.set_title(f"V300 {lab}", fontsize=11, color=INK)
        ax.set_xlabel("training points (q x chi1)", fontsize=9, color=INK)
    axs[0].set_ylabel(r"$\mathcal{E}$ / $\mathcal{E}$(G96 refit)" if norm else r"full-window $\mathcal{E}$", fontsize=10, color=INK)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=len(series), frameon=False, fontsize=9)
    name = ("learning_ratio" if norm else "learning_curve") + tag
    fig.suptitle("Training-set size: models refit on uniform q x chi1 grids, scored on 300 fresh points (V300)"
                 + ("; each model / its own G96 refit" if norm else ""), fontsize=11, color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


if __name__ == "__main__":
    draw(False); draw(True); draw(False, LAB_LEAN, "_lean"); draw(True, LAB_LEAN, "_lean")
