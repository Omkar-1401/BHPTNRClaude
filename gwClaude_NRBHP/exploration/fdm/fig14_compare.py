"""Zhong et al. 2025 Fig. 14 style: per-mode match O_IMR vs q at chi1 = 0 for (2,2), (3,3), (4,4).
Their BHP2NRMLSur curves (vector data from their m3.pdf; target NRHybSur3dq8_CCE; PSD and total mass not stated) against
our models (mode_match.py; target NRHybSur3dq8, aLIGO ZDHP).  Logit y axis with their ticks.
f_low is 20 Hz unless a mode starts above that in our -5000..+100 M window; then it is raised per point (fd_core).  The
actual per-mode f_low range is printed in the legend, and points with a raised f_low are drawn hollow.
    python fdm/fig14_compare.py          -> fdm/fig14_compare.{png,pdf}        (M = 100, 200; modes_<m>.json)
    python fdm/fig14_compare.py 50       -> fdm/fig14_compare_M50.{png,pdf}    (modes_<m>_M50.json)
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; RES = HERE / "results"
COL = {"22": "#1f77b4", "33": "#ff7f0e", "44": "#2ca02c"}
OURS = [("p40", "pass 40 (18)", "--", "s"), ("p49b12", "lean-12", ":", "^")]


def rows_for(m, M):
    for f in (RES / f"modes_{m}.json", RES / f"modes_{m}_M{M}.json"):
        if f.exists():
            rr = json.loads(f.read_text())["rows"]
            if f"22_M{M}" in rr[0]:
                return sorted(rr, key=lambda r: r["q"])
    return None


def main():
    masses = [int(x) for x in sys.argv[1:]] or [100, 200]
    Z = json.loads((RES / "zhong_fig14_digitized.json").read_text())["curves"]
    fig, axs = plt.subplots(1, len(masses), figsize=(7 * len(masses) if len(masses) > 1 else 9.5, 7.2), sharey=True, squeeze=False)
    for ax, M in zip(axs[0], masses):
        for c in Z:
            if c["model"] != "BHP2NRMLSur":
                continue
            md = c["mode"].strip("()").replace(",", "")
            ax.plot(c["q"], c["O"], "-", color=COL[md], lw=1.6, alpha=0.9, label=f"{c['mode']} BHP2NRMLSur (Zhong et al.)")
        for m, lab, ls, mk in OURS:
            rr = rows_for(m, M)
            if rr is None:
                continue
            for md in ("22", "33", "44"):
                q = np.array([r["q"] for r in rr]); O = np.array([r[f"{md}_M{M}"]["O"] for r in rr])
                fl = np.array([r[f"{md}_M{M}"]["f_low"] for r in rr]); raised = np.array([r[f"{md}_M{M}"]["flagged"] for r in rr])
                frng = "20 Hz" if not raised.any() else (f"{fl.min():.0f}-{fl.max():.0f} Hz" if fl.min() > 20 else
                                                          f"20-{fl.max():.0f} Hz")
                ax.plot(q, O, ls=ls, color=COL[md], lw=1.4, label=f"({md[0]}, {md[1]}) {lab}, $f_{{\\rm low}}$ {frng}")
                ax.scatter(q[~raised], O[~raised], marker=mk, s=18, color=COL[md], zorder=3)
                ax.scatter(q[raised], O[raised], marker=mk, s=22, facecolors="white", edgecolors=COL[md], lw=1.0, zorder=3)
        ax.set_yscale("logit"); ax.set_ylim(0.9, 0.99985)
        ax.set_yticks([0.9, 0.97, 0.99, 0.997, 0.999, 0.9997]); ax.set_yticklabels(["0.900", "0.970", "0.990", "0.997", "0.999", "0.9997"])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("q"); ax.grid(True, color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
        ax.set_title(rf"ours: $M={M}\,M_\odot$, aLIGO ZDHP vs NRHybSur3dq8 (hollow: $f_{{\rm low}}$ raised above 20 Hz)"
                     "\n" + r"Zhong et al.: vs NRHybSur3dq8_CCE, PSD / mass not stated", fontsize=10)
        ax.legend(loc="lower right", fontsize=7.5, frameon=False, ncol=1) if len(masses) > 1 else None
    axs[0, 0].set_ylabel(r"$\mathcal{O}_{\rm IMR}$ (per mode, $\chi_1=0$)")
    if len(masses) == 1:
        h, l = axs[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, fontsize=8.5, frameon=False)
        fig.tight_layout(rect=(0, 0.19, 1, 1))
    else:
        fig.tight_layout()
    stem = "fig14_compare" if masses == [100, 200] else "fig14_compare_" + "_".join(f"M{M}" for M in masses)
    for e in ("png", "pdf"):
        fig.savefig(HERE / f"{stem}.{e}", dpi=160)
    for m, *_ in OURS:
        for M in masses:
            rr = rows_for(m, M)
            if rr:
                print(m, M, {md: "O %.4f..%.4f, f_low %.1f..%.1f" % (
                    min(r[f"{md}_M{M}"]["O"] for r in rr), max(r[f"{md}_M{M}"]["O"] for r in rr),
                    min(r[f"{md}_M{M}"]["f_low"] for r in rr), max(r[f"{md}_M{M}"]["f_low"] for r in rr)) for md in ("22", "33", "44")})


if __name__ == "__main__":
    main()
