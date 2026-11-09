"""Figure for twodet_incl_run.py: median two-detector mismatch (SNR^3-weighted over 8 signal orbital phases) vs total
mass at iota = pi/3 and pi/2, every model; the face-on medians (twodet_run.py) of pass 40 and Rink et al. as thin grey
references.  No computation.
    python fdm/twodet_incl_plot.py   -> fdm/twodet_incl_mismatch_vs_M.{png,pdf}, fdm/results/twodet_incl_summary.json
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from fd_plot import ORDER

HERE = Path(__file__).resolve().parent; RES = HERE / "results"
INC = {"pi/3": r"$\iota=\pi/3$", "pi/2": r"$\iota=\pi/2$ (edge-on)"}


def med(d, key):
    rr = [r for r in d["rows"] if "error" not in r]
    M = np.array(d["masses"])
    Y = np.array([[r[f"{key}_M{int(m)}"]["mm"] for m in M] for r in rr])
    return M, np.median(Y, axis=0), Y.max(axis=0)


def main():
    D = {m: json.loads((RES / f"twodet_incl_{m}.json").read_text()) for m, *_ in ORDER if (RES / f"twodet_incl_{m}.json").exists()}
    F = {m: json.loads((RES / f"twodet_{m}.json").read_text()) for m in ("p40", "rink") if (RES / f"twodet_{m}.json").exists()}
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True); summ = {}
    for ax, (inc, lab) in zip(axs, INC.items()):
        for m, ls in (("p40", "-"), ("rink", "--")):
            if m in F:
                M, y, _ = med(F[m], "rink")
                ax.plot(M, y, color="#c9c9c9", ls=ls, lw=2.4, zorder=0, label=f"{'pass 40' if m == 'p40' else 'Rink et al.'} face-on")
        for m, name, col, ls, mk in ORDER:
            if m in D:
                M, y, mx = med(D[m], inc)
                ax.plot(M, y, color=col, ls=ls, marker=mk, ms=4, lw=1.5, label=name)
                summ.setdefault(m, {})[inc] = {f"M{int(x)}": dict(median=float(a), max=float(b)) for x, a, b in zip(M, y, mx)}
        ax.set_xscale("log"); ax.set_yscale("log"); ax.axhline(0.01, color="#777", ls=":", lw=0.9)
        ax.set_title(lab, fontsize=11); ax.set_xlabel(r"total mass $M\ [M_\odot]$")
        ax.grid(True, which="both", color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
    axs[0].set_ylabel(r"median mismatch $1-\mathcal{O}$")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Two-detector FD mismatch vs NRHybSur3dq8, 8 modes, SNR-weighted over 8 signal orbital phases; "
                 r"Rink et al. procedure ($f_{\rm min}$ from the waveform)", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    for e in ("png", "pdf"):
        fig.savefig(HERE / f"twodet_incl_mismatch_vs_M.{e}", dpi=160)
    (RES / "twodet_incl_summary.json").write_text(json.dumps(summ, indent=1))
    for m, s in summ.items():
        print(m, {inc: {k: f"{v['median']:.2e}" for k, v in s[inc].items() if k in ("M10", "M50", "M100", "M300")} for inc in s})


if __name__ == "__main__":
    main()
