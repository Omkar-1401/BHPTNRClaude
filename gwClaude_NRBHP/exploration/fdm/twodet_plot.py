"""Figures for the two-detector study (twodet_run.py results).  No computation.
Rink et al. 2024 Fig. 12 layout: mismatch vs total mass, face-on, f_min from the waveform (Rink's rule).
  left / middle: Rink et al. 2024 and pass 40, one thin line per point with |chi1| <= 0.6, coloured by mass-ratio band
                 (q 3-6, q 6-8, as their q 3-6 / 6-15 bands), median of each band bold; 0.01 dashed.
  right:         median over all 126 points vs M for every model.
    python fdm/twodet_plot.py   -> fdm/twodet_mismatch_vs_M.{png,pdf}, fdm/results/twodet_summary.json
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from fd_plot import ORDER

HERE = Path(__file__).resolve().parent; RES = HERE / "results"
BAND = [("q 3-6", lambda q: q <= 6.0, "#2a78d6"), ("q 6-8", lambda q: q > 6.0, "#eb6834")]


def series(d, key="rink"):
    M = d["masses"]; rows = [r for r in d["rows"] if "error" not in r]
    return np.array(M), rows, lambda r: np.array([r[f"{key}_M{int(m)}"]["mm"] for m in M])


def main():
    D = {m: json.loads((RES / f"twodet_{m}.json").read_text()) for m, *_ in ORDER if (RES / f"twodet_{m}.json").exists()}
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.6), sharey=True)
    for ax, m, title in ((axs[0], "rink", "Rink et al. 2024 (published BHPTNRSur2dq1e3)"), (axs[1], "p40", "pass 40 (18 coef)")):
        M, rows, y = series(D[m])
        for lab, sel, col in BAND:
            rr = [r for r in rows if sel(r["q"]) and abs(r["chi1"]) <= 0.6 + 1e-9]
            Y = np.array([y(r) for r in rr])
            for v in Y:
                ax.plot(M, v, color=col, lw=0.6, alpha=0.25)
            ax.plot(M, np.median(Y, axis=0), color=col, lw=2.2, label=f"{lab}, $|\\chi_1|\\leq0.6$ (median of {len(rr)})")
        ax.set_title(title, fontsize=11)
    M, rows, _ = series(D["p40"])
    for m, lab, col, ls, mk in ORDER:
        if m in D:
            Mm, rr, y = series(D[m])
            ax2 = axs[2]; ax2.plot(Mm, np.median([y(r) for r in rr], axis=0), ls=ls, color=col, marker=mk, ms=4, lw=1.5, label=lab)
    axs[2].set_title("median over all 126 points", fontsize=11)
    for ax in axs:
        ax.set_xscale("log"); ax.set_yscale("log"); ax.axhline(0.01, color="#777", ls="--", lw=0.9)
        ax.set_xlabel(r"total mass $M\ [M_\odot]$"); ax.grid(True, which="both", color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    axs[0].set_ylabel(r"mismatch $1-\mathcal{O}$")
    fig.suptitle(r"Two-detector FD mismatch vs NRHybSur3dq8, face-on ($\iota=0$), aLIGO design (ZDHP); "
                 r"$f_{\rm min}$ = (2,2) frequency after a 1.5-cycle taper of the $-5000\,M$ window (Rink et al. procedure)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for e in ("png", "pdf"):
        fig.savefig(HERE / f"twodet_mismatch_vs_M.{e}", dpi=160)
    summ = {}
    for m, d in D.items():
        Mm, rr, _ = series(d)
        for key in d["fmins"]:
            _, _, y = series(d, key)
            Y = np.array([y(r) for r in rr])
            summ.setdefault(m, {})[key] = {f"M{int(x)}": dict(median=float(np.median(Y[:, i])), max=float(Y[:, i].max()))
                                           for i, x in enumerate(Mm)}
    (RES / "twodet_summary.json").write_text(json.dumps(summ, indent=1))
    print("wrote twodet_mismatch_vs_M.{png,pdf}, results/twodet_summary.json")


if __name__ == "__main__":
    main()
