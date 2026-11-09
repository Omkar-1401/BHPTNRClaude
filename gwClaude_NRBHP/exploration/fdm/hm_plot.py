"""Figures for the multimode FD study (hm_run.py results; SNR-weighted over orientations).  No computation.
    python fdm/hm_plot.py   -> fdm/hm_mismatch_vs_q_<iota>_M<M>.{png,pdf}, fdm/results/hm_summary.json
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from fd_plot import ORDER
HERE = Path(__file__).resolve().parent; RES = HERE / "results"
INC = {"pi/3": r"\iota=\pi/3", "pi/2": r"\iota=\pi/2"}


def main():
    D = {m: json.loads((RES / f"hm_{m}.json").read_text()) for m, *_ in ORDER if (RES / f"hm_{m}.json").exists()}
    chis = sorted({r["chi1"] for d in D.values() for r in d["rows"]}); summ = {}
    for inc, lab_i in INC.items():
        for M in (100, 200):
            key = f"{inc}_M{M}"
            fig, axs = plt.subplots(2, 3, figsize=(15, 8.2), sharex=True, sharey=True)
            for ax, c in zip(axs.flat, chis):
                for m, lab, col, ls, mk in ORDER:
                    if m not in D:
                        continue
                    rr = sorted([r for r in D[m]["rows"] if r["chi1"] == c and "error" not in r], key=lambda r: r["q"])
                    ax.plot([r["q"] for r in rr], [r[key]["mm"] for r in rr], ls=ls, color=col, marker=mk, ms=3.5, lw=1.3, label=lab)
                ax.set_yscale("log"); ax.set_title(rf"$\chi_1 = {c:+.1f}$", fontsize=11)
                ax.grid(True, color="#e6e6e6", lw=0.6, which="both"); ax.set_axisbelow(True)
            for ax in axs[-1]:
                ax.set_xlabel("q")
            for ax in axs[:, 0]:
                ax.set_ylabel(r"$1 - \mathcal{O}$ (aLIGO ZDHP)")
            h, l = axs[0, 0].get_legend_handles_labels()
            fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=9, frameon=False)
            flo = "20 Hz" if M == 200 else r"20 Hz, raised to 23-34 Hz so (4,4) starts in band"
            fig.suptitle(rf"8-mode strain mismatch vs NRHybSur3dq8, ${lab_i}$, $M={M}\,M_\odot$, $f_\mathrm{{low}}$ = {flo}; "
                         "SNR-weighted over 12 signal orientations", fontsize=11)
            fig.tight_layout(rect=(0, 0.05, 1, 0.97))
            for e in ("png", "pdf"):
                fig.savefig(HERE / f"hm_mismatch_vs_q_{inc.replace('/', '')}_M{M}.{e}", dpi=160)
            plt.close(fig)
            for m, d in D.items():
                v = np.array([r[key]["mm"] for r in d["rows"] if "error" not in r])
                summ.setdefault(m, {})[key] = dict(median=float(np.median(v)), max=float(v.max()))
    (RES / "hm_summary.json").write_text(json.dumps(summ, indent=1))
    for m, s in summ.items():
        print(m, {k: f"{v['median']:.2e}/{v['max']:.2e}" for k, v in s.items()})


if __name__ == "__main__":
    main()
