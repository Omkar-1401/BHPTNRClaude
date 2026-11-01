"""Figures for the frequency-domain mismatch study (fd_run.py results).  No computation.
    python fdm/fd_plot.py   -> fdm/fd_mismatch_vs_q_M{100,200}.{png,pdf}, fdm/fd_vs_E.{png,pdf}, fdm/results/fd_summary.json
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; RES = HERE / "results"
ORDER = [("p40", "pass 40 (18)", "#173f7a", "-", "o"), ("p40lean", "lean-16", "#2a78d6", "-", "s"),
         ("p49b12", "lean-12", "#1baf7a", "-", "^"), ("p49b9", "lean-9", "#7cc242", "-", "D"),
         ("flux_orbit", "flux_anc_spin_with_orbit (9)", "#8c8c8c", "-", "v"),
         ("rink", "Rink et al. 2024 (12)", "#e34948", "--", "x"), ("zhong", "Zhong et al. 2025 (8, spin-blind)", "#b07a10", "--", "+")]


def load():
    D = {}
    for m, *_ in ORDER:
        f = RES / f"fd_{m}.json"
        if f.exists():
            D[m] = json.loads(f.read_text())
    return D


def main():
    D = load(); summ = {}
    chis = sorted({r["chi1"] for d in D.values() for r in d["rows"]})
    for M in (100, 200):
        fig, axs = plt.subplots(2, 3, figsize=(15, 8.2), sharex=True, sharey=True)
        for ax, c in zip(axs.flat, chis):
            for m, lab, col, ls, mk in ORDER:
                if m not in D:
                    continue
                rr = sorted([r for r in D[m]["rows"] if r["chi1"] == c and "error" not in r], key=lambda r: r["q"])
                q = [r["q"] for r in rr]; y = [r[f"M{M}"]["mm"] for r in rr]
                ax.plot(q, y, ls=ls, color=col, marker=mk, ms=3.5, lw=1.3, label=lab)
                fl = [r for r in rr if r[f"M{M}"]["flagged"]]
                if fl:
                    ax.scatter([r["q"] for r in fl], [r[f"M{M}"]["mm"] for r in fl], s=40, facecolors="none", edgecolors=col, lw=0.8)
            ax.set_yscale("log"); ax.set_title(rf"$\chi_1 = {c:+.1f}$", fontsize=11)
            ax.grid(True, color="#e6e6e6", lw=0.6, which="both"); ax.set_axisbelow(True)
        for ax in axs[-1]:
            ax.set_xlabel("q")
        for ax in axs[:, 0]:
            ax.set_ylabel(r"$1 - \mathcal{O}$ (aLIGO ZDHP)")
        h, l = axs[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=9, frameon=False)
        fig.suptitle(rf"(2,2) frequency-domain mismatch vs NRHybSur3dq8, $M = {M}\,M_\odot$, $f_\mathrm{{low}} = 20$ Hz; "
                     "maximised over time, template polarisation and phase; mean over 4 signal polarisations", fontsize=11)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        for e in ("png", "pdf"):
            fig.savefig(HERE / f"fd_mismatch_vs_q_M{M}.{e}", dpi=160)
        plt.close(fig)
        for m in D:
            ok = [r for r in D[m]["rows"] if "error" not in r]
            v = np.array([r[f"M{M}"]["mm"] for r in ok]); w = np.array([r[f"M{M}"]["mm_worst"] for r in ok])
            summ.setdefault(m, {})[f"M{M}"] = dict(median=float(np.median(v)), mean=float(v.mean()), max=float(v.max()),
                                                   worst_pol_max=float(w.max()), n=len(v),
                                                   flagged=int(sum(r[f"M{M}"]["flagged"] for r in ok)))
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for m, lab, col, ls, mk in ORDER:
        if m in D:
            ok = [r for r in D[m]["rows"] if "error" not in r]
            ax.scatter([r["E"] for r in ok], [r["M100"]["mm"] for r in ok], s=12, color=col, marker=mk, label=lab, alpha=0.8)
            summ[m]["E"] = dict(median=float(np.median([r["E"] for r in ok])))
    lim = ax.get_xlim(); ax.plot([1e-6, 1], [1e-6, 1], color="#999", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"time-domain $\mathcal{E}$ (flat noise)")
    ax.set_ylabel(r"FD mismatch, $M = 100\,M_\odot$"); ax.legend(fontsize=7.5, frameon=False)
    ax.grid(True, color="#e6e6e6", lw=0.6, which="both"); fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(HERE / f"fd_vs_E.{e}", dpi=160)
    (RES / "fd_summary.json").write_text(json.dumps(summ, indent=1))
    for m, s in summ.items():
        print(m, {k: (f"{v['median']:.2e}/{v['max']:.2e}" if "max" in v else f"{v['median']:.2e}") for k, v in s.items()})


if __name__ == "__main__":
    main()
