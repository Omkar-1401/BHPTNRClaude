"""alpha/beta overlay figures for pn_anc_spin_physical (the PN-ordered spin model).

Format is VERBATIM that of the nonspinning overlays and of
`spin_alpha_beta_overlay.py`: 1 x 2 (alpha left, beta right), figsize (12, 4), one
curve per q in {2,3,5,8} coloured tab:red/orange/green/blue, resampled onto the NR time
grid, lw 2.2 for the q = 2 extrapolation, xlim (-1000, 80), grid on, default rcParams,
PDF + PNG.  Axis labels are mathtext.  Nothing else about the format changes, so these
overlay directly onto the predecessor's figures.

ONE FIGURE PER chi1, for chi1 in {-0.6, -0.2, +0.2, +0.6} -- two negative, two
positive, all four INSIDE the fit's chi1 grid [-0.6, +0.8].  Plus one 4 x 2 contact
sheet stacking them, which is the only view in which the spin trend is directly
readable, and a 4 x 2 DIFFERENCE sheet against the 13-coefficient predecessor, which is
where the PN re-ordering actually shows up (the spin correction moves out of the merger
plateau and into the inspiral).

DEFINITIONS.  alpha(t), beta(t) come from `fit_pn_anc_spin_physical.model_ab` with all
11 coefficients read from `pn_anc_spin_physical_results/coeffs_physical.json`; see that
module's docstring for what every symbol means and where it comes from.  The curves are
drawn against t_NR (the NR-proxy time coordinate), reached from the ppBHPT time t by

    tau = t0 + cumtrapz(beta, t) - interp(T_ANCHOR, t, cumtrapz(beta, t))

with T_ANCHOR = -100 M.  The time offset t0 is the per-case mismatch optimum found by
the same two-stage (fit-window then full-window) search the fit itself uses, so the E
quoted in each legend entry is the model's ACTUAL grid mismatch at that (q, chi1) --
not a replotted number.

TWO KINDS OF EXTRAPOLATION ARE ON THESE AXES, and they are not the same thing:
  * q = 2 is outside the calibration's fitted range nu in [0.0988, 0.1875]
    (q in [3, 8]); q = 2 is nu = 0.2222, 18.5% ABOVE the top of it -- the same extrapolation the nonspinning overlays carry;
  * q = 2 is ALSO outside BHPTNRSur2dq1e3's own domain, which starts at q = 3
    (the surrogate only prints a warning and evaluates).  Any q = 2 pathology here may
    be the source waveform, not the calibration; the legend says so.

Usage:  python plot_pn_anc_spin_physical_overlay.py [--partial] [--no-diff]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fit_pn_anchored_spin as S
import fit_pn_anc_spin_physical as P

mpl.rcdefaults()                      # AFTER the imports -- gw_remnant sets fonts
mpl.rcParams["text.usetex"] = False

OUT = HERE / "Agentic_spin_plots" / "pn_anc_spin_physical" / "chi_const"
OUT.mkdir(parents=True, exist_ok=True)

Q_INPUTS = [2.0, 3.0, 5.0, 8.0]
CHI_PLOT = [-0.6, -0.2, 0.2, 0.6]
COLS = {2.0: "tab:red", 3.0: "tab:orange", 5.0: "tab:green", 8.0: "tab:blue"}
Q_TRAIN_LO = 3.0
Q_SUR_LO = 3.0                        # BHPTNRSur2dq1e3 lower bound


def curve(th, q, chi1):
    """alpha(tau), beta(tau) on the NR clock, at the mismatch-optimal t0."""
    case = S.build_case(q, chi1)
    _, t0 = P.mismatch(th, q, chi1, case, None)
    err, t0 = P.mismatch(th, q, chi1, case, t0)
    t, x, pl = case["t_bhpt"], case["x"], case["p_loss"]
    alpha, beta = P.model_ab(th, q, chi1, x, pl)
    bc = S.creative.cumulative_trapezoid(beta, t)
    tau = t0 + bc - np.interp(S.T_ANCHOR, t, bc)
    return tau, alpha, beta, err, case["t_nr"]


def curve_pred(th9, ths, q, chi1):
    """Same, for the 13-coefficient predecessor, for the difference sheet."""
    case = S.build_case(q, chi1)
    _, t0 = S.mismatch(th9, ths, q, chi1, case, "fit", -50.0)
    err, t0 = S.mismatch(th9, ths, q, chi1, case, "full", t0)
    t = case["t_bhpt"]
    alpha, beta = S.model_ab(th9, ths, q, chi1, case["x"], case["p_loss"])
    bc = S.creative.cumulative_trapezoid(beta, t)
    tau = t0 + bc - np.interp(S.T_ANCHOR, t, bc)
    return tau, alpha, beta, err, case["t_nr"]


def draw(axa, axb, data, chi1, legend=True):
    for q in Q_INPUTS:
        tau, alpha, beta, err, t_nr = data[(q, chi1)]
        m = (t_nr >= max(t_nr[0], tau[0])) & (t_nr <= min(t_nr[-1], tau[-1]))
        t = t_nr[m]
        lw = 2.2 if q < Q_TRAIN_LO else 1.3
        lab = f"q={q:.0f}"
        if q < Q_TRAIN_LO:
            lab += " [extrap: nu AND surrogate]" if q < Q_SUR_LO else " [extrap]"
        lab += f"  E={err:.2e}"
        axa.plot(t, np.interp(t, tau, alpha), color=COLS[q], lw=lw, label=lab)
        axb.plot(t, np.interp(t, tau, beta), color=COLS[q], lw=lw, label=f"q={q:.0f}")
    for ax, yl in ((axa, r"$\alpha(t)$"), (axb, r"$\beta(t)$")):
        ax.axvline(0, color="gray", ls=":", lw=0.8)
        ax.set_xlim(-1000, 80)
        ax.grid(True)
        ax.set_xlabel(r"$t_{\rm NR}\,/\,M$")
        ax.set_ylabel(yl)
        if legend:
            ax.legend(fontsize=8)


def draw_diff(axa, axb, data, dpred, chi1, legend=True):
    """Fractional change from the predecessor, on the shared NR clock."""
    for q in Q_INPUTS:
        tau, alpha, beta, _, t_nr = data[(q, chi1)]
        tp, ap, bp, _, _ = dpred[(q, chi1)]
        lo = max(t_nr[0], tau[0], tp[0]); hi = min(t_nr[-1], tau[-1], tp[-1])
        t = t_nr[(t_nr >= lo) & (t_nr <= hi)]
        an, bn = np.interp(t, tau, alpha), np.interp(t, tau, beta)
        ao, bo = np.interp(t, tp, ap), np.interp(t, tp, bp)
        lw = 2.2 if q < Q_TRAIN_LO else 1.3
        axa.plot(t, 100 * (an - ao) / np.abs(ao), color=COLS[q], lw=lw,
                 label=f"q={q:.0f}")
        axb.plot(t, 100 * (bn - bo) / np.abs(bo), color=COLS[q], lw=lw,
                 label=f"q={q:.0f}")
    for ax, yl in ((axa, r"$\Delta\alpha/\alpha$  [%]"),
                   (axb, r"$\Delta\beta/\beta$  [%]")):
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(0, color="gray", ls=":", lw=0.8)
        ax.set_xlim(-1000, 80)
        ax.grid(True)
        ax.set_xlabel(r"$t_{\rm NR}\,/\,M$")
        ax.set_ylabel(yl)
        if legend:
            ax.legend(fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true",
                    help="plot the optimiser's in-flight checkpoint instead of the "
                         "stored coefficients")
    ap.add_argument("--no-diff", action="store_true",
                    help="skip the difference sheet against the 13c predecessor")
    args = ap.parse_args()

    th, src = P.shipped11(args.partial)
    tag = "  [PARTIAL]" if args.partial else ""
    print(f"theta from {src.name}: " +
          ", ".join(f"{n}={v:+.4f}" for n, v in zip(P.NAMES, th)), flush=True)

    data = {}
    for chi1 in CHI_PLOT:
        for q in Q_INPUTS:
            data[(q, chi1)] = curve(th, q, chi1)
            print(f"  q={q:.0f} chi1={chi1:+.1f}  E={data[(q,chi1)][3]:.4e}", flush=True)

    # one figure per chi1, in the nonspinning overlay format verbatim
    for chi1 in CHI_PLOT:
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4))
        draw(axa, axb, data, chi1)
        axa.set_title(r"$\alpha(t)$ — master")
        axb.set_title(r"$\beta(t)$ — master")
        fig.suptitle(r"pn_anc_spin_physical (1.5PN SO + 2PN SS, 11 coef): master "
                     r"$\alpha/\beta$ across $q$ at $\chi_1 = "
                     f"{chi1:+.1f}$" + tag)
        fig.tight_layout()
        stem = (f"pn_anc_spin_physical_alpha_beta_overlay_chi{chi1:+.1f}"
                .replace("+", "p").replace("-", "m"))
        for ext in ("pdf", "png"):
            p = OUT / f"{stem}.{ext}"
            fig.savefig(p)
            print("wrote", p, flush=True)
        plt.close(fig)

    # contact sheet: rows = chi1, so the spin trend reads down the column
    fig, axes = plt.subplots(len(CHI_PLOT), 2, figsize=(12, 4 * len(CHI_PLOT)))
    for i, chi1 in enumerate(CHI_PLOT):
        axa, axb = axes[i]
        draw(axa, axb, data, chi1, legend=(i == 0))
        axa.set_title(r"$\alpha(t)$,  $\chi_1 = " + f"{chi1:+.1f}$")
        axb.set_title(r"$\beta(t)$,  $\chi_1 = " + f"{chi1:+.1f}$")
    fig.suptitle(r"pn_anc_spin_physical: master $\alpha/\beta$, "
                 r"$\chi_1 \in \{-0.6, -0.2, +0.2, +0.6\}$" + tag)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = OUT / f"pn_anc_spin_physical_alpha_beta_overlay_all.{ext}"
        fig.savefig(p)
        print("wrote", p, flush=True)
    plt.close(fig)

    # the numbers behind the legends
    print(f"\nmismatch at the plotted cases\n{'q':>5} " +
          " ".join(f"{c:>+10.1f}" for c in CHI_PLOT))
    for q in Q_INPUTS:
        print(f"{q:>5.0f} " + " ".join(f"{data[(q,c)][3]:10.3e}" for c in CHI_PLOT))

    if args.no_diff:
        return

    # difference sheet against the 13-coefficient predecessor
    pred = json.loads(P.PRED.read_text())
    th9, ths = S.model9(), np.array(pred["theta_spin"], float)   # installed 9 (2026-09-27)
    print("\npredecessor (13c) at the same cases:", flush=True)
    dpred = {}
    for chi1 in CHI_PLOT:
        for q in Q_INPUTS:
            dpred[(q, chi1)] = curve_pred(th9, ths, q, chi1)
            print(f"  q={q:.0f} chi1={chi1:+.1f}  E={dpred[(q,chi1)][3]:.4e}", flush=True)

    fig, axes = plt.subplots(len(CHI_PLOT), 2, figsize=(12, 4 * len(CHI_PLOT)))
    for i, chi1 in enumerate(CHI_PLOT):
        axa, axb = axes[i]
        draw_diff(axa, axb, data, dpred, chi1, legend=(i == 0))
        axa.set_title(r"$\Delta\alpha/\alpha$,  $\chi_1 = " + f"{chi1:+.1f}$")
        axb.set_title(r"$\Delta\beta/\beta$,  $\chi_1 = " + f"{chi1:+.1f}$")
    fig.suptitle(r"pn_anc_spin_physical (11 coef) $-$ pn_anchored+spin (13 coef): "
                 r"where the PN re-ordering moves $\alpha/\beta$" + tag)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = OUT / f"pn_anc_spin_physical_vs_predecessor_diff.{ext}"
        fig.savefig(p)
        print("wrote", p, flush=True)
    plt.close(fig)

    print(f"\nmismatch ratio, physical(11c) / predecessor(13c)\n{'q':>5} " +
          " ".join(f"{c:>+10.1f}" for c in CHI_PLOT))
    for q in Q_INPUTS:
        print(f"{q:>5.0f} " + " ".join(
            f"{data[(q,c)][3]/dpred[(q,c)][3]:9.2f}x" for c in CHI_PLOT))


if __name__ == "__main__":
    main()
