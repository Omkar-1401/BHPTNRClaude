"""alpha/beta overlay figures for the SPIN-EXTENDED pn_anchored model.

Format is VERBATIM that of the nonspinning overlays
(NRBHP_alpha_beta_overlays.save_alpha_beta_overlay): 1 x 2 (alpha left, beta right),
figsize (12, 4), one curve per q in {2,3,5,8} coloured tab:red/orange/green/blue,
resampled onto the NR time grid, lw 2.2 for the q = 2 extrapolation, xlim (-1000, 80),
grid on, default rcParams, PDF + PNG.  Axis labels are mathtext (the standing request
to stop shipping raw ASCII symbols); nothing else about the format changes.

ONE FIGURE PER chi1, for chi1 in {-0.6, -0.2, +0.2, +0.6} -- two negative, two
positive, all four INSIDE the fit's chi1 grid [-0.6, +0.8].  Plus one 4 x 2 contact
sheet stacking them, which is the only view in which the spin trend is directly
readable.

TWO KINDS OF EXTRAPOLATION ARE ON THESE AXES, and they are not the same thing:
  * q = 2 is outside the calibration's fitted range nu in [0.0988, 0.1875]
    (q in [3, 8]); q = 2 is nu = 0.2222, 18.5% ABOVE the top of it -- the same extrapolation the nonspinning overlays carry;
  * q = 2 is ALSO outside BHPTNRSur2dq1e3's own domain, which starts at q = 3
    (X_min_q = log10(3); the surrogate only prints a warning and evaluates).  The
    nonspinning BHPTNRSur1dq1e4 reaches q = 2.5, so this is a NEW failure mode
    specific to the spinning source.  Any q = 2 pathology here may be the source
    waveform, not the calibration; the legend says so.

alpha/beta come from fit_pn_anchored_spin.model_ab with the installed 9 (`theta9` in
coeffs_spin.json: b1, b2, a2 from nonspinning pn_anchored, the switch + merger six fitted on
the 2d grid -- 2026-09-27) and the 4 spin coefficients from coeffs_spin.json.  The
time offset t0 is the per-case mismatch optimum (the same two-stage fit/full search the
fit itself uses), so the quoted E in each legend entry is the model's actual 96-point
grid mismatch at that (q, chi1) -- not a replotted number.

Usage:  python spin_alpha_beta_overlay.py [--partial]
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

mpl.rcdefaults()                      # AFTER the imports -- gw_remnant sets fonts
mpl.rcParams["text.usetex"] = False

OUT = HERE / "Agentic_spin_plots" / "pn_anchored_spin" / "chi_const"
OUT.mkdir(parents=True, exist_ok=True)

Q_INPUTS = [2.0, 3.0, 5.0, 8.0]
CHI_PLOT = [-0.6, -0.2, 0.2, 0.6]
COLS = {2.0: "tab:red", 3.0: "tab:orange", 5.0: "tab:green", 8.0: "tab:blue"}
Q_TRAIN_LO, Q_TRAIN_HI = 3.0, 8.0
Q_SUR_LO = 3.0                        # BHPTNRSur2dq1e3 lower bound


def load_theta_spin(partial):
    """The fitted 4 spin coefficients; --partial reads the in-flight checkpoint."""
    p = S.PARTIAL if partial else S.COEFFS
    if not p.exists():
        sys.exit(f"no coefficients at {p} -- the fit has not written yet")
    d = json.loads(p.read_text())
    if "theta9" not in d:
        sys.exit(f"{p.name} has no theta9 (pre-2026-09-27 frozen-9 file) -- refusing to pair "
                 "its spin terms with the installed switch+merger 9")
    return np.array(d["theta_spin"], float), p


def curve(th9, ths, q, chi1):
    """alpha(tau), beta(tau) on the NR clock, at the mismatch-optimal t0."""
    case = S.build_case(q, chi1)
    _, t0 = S.mismatch(th9, ths, q, chi1, case, "fit", -50.0)
    err, t0 = S.mismatch(th9, ths, q, chi1, case, "full", t0)
    t, x, pl = case["t_bhpt"], case["x"], case["p_loss"]
    alpha, beta = S.model_ab(th9, ths, q, chi1, x, pl)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true",
                    help="plot the optimiser's in-flight checkpoint instead of the "
                         "converged coefficients")
    args = ap.parse_args()

    ths, src = load_theta_spin(args.partial)
    th9 = np.array(json.loads(src.read_text())["theta9"], float)   # installed 9 (2026-09-27)
    if not args.partial:
        assert np.array_equal(th9, S.model9())
    tag = "  [PARTIAL]" if args.partial else ""
    print(f"theta_spin from {src.name}: " +
          ", ".join(f"{n}={v:+.4f}" for n, v in zip(S.SNAMES, ths)), flush=True)
    print(f"base 9 (installed): {np.round(th9, 4)}\n", flush=True)

    data = {}
    for chi1 in CHI_PLOT:
        for q in Q_INPUTS:
            data[(q, chi1)] = curve(th9, ths, q, chi1)
            print(f"  q={q:.0f} chi1={chi1:+.1f}  E={data[(q,chi1)][3]:.4e}", flush=True)

    # one figure per chi1, in the nonspinning overlay format verbatim
    for chi1 in CHI_PLOT:
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4))
        draw(axa, axb, data, chi1)
        axa.set_title(r"$\alpha(t)$ — master")
        axb.set_title(r"$\beta(t)$ — master")
        fig.suptitle(f"pn-anchored + spin: master "
                     r"$\alpha/\beta$ across $q$ at $\chi_1 = "
                     f"{chi1:+.1f}$" + tag)
        fig.tight_layout()
        stem = f"pn_anchored_spin_alpha_beta_overlay_chi{chi1:+.1f}".replace("+", "p").replace("-", "m")
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
    fig.suptitle(r"pn-anchored + spin: master $\alpha/\beta$, "
                 r"$\chi_1 \in \{-0.6, -0.2, +0.2, +0.6\}$" + tag)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = OUT / f"pn_anchored_spin_alpha_beta_overlay_all.{ext}"
        fig.savefig(p)
        print("wrote", p, flush=True)
    plt.close(fig)

    # the numbers behind the legends
    print(f"\n{'q':>5} " + " ".join(f"{c:>+10.1f}" for c in CHI_PLOT))
    for q in Q_INPUTS:
        print(f"{q:>5.0f} " + " ".join(f"{data[(q,c)][3]:10.3e}" for c in CHI_PLOT))


if __name__ == "__main__":
    main()
