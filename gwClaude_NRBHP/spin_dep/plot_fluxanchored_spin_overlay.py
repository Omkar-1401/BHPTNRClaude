"""alpha/beta overlays for fluxanchored + spin, fitted on the 2d grid (9 coefficients).

House format, matching every other overlay in the workspace: 1 x 2 (alpha left, beta
right), figsize (12, 4), q in {2,3,5,8} in tab:red/orange/green/blue, lw 2.2 for the q=2
extrapolation, xlim (-1000, 80), default rcParams, mathtext axis labels.  One figure per
chi1 in {-0.6, -0.2, +0.2, +0.6}, plus a 4x2 contact sheet with chi1 down the rows.

q = 2 is labelled [extrap: nu AND surrogate] because it is outside BOTH the fit's
training range (q in [3,8]) and BHPTNRSur2dq1e3's own domain (q_min = 3, where the
surrogate only warns).

Reads gwr_flux... no: reads fluxanchored_spin_results/coeffs_nu.json, the converged
W = nu fit (job 3483948).
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")
import fit_scaling_gwr_energy_flux as FL
import fit_fluxanchored_spin as FS

mpl.rcdefaults(); mpl.rcParams["text.usetex"] = False
OUT = HERE / "Agentic_spin_plots" / "fluxanchored_spin" / "chi_const"; OUT.mkdir(parents=True, exist_ok=True)
QS = [2.0, 3.0, 5.0, 8.0]
# chi1 = 0 INCLUDED: every spin term vanishes identically there, so that panel
# is the reference the spinning ones should be read against.
CHI = [-0.6, -0.2, 0.0, 0.2, 0.6]
COLS = {2.0: "tab:red", 3.0: "tab:orange", 5.0: "tab:green", 8.0: "tab:blue"}


def curve(th, q, chi1):
    case = FS.build_case(q, chi1)
    p = FS.params_at(q, chi1, th)
    g = np.linspace(-250.0, 150.0, 81)
    e0 = [FL.fast_mismatch(p, case, t0, "fit")[0] for t0 in g]
    _, t0 = FL.fast_mismatch(p, case, float(g[int(np.argmin(e0))]), "fit")
    err = FL.fast_mismatch(p, case, t0, "full")[0]
    a_pp, a_f, b_pp, b_e = p
    los = case["losses"]
    alpha = a_pp * (1.0 + a_f * los["flux_hat"])
    beta = b_pp * (1.0 + b_e * los["e_oft"])
    t = np.asarray(case["t_bhpt"], float)
    bc = FS.G.creative.cumulative_trapezoid(beta, t)
    tau = t0 + bc - float(np.interp(FS.G.T_ANCHOR, t, bc))
    return tau, alpha, beta, err, case["t_nr"]


def draw(axa, axb, data, chi1, legend=True):
    for q in QS:
        tau, al, be, err, t_nr = data[(q, chi1)]
        m = (t_nr >= max(t_nr[0], tau[0])) & (t_nr <= min(t_nr[-1], tau[-1]))
        t = t_nr[m]; lw = 2.2 if q < 3.0 else 1.3
        lab = f"q={q:.0f}" + (" [extrap: nu AND surrogate]" if q < 3.0 else "")
        lab += f"  E={err:.2e}"
        axa.plot(t, np.interp(t, tau, al), color=COLS[q], lw=lw, label=lab if legend else None)
        axb.plot(t, np.interp(t, tau, be), color=COLS[q], lw=lw,
                 label=(f"q={q:.0f}" if legend else None))
    for ax, yl in ((axa, r"$\alpha(t)$"), (axb, r"$\beta(t)$")):
        ax.axvline(0, color="gray", ls=":", lw=0.8)
        ax.set_xlim(-1000, 80); ax.grid(True)
        ax.set_xlabel(r"$t_{\rm NR}\,/\,M$"); ax.set_ylabel(yl)
        if legend:
            ax.legend(fontsize=8)


def main():
    d = json.loads((HERE / "fluxanchored_spin_results" / "coeffs_nu.json").read_text())
    th = np.array(d["theta"], float)
    FS.WEIGHT = d["weight"]
    print("theta: " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(d["names"], th)))
    data = {}
    for c in CHI:
        for q in QS:
            data[(q, c)] = curve(th, q, c)
            print(f"  q={q:.0f} chi1={c:+.1f}  E={data[(q,c)][3]:.4e}", flush=True)

    for c in CHI:
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4))
        draw(axa, axb, data, c)
        axa.set_title(r"$\alpha(t)$ — master"); axb.set_title(r"$\beta(t)$ — master")
        fig.suptitle(r"fluxanchored + spin (9 coef, fitted on the 2d grid): master "
                     r"$\alpha/\beta$ across $q$ at $\chi_1=" + f"{c:+.1f}$")
        fig.tight_layout()
        stem = f"fluxanchored_spin_alpha_beta_overlay_chi{c:+.1f}".replace("+0.0","0.0").replace("+","p").replace("-","m")
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{stem}.{ext}"); print("wrote", OUT / f"{stem}.{ext}")
        plt.close(fig)

    fig, axes = plt.subplots(len(CHI), 2, figsize=(12, 4 * len(CHI)))
    for i, c in enumerate(CHI):
        draw(axes[i][0], axes[i][1], data, c, legend=(i == 0))
        axes[i][0].set_title(r"$\alpha(t)$,  $\chi_1=" + f"{c:+.1f}$"
                            + ("   [reference: no spin terms]" if c == 0 else ""))
        axes[i][1].set_title(r"$\beta(t)$,  $\chi_1=" + f"{c:+.1f}$")
    fig.suptitle(r"fluxanchored + spin: master $\alpha/\beta$, "
                 r"$\chi_1 \in \{-0.6,-0.2,+0.2,+0.6\}$")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fluxanchored_spin_alpha_beta_overlay_all.{ext}")
        print("wrote", OUT / f"fluxanchored_spin_alpha_beta_overlay_all.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
