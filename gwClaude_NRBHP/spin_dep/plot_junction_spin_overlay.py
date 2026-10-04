"""alpha/beta overlays for the C1 JUNCTION model (7 coefficients, 2d grid).

House format, matching plot_energy_anchored_spin_overlay.py: 1 x 2 (alpha left, beta
right), figsize (12, 4), q in {2,3,5,8} in tab:red/orange/green/blue, lw 2.2 for the
q = 2 extrapolation, xlim (-1000, 80), default rcParams, mathtext labels.  One figure
per chi1, plus a contact sheet, plus the 2 x 2 q-grid with all chi1 per panel.

WHAT THESE SHOW THAT THE OTHER MODELS' PLOTS DO NOT.  This model has no fitted merger-
ringdown branch: alpha and beta are matched C1 at t_* = the |h| peak onto a branch whose
shape is fixed by QNM theory.  Three derived quantities are therefore overlaid as
predictions, not fits:

  beta_r   dashed horizontal -- the QNM time-stretch anchor beta must settle onto.
           Derived from w220(chi1) and the remnant M_f, chi_f; verified against the
           MEASURED omega_pp/omega_NR ratio to 0.05-1.3%.
  t_*      vertical line -- the junction.  alpha and beta are continuous WITH their
           first derivatives there by construction, so the handover should be
           invisible in the curve; a visible kink would mean the matching is wrong.
  kappa    alpha's derived ringdown log-slope, (w_pp/2)(1/Q_pp - 1/Q_f).  beta matches
           the two QNM REAL frequencies and cannot also match the imaginary parts, so
           kappa is the leftover decay-rate mismatch alpha must carry.  Shown as a
           dotted tangent from t_*; alpha should follow it once the transient dies.
           Measured/derived agreed in 21/21 clean grid points (median 1.09).

q = 2 is labelled [extrap: nu AND surrogate]: outside the fit's training range q in
[3,8] AND outside BHPTNRSur2dq1e3's own domain (q_min = 3).

Reads junction_spin_results/coeffs_junction.json (7 coef, corrected tau and W_SCHW).
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")
import fit_pn_anchored_spin as S
import fit_junction_spin as J
import junction_diag as JD
import fit_scaling_PN_opt_creative as creative

mpl.rcdefaults(); mpl.rcParams["text.usetex"] = False
ROOT = HERE / "Agentic_spin_plots" / "junction_spin"
(ROOT / "chi_const").mkdir(parents=True, exist_ok=True)
(ROOT / "q_const").mkdir(parents=True, exist_ok=True)
QS = [2.0, 3.0, 5.0, 8.0]
CHI = [-0.6, -0.2, 0.0, 0.2, 0.6]
COLS = {2.0: "tab:red", 3.0: "tab:orange", 5.0: "tab:green", 8.0: "tab:blue"}
CHI_COLS = {-0.6: "tab:purple", -0.2: "tab:blue", 0.0: "k", 0.2: "tab:green",
            0.6: "tab:orange"}


def curve(th, q, chi1):
    case = S.build_case(q, chi1)
    t = np.asarray(case["t_bhpt"], float); h = case["h_bhpt"]
    x = np.asarray(case["x"], float)
    i_star = int(np.argmax(np.abs(h)))
    alpha, beta = J.model_ab(th, q, chi1, t, x, i_star)
    bc = creative.cumulative_trapezoid(beta, t)
    ts = bc - float(np.interp(S.T_ANCHOR, t, bc))
    f = lambda t0: S._err_at_t0(t0, alpha, ts, t, h, case["t_nr"], case["h_nr"])
    g = np.linspace(-250.0, 150.0, 81)
    t0 = float(g[int(np.argmin([f(v) for v in g]))])
    r = minimize_scalar(f, bounds=(t0 - 60, t0 + 60), method="bounded",
                        options={"xatol": 1e-3})
    tau = float(r.x) + ts
    d = JD.derived(q, chi1)
    return dict(tau=tau, alpha=alpha, beta=beta, err=float(r.fun),
                t_nr=case["t_nr"], t_star=float(tau[i_star]),
                a_star=float(alpha[i_star]), beta_r=d["beta_r"], kappa=d["kappa"],
                beta_r_scale=float(tau[i_star + 1] - tau[i_star]) /
                float(t[i_star + 1] - t[i_star]))


def _panel(axa, axb, data, keys, colour, label, mark_derived=True):
    for k in keys:
        c = data[k]
        tau, t_nr = c["tau"], c["t_nr"]
        m = (t_nr >= max(t_nr[0], tau[0])) & (t_nr <= min(t_nr[-1], tau[-1]))
        t = t_nr[m]
        col = colour(k); lw = 2.2 if (isinstance(k, tuple) and k[0] < 3.0) else 1.3
        axa.plot(t, np.interp(t, tau, c["alpha"]), color=col, lw=lw, label=label(k))
        axb.plot(t, np.interp(t, tau, c["beta"]), color=col, lw=lw, label=label(k))
        if mark_derived:
            # beta_r is on the pp time axis; beta itself is dtau/dt, so plot directly
            axb.axhline(c["beta_r"], color=col, ls="--", lw=0.9, alpha=0.7)
            axa.axvline(c["t_star"], color=col, ls=":", lw=0.8, alpha=0.6)
            axb.axvline(c["t_star"], color=col, ls=":", lw=0.8, alpha=0.6)
            # derived ringdown tangent for alpha, from t_*
            dt = np.linspace(0, 60, 40)
            axa.plot(c["t_star"] + dt, c["a_star"] * np.exp(c["kappa"] * dt),
                     color=col, ls=":", lw=1.0, alpha=0.8)
    for ax, yl in ((axa, r"$\alpha(t)$"), (axb, r"$\beta(t)$")):
        ax.set_xlim(-1000, 80); ax.grid(True)
        ax.set_xlabel(r"$t_{\rm NR}\,/\,M$"); ax.set_ylabel(yl)
        ax.legend(fontsize=7)


def main():
    th = np.array(json.loads(
        (HERE / "junction_spin_results" / "coeffs_junction.json").read_text())["theta"],
        float)
    print("theta:", np.round(th, 4), flush=True)
    data = {}
    for q in QS:
        for c in CHI:
            data[(q, c)] = curve(th, q, c)
            print(f"  q={q:g} chi1={c:+.1f}  E={data[(q,c)]['err']:.4e}  "
                  f"t_*={data[(q,c)]['t_star']:+.1f}  beta_r={data[(q,c)]['beta_r']:.4f}",
                  flush=True)

    # --- one figure per chi1 -------------------------------------------------
    for c in CHI:
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4))
        _panel(axa, axb, data, [(q, c) for q in QS], lambda k: COLS[k[0]],
               lambda k: f"q={k[0]:.0f}"
                         + (" [extrap: nu AND surrogate]" if k[0] < 3 else "")
                         + f"  E={data[k]['err']:.2e}")
        fig.suptitle(f"junction (7 coef), " + r"$\chi_1=" + f"{c:+.1f}$"
                     + "   dashed: derived $\\beta_r$   dotted: $t_*$ and "
                       "$e^{\\kappa\\Delta t}$")
        fig.tight_layout()
        tag = f"chi{'m' if c < 0 else ('p' if c > 0 else '')}{abs(c):.1f}"
        for ext in ("pdf", "png"):
            p = ROOT / "chi_const" / f"junction_spin_alpha_beta_overlay_{tag}.{ext}"
            fig.savefig(p); print("wrote", p, flush=True)
        plt.close(fig)

    # --- 2 x 2 q-grid, all chi1 per panel ------------------------------------
    for idx, sym, nm in ((0, r"\alpha", "alpha"), (1, r"\beta", "beta")):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, q in zip(axes.flat, QS):
            for c in CHI:
                d = data[(q, c)]
                tau, t_nr = d["tau"], d["t_nr"]
                m = (t_nr >= max(t_nr[0], tau[0])) & (t_nr <= min(t_nr[-1], tau[-1]))
                t = t_nr[m]
                y = (d["alpha"], d["beta"])[idx]
                ax.plot(t, np.interp(t, tau, y), color=CHI_COLS[c],
                        ls="--" if c == 0.0 else "-", lw=1.6 if c == 0.0 else 1.3,
                        label=r"$\chi_1=" + f"{c:+.1f}$" + f"  E={d['err']:.2e}")
                if idx == 1:
                    ax.axhline(d["beta_r"], color=CHI_COLS[c], ls=":", lw=0.8, alpha=0.6)
            ax.axvline(0, color="gray", ls=":", lw=0.8); ax.set_xlim(-1000, 80)
            ax.grid(True); ax.set_xlabel(r"$t_{\rm NR}\,/\,M$")
            ax.set_ylabel(f"${sym}(t)$")
            ax.set_title(f"$q={q:.0f}$" + ("  [extrapolated]" if q < 3 else ""))
            ax.legend(fontsize=7)
        fig.suptitle(f"junction (7 coef): master ${sym}(t)$ across " + r"$\chi_1$"
                     + ("   dotted: derived $\\beta_r$" if idx == 1 else ""))
        fig.tight_layout()
        for ext in ("pdf", "png"):
            p = ROOT / "q_const" / f"junction_spin_{nm}_overlay_q_grid.{ext}"
            fig.savefig(p); print("wrote", p, flush=True)
        plt.close(fig)

    # --- does beta actually settle on the derived anchor? --------------------
    print(f"\n  {'q':>4} {'chi1':>6} {'beta late':>10} {'beta_r':>9} {'err%':>7}")
    for q in QS:
        for c in CHI:
            d = data[(q, c)]
            t = np.asarray(S.build_case(q, c)["t_bhpt"], float)
            i = int(np.argmax(np.abs(S.build_case(q, c)["h_bhpt"])))
            late = d["beta"][min(i + 600, len(d["beta"]) - 1)]
            print(f"  {q:>4g} {c:>+6.1f} {late:>10.4f} {d['beta_r']:>9.4f} "
                  f"{100*(late/d['beta_r']-1):>+7.2f}")


if __name__ == "__main__":
    main()
