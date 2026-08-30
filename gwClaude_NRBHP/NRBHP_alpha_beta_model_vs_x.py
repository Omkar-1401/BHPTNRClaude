"""MODEL alpha(x), beta(x) against the PARAMETER-FREE measurement, one curve per q.

`NRBHP_alpha_beta_vs_x.py` plots the measured curves alone.  This adds the model on the
same axes, so the question "does the model reproduce the measured x-dependence" is
answerable by eye instead of only through an integrated mismatch.

COORDINATE.  x = (0.5 * omega_GW,pp)^(2/3) = (M omega_orb,pp)^(2/3), computed from the
ppBHPT waveform alone -- unwrap the phase, differentiate with a Savitzky-Golay filter
whose window is set in PHYSICAL TIME (M) rather than samples.  This is the SAME
definition and the same construction as beta_drift_tests/alpha_beta_of_omega_v2.py
(`x = (0.5*w_p)**(2.0/3.0)`), so the two are directly comparable.

Using the ppBHPT's OWN frequency is essential and is not a detail: x_NR = beta^(-2/3)
x_pp, so a model parameterised by x_NR would need the answer to evaluate its own
coordinate.  x_pp is known from the surrogate before anything is fitted.

WHAT IS PLOTTED.  Model curves are evaluated on the BHPT time grid and plotted
parametrically against x(t) -- the same t is the parameter for both axes, so each curve
traces the inspiral from low x up to merger.  The measurement (thin black) comes from
alpha_beta_of_omega_v2.json, merger-aligned, nothing fitted, matched smoothing in M.

House format, matching NRBHP_alpha_beta_vs_x.py: 1 x 2, figsize (12,4), q=2/3/5/8 in
tab:red/orange/green/blue with lw 2.2 for q=2, intermediates q=2.5/4/6 thin grey, dashed
horizontals in the beta panel at X1^(6/5).  rcdefaults() AFTER the model imports.

Also writes the Ehat(x) universality figure: Ehat = E(t)/E_tot against x, one curve per q.
That answers whether the energy drive and the frequency coordinate are related
q-universally -- i.e. whether an E-driven model already IS an x-model with a fixed shape
function, or whether the shape slides with q.

Usage:
    python NRBHP_alpha_beta_model_vs_x.py                      # E_anchored + flux_anchored
    python NRBHP_alpha_beta_model_vs_x.py --model E_anchored inspiral_anchored
    python NRBHP_alpha_beta_model_vs_x.py --window-M 40
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
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import fit_scaling_gw_remnant_energy as gwre
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_hm_backbone as hm

mpl.rcdefaults()                       # AFTER the imports -- gw_remnant sets 18pt STIX
mpl.rcParams["text.usetex"] = False

PLOTS = ROOT / "Agentic_plots" / "alpha_beta_vs_x"
PLOTS.mkdir(parents=True, exist_ok=True)
MEASURED = ROOT / "beta_drift_tests" / "alpha_beta_of_omega_v2.json"

MAIN = {2.0: "tab:red", 3.0: "tab:orange", 5.0: "tab:green", 8.0: "tab:blue"}
SIDE = [2.5, 4.0, 6.0]
QS = sorted(list(MAIN) + SIDE)


def x_of_t(t, h, window_M=20.0):
    """x = (0.5 * omega_GW)^(2/3) from the unwrapped phase, savgol window set in M."""
    dt = float(np.median(np.diff(t)))
    n = int(round(window_M / dt))
    if n % 2 == 0:
        n += 1
    n = max(n, 7)
    psi = np.unwrap(np.angle(h))
    # savgol's own analytic derivative, not gradient-of-smoothed
    w = savgol_filter(psi, n, 3, deriv=1, delta=dt)
    w = np.abs(w)
    return np.where(w > 0, (0.5 * w) ** (2.0 / 3.0), np.nan)


def model_curves(q, coeffs, window_M):
    """alpha(t), beta(t), x(t) and Ehat(t) for one q on the BHPT grid."""
    case = FL.add_flux(gwre.load_case(q, 3, 5))
    p = np.asarray(coeffs["params_at"](q), float)[:4]
    los = case["losses"]
    alpha = p[0] * (1.0 + p[1] * los[coeffs["coord"]])
    beta = p[2] * (1.0 + p[3] * los["e_oft"])
    e = np.asarray(los["e_oft"], float)
    e_tot = float(e[-1] - e[0])
    x = x_of_t(np.asarray(case["t_bhpt"], float), case["h_bhpt"], window_M)
    return {"x": x, "alpha": alpha, "beta": beta,
            "Ehat": (e - e[0]) / e_tot if e_tot > 0 else e * np.nan,
            "anchor": (q / (1.0 + q)) ** 1.2}


def measured(window_M):
    """{q: {x, alpha, beta}} from the parameter-free merger-aligned measurement."""
    if not MEASURED.exists():
        return {}
    d = json.loads(MEASURED.read_text())
    key = str(int(window_M)) if str(int(window_M)) in d else sorted(d)[0]
    return {float(k): v for k, v in d[key].items()}


def draw(model, res, meas, window_M):
    label = hm.MODEL_LABELS.get(model, model)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4))

    def curve(ax, q, key, color, lw, zorder, label=None):
        r = res[q]
        m = np.isfinite(r["x"]) & np.isfinite(r[key])
        ax.plot(r["x"][m], r[key][m], color=color, lw=lw, zorder=zorder, label=label)

    for q in SIDE:
        for ax, key in ((axa, "alpha"), (axb, "beta")):
            curve(ax, q, key, "0.65", 0.8, 1, label=f"$q={q:g}$")
    for q, c in MAIN.items():
        lw = 2.2 if q == 2 else 1.3
        for ax, key in ((axa, "alpha"), (axb, "beta")):
            curve(ax, q, key, c, lw, 3, label=f"$q={q:g}$")
        axb.axhline(res[q]["anchor"], color=c, lw=0.7, ls=(0, (4, 3)), zorder=2)

    # the parameter-free measurement, thin black, on top
    first = True
    for q in QS:
        mq = meas.get(q)
        if mq is None:
            continue
        for ax, key in ((axa, "alpha"), (axb, "beta")):
            ax.plot(mq["x"], mq[key], color="k", lw=0.7, ls=":", zorder=4,
                    label=("measured (parameter-free)" if first and ax is axa else None))
        first = False

    # "master", not "per-q": these are the nu-REGRESSED coefficients evaluated at each q.
    # Nothing here is fitted to its own q's data -- cf. NRBHP_gwr_anch_perq_overlay.py,
    # which is the per-q-optimal counterpart and where "PER-Q" means something else.
    for ax, yl, tl in ((axa, r"$\alpha$",
                        r"$\alpha$ vs $x$ — MASTER (colour) vs measurement (dotted)"),
                       (axb, r"$\beta$",
                        r"$\beta$ vs $x$ — MASTER   (dashed: $X_1^{6/5}$)")):
        ax.set_xlabel(r"$x = (M\,\Omega_{\mathrm{orb,pp}})^{2/3}$")
        ax.set_ylabel(yl)
        ax.set_title(tl, fontsize=10)
        ax.grid(True)
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0 - 0.30 * (y1 - y0), y1)
        ax.legend(fontsize=7, ncol=2, loc="lower left", framealpha=0.92)
    fig.suptitle(f"{label}: MASTER $\\alpha$, $\\beta$ vs $x$, across $q$  "
                 f"($\\nu$-regressed coefficients, no per-q fitting)  "
                 f"[savgol window {window_M:g} $M$]")
    fig.tight_layout()
    out = []
    for ext in ("pdf", "png"):
        p = PLOTS / f"alpha_beta_vs_x_{model}.{ext}"
        fig.savefig(p); out.append(p)
    plt.close(fig)
    return out


def draw_ehat(res, window_M):
    """Ehat(x): is the energy drive a q-universal function of the frequency coordinate?"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for q in SIDE:
        r = res[q]; m = np.isfinite(r["x"])
        ax.plot(r["x"][m], r["Ehat"][m], color="0.65", lw=0.8, label=f"$q={q:g}$")
    for q, c in MAIN.items():
        r = res[q]; m = np.isfinite(r["x"])
        ax.plot(r["x"][m], r["Ehat"][m], color=c, lw=2.2 if q == 2 else 1.3,
                label=f"$q={q:g}$")
    ax.set_xlabel(r"$x = (M\,\Omega_{\mathrm{orb,pp}})^{2/3}$")
    ax.set_ylabel(r"$\hat{E} = E(t)\,/\,E_{\mathrm{tot}}$")
    ax.set_title(r"Is the energy drive a $q$-universal function of $x$?", fontsize=10)
    ax.grid(True); ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    out = []
    for ext in ("pdf", "png"):
        p = PLOTS / f"Ehat_vs_x.{ext}"
        fig.savefig(p); out.append(p)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["E_anchored", "flux_anchored"])
    ap.add_argument("--window-M", type=float, default=20.0,
                    help="savgol window in M for omega; 10/20/40 available in the "
                         "measured file")
    args = ap.parse_args()

    meas = measured(args.window_M)
    print(f"measured curves available at q = {sorted(meas)}", flush=True)

    ehat_done = False
    for model in args.model:
        coeffs = hm.read_stiff_coeffs(model)
        print(f"\n[{model}] coord={coeffs['coord']}", flush=True)
        res = {}
        for q in QS:
            res[q] = model_curves(q, coeffs, args.window_M)
            r = res[q]
            xm = np.nanmax(r["x"])
            print(f"  q={q:<4g} x range {np.nanmin(r['x']):.4f}..{xm:.4f}  "
                  f"alpha {r['alpha'][0]:.4f} -> {r['alpha'][-1]:.4f}  "
                  f"beta {r['beta'][0]:.4f} -> {r['beta'][-1]:.4f}", flush=True)
        for p in draw(model, res, meas, args.window_M):
            print("wrote", p)
        if not ehat_done:
            for p in draw_ehat(res, args.window_M):
                print("wrote", p)
            # quantify the universality question numerically as well
            xs = np.linspace(0.10, 0.25, 40)
            band = []
            for xq in xs:
                vals = []
                for q in QS:
                    r = res[q]; m = np.isfinite(r["x"])
                    xx, ee = r["x"][m], r["Ehat"][m]
                    o = np.argsort(xx)
                    if xq < xx[o][0] or xq > xx[o][-1]:
                        continue
                    vals.append(float(np.interp(xq, xx[o], ee[o])))
                if len(vals) > 1:
                    band.append((xq, min(vals), max(vals)))
            if band:
                print("\nEhat spread across q at fixed x:")
                print(f"{'x':>7} {'min':>9} {'max':>9} {'spread':>9}")
                for xq, lo, hi in band[::6]:
                    print(f"{xq:7.3f} {lo:9.4f} {hi:9.4f} {hi-lo:9.4f}")
            ehat_done = True


if __name__ == "__main__":
    main()
