"""
Plotting script for the gwr_energy_anchored model (6 coefficients, globally jointly fitted).

Run fit_scaling_gwr_energy_anchored.py --global --maxiter 80 first.
Plots written to Agentic_plots/gwr_energy_anchored/ (always saved).

Q_INPUTS = [2.0, 3.0, 5.0, 8.0]  — q=2 is the extrapolation test.

Format is deliberately IDENTICAL to NRBHP_gw_remnant_energy_plots.py: same figure
set, sizes, rcParams, colours and fontsizes, so the two models' figures can be laid
side by side.  The per-q model is the same `mult` form, so only the coefficient layer
differs: the four per-q parameters come from anch.params_at (alpha_PP and beta_PP on the
derived X1^(6/5) base, alpha_E = A0/nu, beta_E = P(nu)/beta_PP).

Model waveforms come from fit_scaling_gw_remnant_energy.evaluate_model, i.e. the same
evaluator that produced the mismatches in scaling_gwr_energy_anchored.md, so the figures
and the quoted numbers cannot drift apart.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT    = Path(__file__).resolve().parent
UT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UT_ROOT / "BHPTutils"))

import bhpt_utils
import fit_scaling_PN_opt_creative as creative
import fit_scaling_PN_opt_creative_q_dep as qdep
import fit_scaling_gw_remnant_energy as gwre
import fit_scaling_gwr_energy_anchored as anch

mpl.rcdefaults()
mpl.rcParams["text.usetex"] = False   # bhpt_utils re-enables usetex; latex not installed

FORM     = "mult"
Q_INPUTS = [2.0, 3.0, 5.0, 8.0]
PLOT_DIR = ROOT / "Agentic_plots" / "gwr_energy_anchored"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
COEFF_JSON = anch.COEFFS


def read_coefficients(path: Path = COEFF_JSON):
    d = json.loads(path.read_text())
    theta = np.concatenate([np.asarray(d["c"], float), np.asarray(d["A"], float),
                            [float(d["b"])], np.asarray(d["P"], float)])
    return (theta, int(d["alphaE_degree"])), int(d["n_coefficients"])


def build_model(q, coeffs):
    qdep.generate_and_cache_waveform(q)
    case   = gwre.load_case(q, source_stride=3, nr_stride=5)
    params = anch.params_at(q, coeffs[0], coeffs[1])
    params = gwre.polish_nuisance(params, case, FORM)
    ev = gwre.evaluate_model(
        params, case["t_bhpt"], case["h_bhpt"],
        case["t_nr"], case["h_nr"], case["losses"], FORM)
    return {"eval": ev, "params": params, "case": case}


def _extrap(q): return q < 3.0 or q > 8.0
def _tag(q):
    s = f"q{int(q)}" if q == int(q) else f"q{q}"
    return s + ("_extrap" if _extrap(q) else "")
def _title(q, error=float("nan")):
    ex = " [extrapolation]" if _extrap(q) else ""
    es = f";  mathcalE = {error:.3g}" if np.isfinite(error) else ""
    return f"gwr-energy-anchored  q={q}{ex};  nu={gwre.get_nu(q):.4f}{es}"


def save_waveform_plot(q, common_t, h_nr, h_model, error):
    split = -200.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 2]})
    for ax, xlim in [(ax1, (-1000, split)), (ax2, (split, 100))]:
        ax.plot(common_t, np.real(h_nr),    label="NR 22 mode")
        ax.plot(common_t, np.real(h_model), label="BHPT gwr-energy-anchored", alpha=0.8)
        ax.set_xlim(*xlim); ax.grid(True); ax.set_xlabel("t/M")
    ax1.legend(fontsize=8)
    fig.suptitle(_title(q, error)); fig.tight_layout()
    path = PLOT_DIR / f"gwr_anch_{_tag(q)}_waveform.pdf"
    fig.savefig(path); plt.close(fig); return path


def save_zoomed_plot(q, common_t, h_nr, h_model):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(common_t, np.real(h_nr),    label="NR 22 mode")
    ax.plot(common_t, np.real(h_model), label="BHPT gwr-energy-anchored", alpha=0.8)
    ax.set_xlim(-100, 100); ax.grid(True); ax.set_xlabel("t/M"); ax.legend(fontsize=8)
    ax.set_title(_title(q)); fig.tight_layout()
    path = PLOT_DIR / f"gwr_anch_{_tag(q)}_zoomed.pdf"
    fig.savefig(path); plt.close(fig); return path


# --- parameter panels -------------------------------------------------------
# These keep their own look (stacked alpha / beta / E on a shared time axis) rather
# than the 1x3 house row: E(t) is the whole point of this model, and showing the
# three stacked makes it obvious that E supplies the merger ramp the logistic gate
# used to fake.  Styling is applied in a local rc_context so it cannot leak into the
# waveform/zoom figures, which do follow the house format.

_PARAM_RC = {
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": "#6b6a63", "axes.labelcolor": "#1a1a19", "text.color": "#1a1a19",
    "xtick.color": "#6b6a63", "ytick.color": "#6b6a63",
    "axes.linewidth": 0.6, "lines.linewidth": 1.3,
    "figure.dpi": 140, "savefig.bbox": "tight", "text.usetex": False,
}
C_ALPHA, C_BETA, C_E = "#eb6834", "#2a78d6", "#4a3aa7"
MUTED, GRID = "#6b6a63", "#e3e2dd"


def _param_style(ax):
    ax.grid(True, color=GRID, lw=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save_parameter_plot(q, tau, alpha, beta, e_oft, error, t_lo, t_hi):
    head = (f"$q={q:g}$   $\\nu={gwre.get_nu(q):.4f}$   "
            f"$\\mathcal{{E}}={error:.3g}$"
            f"   ({'extrapolation' if _extrap(q) else 'training range [3, 8]'})")
    m = (tau >= t_lo) & (tau <= t_hi)
    with plt.rc_context(_PARAM_RC):
        fig, ax = plt.subplots(3, 1, figsize=(9.0, 6.4), sharex=True)
        ax[0].plot(tau[m], alpha[m], color=C_ALPHA)
        ax[0].set_ylabel("$\\alpha(t)$")
        ax[1].plot(tau[m], beta[m], color=C_BETA)
        ax[1].set_ylabel("$\\beta(t)$")
        ax[2].plot(tau[m], e_oft[m], color=C_E)
        ax[2].set_ylabel("$E(t)\\ [M]$")
        ax[2].set_xlabel("$t\\ [M]$   (merger at $t=0$)")
        ax[0].set_title(head + "\n" + _master_row(q), pad=6, fontsize=8.5)
        for a in ax:
            _param_style(a)
            a.axvline(0.0, color=MUTED, lw=0.6, ls=(0, (4, 3)))
        fig.tight_layout()
        path = PLOT_DIR / f"gwr_anch_{_tag(q)}_params.pdf"
        fig.savefig(path)
        plt.close(fig)
    return path


def _master_row(q):
    p = _LAST_PARAMS[q]
    return (f"$\\alpha_{{\\rm PP}}={p[0]:.4f}$,  $\\alpha_E={p[1]:.3f}$,  "
            f"$\\beta_{{\\rm PP}}={p[2]:.4f}$,  $\\beta_E={p[3]:.3f}$")


_LAST_PARAMS: dict[float, np.ndarray] = {}


def save_coefficients_vs_q(coeffs, ncoef):
    cache = json.loads((gwre.RESULTS_DIR / "per_q_cache_mult.json").read_text())
    rows  = sorted(cache.values(), key=lambda r: r["q"])
    qs    = np.array([r["q"] for r in rows])
    P     = np.array([r["params"] for r in rows], float)
    q_fine = np.linspace(1.9, 8.3, 250)
    names  = ["alpha_PP", "alpha_E", "beta_PP", "beta_E"]
    master = np.array([anch.params_at(float(x), coeffs[0], coeffs[1])[:4]
                       for x in q_fine])

    fig, axes = plt.subplots(1, len(names), figsize=(15, 4))
    for i, (ax, name) in enumerate(zip(axes, names)):
        ax.axvspan(3.0, 8.0, color="0.9", zorder=0, label="training range")
        ax.plot(qs, P[:, i], "o", ms=3, label="per-q fits")
        ax.plot(q_fine, master[:, i], lw=1.5, label=f"{ncoef}-coef master")
        ax.axvline(gwre.BHPT_Q_MIN, color="k", ls=":", lw=0.8, label="BHPT domain q=2.5")
        ax.set_xlabel("q"); ax.set_ylabel(name); ax.grid(True)
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle("gwr-energy-anchored: master coefficients vs q")
    fig.tight_layout()
    path = PLOT_DIR / "gwr_anch_coefficients.pdf"
    fig.savefig(path); plt.close(fig); return path


def save_alpha_beta_overlay(results):
    cols = {2.0: "tab:red", 3.0: "tab:orange", 5.0: "tab:green", 8.0: "tab:blue"}
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4))
    for q in Q_INPUTS:
        r = results[q]; ev = r["eval"]; case = r["case"]
        tau = ev["tau"]; t_nr = case["t_nr"]
        m = (t_nr >= max(t_nr[0], tau[0])) & (t_nr <= min(t_nr[-1], tau[-1]))
        t = t_nr[m]; lw = 2.2 if q == 2 else 1.3
        axa.plot(t, np.interp(t, tau, ev["alpha"]), color=cols[q], lw=lw,
                 label=f"q={q:.0f}{' [extrap]' if q == 2 else ''}  E={ev['error']:.2e}")
        axb.plot(t, np.interp(t, tau, ev["beta"]), color=cols[q], lw=lw, label=f"q={q:.0f}")
    for ax, yl, tl in [(axa, "alpha(t)", "alpha(t) — master"), (axb, "beta(t)", "beta(t) — master")]:
        ax.axvline(0, color="gray", ls=":", lw=0.8); ax.set_xlim(-1000, 80); ax.grid(True)
        ax.set_xlabel("t_NR / M"); ax.set_ylabel(yl); ax.set_title(tl); ax.legend(fontsize=8)
    fig.suptitle("gwr-energy-anchored: master alpha/beta across q"); fig.tight_layout()
    path = PLOT_DIR / "gwr_anch_alpha_beta_overlay.pdf"
    fig.savefig(path); plt.close(fig); return path


def save_energy_vs_q(results):
    cols = {2.0: "tab:red", 3.0: "tab:orange", 5.0: "tab:green", 8.0: "tab:blue"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for q in Q_INPUTS:
        r = results[q]; ev = r["eval"]; case = r["case"]
        tau = ev["tau"]; e = case["losses"]["e_oft"]
        m = (tau >= -1000) & (tau <= 100)
        lw = 2.2 if q == 2 else 1.3
        ax1.plot(tau[m], e[m], color=cols[q], lw=lw, label=f"q={q:.0f}")
        ax2.plot(tau[m], e[m] / gwre.get_nu(q) ** 2, color=cols[q], lw=lw, label=f"q={q:.0f}")
    for ax, yl, tl in [(ax1, "E(t) [M]", "gw_remnant radiated energy"),
                       (ax2, "E(t) / nu^2", "scaled by nu^2 (E_tot ~ nu^2.31)")]:
        ax.axvline(0, color="gray", ls=":", lw=0.8); ax.set_xlim(-1000, 100); ax.grid(True)
        ax.set_xlabel("t_NR / M"); ax.set_ylabel(yl); ax.set_title(tl); ax.legend(fontsize=8)
    fig.suptitle("gwr-energy-anchored: the E(t) coordinate across q")
    fig.tight_layout()
    path = PLOT_DIR / "gwr_anch_Eoft_vs_q.pdf"
    fig.savefig(path); plt.close(fig); return path


def main():
    coeffs, ncoef = read_coefficients()
    results, errors, all_paths = {}, {}, []

    for q in Q_INPUTS:
        print(f"\n--- q={q}{' [EXTRAPOLATION]' if _extrap(q) else ''} ---", flush=True)
        r = build_model(q, coeffs)
        ev = r["eval"]; params = r["params"]; case = r["case"]
        errors[q] = ev["error"]; results[q] = r
        print(f"  nu = {gwre.get_nu(q):.4f}")
        for i, name in enumerate(gwre.FORMS[FORM]):
            print(f"  {name:<9} = {params[i]:.6g}")
        print(f"  mathcalE = {ev['error']:.6g}")
        common_t = case["t_nr"][ev["common"]]
        _LAST_PARAMS[q] = params
        all_paths += [
            save_waveform_plot(q, common_t, ev["h_ref"], ev["h_model"], ev["error"]),
            save_parameter_plot(q, ev["tau"], ev["alpha"], ev["beta"],
                                case["losses"]["e_oft"], ev["error"],
                                common_t[0], common_t[-1]),
            save_zoomed_plot(q, common_t, ev["h_ref"], ev["h_model"]),
        ]

    all_paths += [save_coefficients_vs_q(coeffs, ncoef),
                  save_alpha_beta_overlay(results),
                  save_energy_vs_q(results)]

    print("\n=== Summary ===")
    for q in Q_INPUTS:
        flag = " [extrap]" if _extrap(q) else "        "
        print(f"  q={q:<4}{flag}  nu={gwre.get_nu(q):.4f}  mathcalE={errors[q]:.4e}")
    print(f"\nPlots: {PLOT_DIR}")
    for p in all_paths:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
