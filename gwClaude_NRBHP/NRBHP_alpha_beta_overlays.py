"""Per-model alpha/beta overlay figures, one per model, in that model's own folder.

Format is VERBATIM that of NRBHP_gwr_energy_stiff_plots.save_alpha_beta_overlay:
1 x 2 (alpha left, beta right), one curve per q in {2,3,5,8} coloured
tab:red/orange/green/blue, resampled onto the NR time grid, lw 2.2 for the q=2
extrapolation, xlim (-1000, 80), grid on, default rcParams, PDF.  No waveform panel.

  model                 file
  pn_anchored           Agentic_plots/pn_anchored/pn_anchored_alpha_beta_overlay.pdf
  gwr_energy_global     Agentic_plots/gwr_global_E/gwr_globalE_alpha_beta_overlay.pdf
  gwr_energy_anchored   Agentic_plots/gwr_energy_anchored/gwr_anch_alpha_beta_overlay.pdf
  gwr_energy_flux       Agentic_plots/gwr_global_flux/gwr_globalF_alpha_beta_overlay.pdf
  gwr_energy_fluxanch.  Agentic_plots/gwr_global_fluxanchored/gwr_globalFA_alpha_beta_overlay.pdf

Four of those five names already existed, written by the per-model plotting scripts
(NRBHP_gwr_energy_anchored_plots.py and NRBHP_gwr_global_plots.py, whose overlay
functions are line-for-line the same as stiff's); this script regenerates them from
one place and adds the missing pn_anchored one.

The four gwr models share the per-q family

    alpha(t) = alpha_PP(q) * (1 + a_c(q) * X(t))     X = E(t) or F(t)=Edot/max(Edot)
    beta (t) = beta_PP (q) * (1 + b_c(q) * E(t))

so one evaluator serves all four (generalised over alpha's coordinate, exactly as
NRBHP_gwr_global_plots.py does); pn_anchored has its own gated form and supplies
alpha/beta through BHPTNRPNAnchored.get_alpha_beta.  All gwr models use their
GLOBAL (jointly fitted) coefficients.  pn_anchored carries no mismatch in its
legend: it uses the phi0 = 0 convention and is not scored by this evaluator.

NOTE the rcdefaults() ordering: it MUST come after the model imports, because this
chain pulls in gw_remnant, whose gw_utils/gw_plotter.py sets font.size=18,
font.family='STIXGeneral' and figure.figsize=(14,10) at import time.

Usage:  python NRBHP_alpha_beta_overlays.py [--model ...] [--png]
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "BHPTutils"))

import fit_scaling_PN_opt_creative as creative
import fit_scaling_PN_opt_creative_q_dep as qdep
import fit_scaling_gw_remnant_energy as gwre
import fit_scaling_gwr_energy_global as GG
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_gwr_energy_anchored as AN
import fit_scaling_gwr_energy_fluxanchored as FA
import BHPTNRPNAnchored as PNA
import fit_scaling_gwr_beta_monotone as BM

mpl.rcdefaults()                      # AFTER the imports -- see module docstring
mpl.rcParams["text.usetex"] = False

Q_INPUTS = [2.0, 3.0, 5.0, 8.0]
PLOTS = ROOT / "Agentic_plots"

SPECS = {
    "pn_anchored": dict(
        label="pn-anchored", dirname="pn_anchored",
        fname="pn_anchored_alpha_beta_overlay.pdf", coord=None),
    "E_deg3": dict(
        label="gwr-energy-global", dirname="gwr_global_E",
        fname="gwr_globalE_alpha_beta_overlay.pdf", coord="e_oft"),
    "E_anchored": dict(
        label="gwr-energy-anchored", dirname="gwr_energy_anchored",
        fname="gwr_anch_alpha_beta_overlay.pdf", coord="e_oft"),
    "flux_deg3": dict(
        label="gwr-energy-flux", dirname="gwr_global_flux",
        fname="gwr_globalF_alpha_beta_overlay.pdf", coord="flux_hat"),
    "flux_anchored": dict(
        label="gwr-energy-fluxanchored", dirname="gwr_global_fluxanchored",
        fname="gwr_globalFA_alpha_beta_overlay.pdf", coord="flux_hat"),
    "beta_monotone": dict(
        label="gwr-beta-monotone, Edot alpha (P>=0, beta never decreases)",
        dirname="gwr_beta_monotone",
        fname="gwr_beta_monotone_alpha_beta_overlay.pdf", coord="flux_hat"),
    "beta_monotone_E": dict(
        label="gwr-beta-monotone, E alpha (P>=0, beta never decreases)",
        dirname="gwr_beta_monotone",
        fname="gwr_beta_monotone_Ebase_alpha_beta_overlay.pdf", coord="e_oft"),
}
GWR = ("E_deg3", "E_anchored", "flux_deg3", "flux_anchored",
       "beta_monotone", "beta_monotone_E")


def load_params_at():
    """key -> params_at(q) giving [a_pp, a_c, b_pp, b_c, ...]."""
    out = {}
    d = json.loads((ROOT / "gwr_energy_global_results" / "coeffs.json").read_text())
    cf = {k: np.asarray(v, float) for k, v in d.items() if not k.startswith("_")}
    out["E_deg3"] = lambda q: GG.params_at(q, cf)

    d = json.loads((FL.RESULTS / "coeffs.json").read_text())["coeffs"]
    cf2 = {k: np.asarray(v, float) for k, v in d.items()}
    out["flux_deg3"] = lambda q: FL.params_at(q, cf2)

    d = json.loads((AN.RESULTS / "coeffs.json").read_text())
    th = np.concatenate([np.asarray(d["c"], float), np.asarray(d["A"], float),
                         [float(d["b"])], np.asarray(d["P"], float)])
    deg = int(d["alphaE_degree"])
    out["E_anchored"] = lambda q: AN.params_at(q, th, deg)

    d = json.loads((FA.RESULTS / "coeffs.json").read_text())
    if not d.get("global_refit", False):
        print("  WARNING: fluxanchored coeffs.json holds SEEDED values "
              "(global_refit=false) -- rerun with --global for the joint fit.")
    th2 = np.concatenate([np.asarray(d["c"], float), np.asarray(d["A"], float),
                          [float(d["b"])], np.asarray(d["P"], float)])
    deg2 = int(d["aF_degree"])
    out["flux_anchored"] = lambda q: FA.params_at(q, th2, deg2)

    # beta_monotone: same 7-coefficient form as fluxanchored except
    # P(nu) = |a|*max(0, nu-nu_c) >= 0, so beta never decreases at any q.
    for key, fn, bs in (("beta_monotone", "coeffs.json", "flux"),
                        ("beta_monotone_E", "coeffs_Ebase.json", "E")):
        f = BM.RESULTS / fn
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        th3 = np.concatenate([np.asarray(d["c"], float), np.asarray(d["A"], float),
                              [float(d["b"]), float(d["P_a"]), float(d["P_nu_c"])]])
        # na = number of alpha coefficients; params_at wants the flux-convention degree,
        # which is na for the flux base and na - 1 for the E base.  Accept either the
        # new "alpha_deg" key or the original "aF_degree" one.
        na = int(d.get("alpha_deg", d.get("aF_degree")))
        adeg = na if bs == "flux" else na - 1
        out[key] = (lambda th, ad, b: lambda q: BM.params_at(q, th, ad, b))(th3, adeg, bs)
    return out


def evaluate(p, case, coord, t0):
    """gwre.evaluate_model generalised over alpha's coordinate (mirrors the
    NRBHP_gwr_global_plots.py evaluator exactly)."""
    a_pp, a_c, b_pp, b_c = p[:4]
    los = case["losses"]
    alpha = a_pp * (1.0 + a_c * los[coord])
    beta = b_pp * (1.0 + b_c * los["e_oft"])
    if np.any(beta <= 0) or not np.all(np.isfinite(np.r_[alpha, beta])):
        return {"error": 50.0}
    bc = creative.cumulative_trapezoid(beta, case["t_bhpt"])
    tau = t0 + bc - float(np.interp(gwre.T_ANCHOR, case["t_bhpt"], bc))
    if not np.all(np.diff(tau) > 0):
        return {"error": 50.0}
    t_nr, h_nr = case["t_nr"], case["h_nr"]
    msk = (t_nr >= max(t_nr[0], tau[0])) & (t_nr <= min(t_nr[-1], tau[-1]))
    tc, href = t_nr[msk], h_nr[msk]
    h0 = np.interp(tc, tau, alpha) * (np.interp(tc, tau, case["h_bhpt"].real)
                                      + 1j * np.interp(tc, tau, case["h_bhpt"].imag))
    z = np.sum(href * h0.conjugate())
    n1 = np.sum(np.abs(href) ** 2); n2 = np.sum(np.abs(h0) ** 2)
    return {"error": float((n1 + n2 - 2.0 * abs(z)) / (2.0 * n1)),
            "tau": tau, "alpha": alpha, "beta": beta}


def save_alpha_beta_overlay(key, curves, png=False):
    """Verbatim the format of NRBHP_gwr_energy_stiff_plots.save_alpha_beta_overlay:
    tab: colours keyed on q, figsize (12, 4), curves resampled onto the NR time grid,
    lw 2.2 for the q=2 extrapolation, xlim (-1000, 80), default rcParams."""
    S = SPECS[key]
    out_dir = PLOTS / S["dirname"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = {2.0: "tab:red", 3.0: "tab:orange", 5.0: "tab:green", 8.0: "tab:blue"}
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4))
    for q in Q_INPUTS:
        tau, alpha, beta, err, t_nr = curves[(key, q)]
        m = (t_nr >= max(t_nr[0], tau[0])) & (t_nr <= min(t_nr[-1], tau[-1]))
        t = t_nr[m]; lw = 2.2 if q == 2 else 1.3
        lab = f"q={q:.0f}{' [extrap]' if q == 2 else ''}"
        if np.isfinite(err):
            lab += f"  E={err:.2e}"
        axa.plot(t, np.interp(t, tau, alpha), color=cols[q], lw=lw, label=lab)
        axb.plot(t, np.interp(t, tau, beta), color=cols[q], lw=lw, label=f"q={q:.0f}")
    for ax, yl, tl in [(axa, "alpha(t)", "alpha(t) — master"),
                       (axb, "beta(t)", "beta(t) — master")]:
        ax.axvline(0, color="gray", ls=":", lw=0.8); ax.set_xlim(-1000, 80); ax.grid(True)
        ax.set_xlabel("t_NR / M"); ax.set_ylabel(yl); ax.set_title(tl); ax.legend(fontsize=8)
    fig.suptitle(f"{S['label']}: master alpha/beta across q"); fig.tight_layout()
    paths = [out_dir / S["fname"]]
    if png:
        paths.append(out_dir / S["fname"].replace(".pdf", ".png"))
    for p in paths:
        fig.savefig(p)
    plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=list(SPECS), choices=list(SPECS))
    ap.add_argument("--png", action="store_true",
                    help="also write a PNG beside the PDF (for quick viewing)")
    args = ap.parse_args()

    want = list(args.model)
    params_at = load_params_at()
    curves = {}

    for q in Q_INPUTS:
        print(f"q={q:g}", flush=True)
        qdep.generate_and_cache_waveform(q)
        case = FL.add_flux(gwre.load_case(q, source_stride=3, nr_stride=5))
        t_nr = case["t_nr"]
        for key in [k for k in want if k in GWR]:
            p = params_at[key](q)
            f = lambda t0: evaluate(p, case, SPECS[key]["coord"], t0)["error"]
            t0 = float(minimize_scalar(f, bounds=(-200.0, 40.0), method="bounded",
                                       options={"xatol": 1e-3}).x)
            ev = evaluate(p, case, SPECS[key]["coord"], t0)
            curves[(key, q)] = (ev["tau"], ev["alpha"], ev["beta"], ev["error"], t_nr)
            print(f"   {key:14s} E={ev['error']:.4e}", flush=True)
        if "pn_anchored" in want:
            tau, al, be = PNA.get_alpha_beta(q)
            # no mismatch quoted: pn_anchored uses the phi0 = 0 convention and is not
            # scored by this evaluator.  See scaling_pn_anchored.md for its numbers.
            curves[("pn_anchored", q)] = (tau, al, be, np.nan, t_nr)
            print(f"   {'pn_anchored':14s} (own gated form)", flush=True)

    print()
    for key in want:
        for p in save_alpha_beta_overlay(key, curves, png=args.png):
            print("wrote", p)


if __name__ == "__main__":
    main()
