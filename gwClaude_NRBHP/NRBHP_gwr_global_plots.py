"""
Plotting script for the three GLOBAL (joint-fit) (2,2) models, side by side.

    --model E             gwr_energy_global        E-coupled alpha, uniform deg-3, 14 coef
    --model flux          gwr_energy_flux          Edot-coupled alpha, uniform deg-3, 14 coef
    --model fluxanchored  gwr_energy_fluxanchored  Edot-coupled alpha, PN-anchored, 7 coef

All three share the per-q model family

    alpha(t) = alpha_PP(q) * (1 + a_c(q) * X(t))      X = E(t) or F(t) = Edot/max(Edot)
    beta (t) = beta_PP (q) * (1 + b_c(q) * E(t))
    tau(t)   = t0_nr + integral_{T_ANCHOR}^{t} beta dt'

and differ only in alpha's coordinate and in the nu-regression layer, so one evaluator
serves all three and the three figure sets are directly comparable.  Only the GLOBAL
(jointly fitted) coefficients are used; the `seeded` variant of fluxanchored is ignored
here.

Plots written to Agentic_plots/gwr_global_<model>/ (always saved).
Q_INPUTS = [2.0, 3.0, 5.0, 8.0] -- q=2 is the extrapolation test.

Format follows NRBHP_gw_remnant_energy_plots.py exactly (same figure set, sizes,
rcParams, colours and fontsizes), so these can be laid beside the earlier models.

NOTE the rcdefaults() ordering: it must come AFTER the model imports, because this
chain pulls in `gw_remnant`, whose gw_utils/gw_plotter.py sets font.size=18,
font.family='STIXGeneral', axes.linewidth=1 and figure.figsize=(14,10) at import time.
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
UT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UT_ROOT / "BHPTutils"))

import bhpt_utils
import fit_scaling_PN_opt_creative as creative
import fit_scaling_PN_opt_creative_q_dep as qdep
import fit_scaling_gw_remnant_energy as gwre
import fit_scaling_gwr_energy_global as GG
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_gwr_energy_fluxanchored as FA

mpl.rcdefaults()
mpl.rcParams["text.usetex"] = False

Q_INPUTS = [2.0, 3.0, 5.0, 8.0]

MODELS = {
    "E": dict(
        label="gwr-energy-global", prefix="gwr_globalE", coord="e_oft",
        acoef="alpha_E", ncoef=14,
        perq=gwre.RESULTS_DIR / "per_q_cache_mult.json"),
    "flux": dict(
        label="gwr-energy-flux", prefix="gwr_globalF", coord="flux_hat",
        acoef="alpha_F", ncoef=14,
        perq=FL.RESULTS / "per_q_cache_flux.json"),
    "fluxanchored": dict(
        label="gwr-energy-fluxanchored", prefix="gwr_globalFA", coord="flux_hat",
        acoef="alpha_F", ncoef=7,
        perq=FL.RESULTS / "per_q_cache_flux.json"),
}


def load_model(name):
    m = dict(MODELS[name])
    if name == "E":
        d = json.loads((ROOT / "gwr_energy_global_results" / "coeffs.json").read_text())
        cf = {k: np.asarray(v, float) for k, v in d.items() if not k.startswith("_")}
        m["params_at"] = lambda q: GG.params_at(q, cf)
    elif name == "flux":
        d = json.loads((FL.RESULTS / "coeffs.json").read_text())["coeffs"]
        cf = {k: np.asarray(v, float) for k, v in d.items()}
        m["params_at"] = lambda q: FL.params_at(q, cf)
    else:
        d = json.loads((FA.RESULTS / "coeffs.json").read_text())
        th = np.concatenate([np.asarray(d["c"], float), np.asarray(d["A"], float),
                             [float(d["b"])], np.asarray(d["P"], float)])
        deg = int(d["aF_degree"])
        m["params_at"] = lambda q: FA.params_at(q, th, deg)
    return m


# --- one evaluator for all three ------------------------------------------------

def evaluate(p, case, coord, t0):
    """gwre.evaluate_model, generalised over alpha's coordinate."""
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
            "tau": tau, "alpha": alpha, "beta": beta, "h_ref": href,
            "h_model": h0 * np.exp(1j * float(np.angle(z))), "common": msk}


def build_model(q, m):
    qdep.generate_and_cache_waveform(q)
    case = FL.add_flux(gwre.load_case(q, source_stride=3, nr_stride=5))
    p = m["params_at"](q)
    f = lambda t0: evaluate(p, case, m["coord"], t0)["error"]
    t0 = float(minimize_scalar(f, bounds=(-200.0, 40.0), method="bounded",
                               options={"xatol": 1e-3}).x)
    ev = evaluate(p, case, m["coord"], t0)
    return {"eval": ev, "params": np.r_[p[:4], t0, 0.0], "case": case}


def _extrap(q): return q < 3.0 or q > 8.0
def _tag(q):
    s = f"q{int(q)}" if q == int(q) else f"q{q}"
    return s + ("_extrap" if _extrap(q) else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = ap.parse_args()
    for name in args.model:
        run(name)


def run(name):
    M = load_model(name)
    LABEL, PRE = M["label"], M["prefix"]
    PLOT_DIR = ROOT / "Agentic_plots" / f"gwr_global_{name}"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    def _title(q, error=float("nan")):
        ex = " [extrapolation]" if _extrap(q) else ""
        es = f";  mathcalE = {error:.3g}" if np.isfinite(error) else ""
        return f"{LABEL}  q={q}{ex};  nu={gwre.get_nu(q):.4f}{es}"

    def save_waveform_plot(q, common_t, h_nr, h_model, error):
        split = -200.0
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                       gridspec_kw={"width_ratios": [3, 2]})
        for ax, xlim in [(ax1, (-1000, split)), (ax2, (split, 100))]:
            ax.plot(common_t, np.real(h_nr), label="NR 22 mode")
            ax.plot(common_t, np.real(h_model), label=f"BHPT {LABEL}", alpha=0.8)
            ax.set_xlim(*xlim); ax.grid(True); ax.set_xlabel("t/M")
        ax1.legend(fontsize=8)
        fig.suptitle(_title(q, error)); fig.tight_layout()
        path = PLOT_DIR / f"{PRE}_{_tag(q)}_waveform.pdf"
        fig.savefig(path); plt.close(fig); return path

    def save_zoomed_plot(q, common_t, h_nr, h_model):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(common_t, np.real(h_nr), label="NR 22 mode")
        ax.plot(common_t, np.real(h_model), label=f"BHPT {LABEL}", alpha=0.8)
        ax.set_xlim(-100, 100); ax.grid(True); ax.set_xlabel("t/M"); ax.legend(fontsize=8)
        ax.set_title(_title(q)); fig.tight_layout()
        path = PLOT_DIR / f"{PRE}_{_tag(q)}_zoomed.pdf"
        fig.savefig(path); plt.close(fig); return path

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

    def _master_row(q):
        p = _LAST[q]
        sym = r"\alpha_F" if M["coord"] == "flux_hat" else "\\alpha_E"
        return (f"$\\alpha_{{\\rm PP}}={p[0]:.4f}$,  ${sym}={p[1]:.3f}$,  "
                f"$\\beta_{{\\rm PP}}={p[2]:.4f}$,  $\\beta_E={p[3]:.3f}$")

    _LAST = {}

    def save_parameter_plot(q, tau, alpha, beta, drive, error, t_lo, t_hi):
        head = (f"$q={q:g}$   $\\nu={gwre.get_nu(q):.4f}$   "
                f"$\\mathcal{{E}}={error:.3g}$"
                f"   ({'extrapolation' if _extrap(q) else 'training range [3, 8]'})")
        mk = (tau >= t_lo) & (tau <= t_hi)
        ylab = ("$F(t)=\\dot{E}/\\max\\dot{E}$" if M["coord"] == "flux_hat"
                else "$E(t)\\ [M]$")
        with plt.rc_context(_PARAM_RC):
            fig, ax = plt.subplots(3, 1, figsize=(9.0, 6.4), sharex=True)
            ax[0].plot(tau[mk], alpha[mk], color=C_ALPHA)
            ax[0].set_ylabel("$\\alpha(t)$")
            ax[1].plot(tau[mk], beta[mk], color=C_BETA)
            ax[1].set_ylabel("$\\beta(t)$")
            ax[2].plot(tau[mk], drive[mk], color=C_E)
            ax[2].set_ylabel(ylab)
            ax[2].set_xlabel("$t\\ [M]$   (merger at $t=0$)")
            ax[0].set_title(head + "\n" + _master_row(q), pad=6, fontsize=8.5)
            for a in ax:
                _param_style(a)
                a.axvline(0.0, color=MUTED, lw=0.6, ls=(0, (4, 3)))
            fig.tight_layout()
            path = PLOT_DIR / f"{PRE}_{_tag(q)}_params.pdf"
            fig.savefig(path); plt.close(fig)
        return path

    def save_coefficients_vs_q():
        cache = json.loads(Path(M["perq"]).read_text())
        rows = sorted(cache.values(), key=lambda r: r["q"])
        qs = np.array([r["q"] for r in rows])
        P = np.array([r["params"] for r in rows], float)[:, :4]
        q_fine = np.linspace(1.9, 8.3, 250)
        names = ["alpha_PP", M["acoef"], "beta_PP", "beta_E"]
        master = np.array([M["params_at"](float(x))[:4] for x in q_fine])
        fig, axes = plt.subplots(1, len(names), figsize=(15, 4))
        for i, (ax, nm) in enumerate(zip(axes, names)):
            ax.axvspan(3.0, 8.0, color="0.9", zorder=0, label="training range")
            ax.plot(qs, P[:, i], "o", ms=3, label="per-q fits")
            ax.plot(q_fine, master[:, i], lw=1.5, label=f"{M['ncoef']}-coef master")
            ax.axvline(gwre.BHPT_Q_MIN, color="k", ls=":", lw=0.8,
                       label="BHPT domain q=2.5")
            ax.set_xlabel("q"); ax.set_ylabel(nm); ax.grid(True)
            if i == 0:
                ax.legend(fontsize=7)
        fig.suptitle(f"{LABEL}: master coefficients vs q"); fig.tight_layout()
        path = PLOT_DIR / f"{PRE}_coefficients.pdf"
        fig.savefig(path); plt.close(fig); return path

    def save_alpha_beta_overlay(results):
        cols = {2.0: "tab:red", 3.0: "tab:orange", 5.0: "tab:green", 8.0: "tab:blue"}
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4))
        for q in Q_INPUTS:
            r = results[q]; ev = r["eval"]; case = r["case"]
            tau = ev["tau"]; t_nr = case["t_nr"]
            mk = (t_nr >= max(t_nr[0], tau[0])) & (t_nr <= min(t_nr[-1], tau[-1]))
            t = t_nr[mk]; lw = 2.2 if q == 2 else 1.3
            axa.plot(t, np.interp(t, tau, ev["alpha"]), color=cols[q], lw=lw,
                     label=f"q={q:.0f}{' [extrap]' if q == 2 else ''}  E={ev['error']:.2e}")
            axb.plot(t, np.interp(t, tau, ev["beta"]), color=cols[q], lw=lw,
                     label=f"q={q:.0f}")
        for ax, yl, tl in [(axa, "alpha(t)", "alpha(t) — master"),
                           (axb, "beta(t)", "beta(t) — master")]:
            ax.axvline(0, color="gray", ls=":", lw=0.8); ax.set_xlim(-1000, 80)
            ax.grid(True); ax.set_xlabel("t_NR / M"); ax.set_ylabel(yl)
            ax.set_title(tl); ax.legend(fontsize=8)
        fig.suptitle(f"{LABEL}: master alpha/beta across q"); fig.tight_layout()
        path = PLOT_DIR / f"{PRE}_alpha_beta_overlay.pdf"
        fig.savefig(path); plt.close(fig); return path

    def save_drive_vs_q(results):
        cols = {2.0: "tab:red", 3.0: "tab:orange", 5.0: "tab:green", 8.0: "tab:blue"}
        flux = M["coord"] == "flux_hat"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        for q in Q_INPUTS:
            r = results[q]; ev = r["eval"]; case = r["case"]
            tau = ev["tau"]; d = case["losses"][M["coord"]]
            mk = (tau >= -1000) & (tau <= 100)
            lw = 2.2 if q == 2 else 1.3
            ax1.plot(tau[mk], d[mk], color=cols[q], lw=lw, label=f"q={q:.0f}")
            sc = d[mk] if flux else d[mk] / gwre.get_nu(q) ** 2
            ax2.plot(tau[mk], sc, color=cols[q], lw=lw, label=f"q={q:.0f}")
        t1 = ("F(t) = Edot / max(Edot)" if flux else "gw_remnant radiated energy")
        t2 = ("already peak-normalised" if flux else "scaled by nu^2 (E_tot ~ nu^2.31)")
        y1 = "F(t)" if flux else "E(t) [M]"
        y2 = "F(t)" if flux else "E(t) / nu^2"
        for ax, yl, tl in [(ax1, y1, t1), (ax2, y2, t2)]:
            ax.axvline(0, color="gray", ls=":", lw=0.8); ax.set_xlim(-1000, 100)
            ax.grid(True); ax.set_xlabel("t_NR / M"); ax.set_ylabel(yl)
            ax.set_title(tl); ax.legend(fontsize=8)
        fig.suptitle(f"{LABEL}: the alpha drive coordinate across q")
        fig.tight_layout()
        path = PLOT_DIR / f"{PRE}_drive_vs_q.pdf"
        fig.savefig(path); plt.close(fig); return path

    print(f"\n########## {LABEL} ({M['ncoef']} coef, alpha ~ {M['coord']}) ##########")
    results, errors, paths = {}, {}, []
    for q in Q_INPUTS:
        print(f"\n--- q={q}{' [EXTRAPOLATION]' if _extrap(q) else ''} ---", flush=True)
        r = build_model(q, M)
        ev = r["eval"]; params = r["params"]; case = r["case"]
        errors[q] = ev["error"]; results[q] = r
        _LAST[q] = params
        print(f"  nu = {gwre.get_nu(q):.4f}")
        for nm, v in zip(["alpha_PP", M["acoef"], "beta_PP", "beta_E", "t0_nr"],
                         params[:5]):
            print(f"  {nm:<9} = {v:.6g}")
        print(f"  mathcalE = {ev['error']:.6g}")
        common_t = case["t_nr"][ev["common"]]
        paths += [
            save_waveform_plot(q, common_t, ev["h_ref"], ev["h_model"], ev["error"]),
            save_parameter_plot(q, ev["tau"], ev["alpha"], ev["beta"],
                                case["losses"][M["coord"]], ev["error"],
                                common_t[0], common_t[-1]),
            save_zoomed_plot(q, common_t, ev["h_ref"], ev["h_model"]),
        ]
    paths += [save_coefficients_vs_q(), save_alpha_beta_overlay(results),
              save_drive_vs_q(results)]

    print("\n=== Summary ===")
    for q in Q_INPUTS:
        flag = " [extrap]" if _extrap(q) else "        "
        print(f"  q={q:<4}{flag}  nu={gwre.get_nu(q):.4f}  mathcalE={errors[q]:.4e}")
    print(f"\nPlots: {PLOT_DIR}")
    for p in paths:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
