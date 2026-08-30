"""alpha/beta overlay for the FITTED (nu-regressed) inspiral-only models, one per model.

    python NRBHP_inspiral_alpha_beta_overlay.py                      # both, seeded coeffs
    python NRBHP_inspiral_alpha_beta_overlay.py --global             # joint-fit coeffs
    python NRBHP_inspiral_alpha_beta_overlay.py --model anchored

    Agentic_plots/inspiral_fluxanchored/inspiral_FA_alpha_beta_overlay.pdf
    Agentic_plots/inspiral_anchored/inspiral_AN_alpha_beta_overlay.pdf

These are MASTER curves: alpha and beta evaluated from the regressed coefficients in
`inspiral_<model>_results/coeffs[_global].json` (written by
`fit_scaling_inspiral_regress.py`), NOT per-q optima.  So q=2 is a genuine evaluation of
the model off its [3,8] training range and gets the house extrapolation emphasis.

FORMAT is that of NRBHP_alpha_beta_overlays.py / NRBHP_gwr_anch_perq_overlay.py: default
rcParams (10 pt DejaVu Sans -- see the rcdefaults() note below), 1 x 2 at figsize (12, 4),
alpha left / beta right, house colours q=2 tab:red / 3 tab:orange / 5 tab:green /
8 tab:blue with lw 2.2 on the q=2 extrapolation, intermediate q thin grey 0.65, grid on,
`legend(fontsize=8)`, xlabel "t_NR / M", suptitle, `tight_layout`, PDF + PNG.

THE FULL RANGE IS SHOWN, INCLUDING THE EXTRAPOLATION.  These models are fitted on
`t_nr < t_cut` only, but alpha and beta remain *defined* past the cut -- they are functions
of E(t) and F(t), which run to the end of the array -- so the continuation to +80 M is
drawn and the unscored region is SHADED (grey band, faded curves, dashed rule at the cut).
That region is where E and F do 66-82% of their range, so the excursions there are large
and are pure model form: nothing constrains them.  In particular the flux model's
unconstrained `alpha_C` sends alpha well above its inspiral value near merger -- that is
the pathology, not a fitting artefact, and showing it is the point.

`--clip-at-cut` restores the clipped version (xlim ending at t_cut) if a figure is wanted
that shows only what was actually fitted.

NOTE the rcdefaults() ordering: it MUST come after the model imports.  This chain pulls in
gw_remnant, whose gw_utils/gw_plotter.py sets font.size=18, font.family='STIXGeneral' and
figure.figsize=(14,10) at import time, so resetting first silently gives 18 pt STIX figures
instead of the 10 pt DejaVu Sans of every other overlay in Agentic_plots/.
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

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PARENT))

import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_inspiral_fluxanchored as IN
import fit_scaling_inspiral_regress as RG

mpl.rcdefaults()                       # AFTER the imports -- see module docstring
mpl.rcParams["text.usetex"] = False

PLOTS = PARENT / "Agentic_plots"

MAIN = {2.0: "tab:red", 3.0: "tab:orange", 5.0: "tab:green", 8.0: "tab:blue"}
SIDE = [4.0, 6.0]
QS = sorted(list(MAIN) + SIDE)

SPECS = {
    "fluxanchored": dict(dirname="inspiral_fluxanchored",
                         fname="inspiral_FA_alpha_beta_overlay",
                         label="inspiral-fluxanchored, FITTED (7 coef, alpha on F=Edot/max)"),
    "anchored": dict(dirname="inspiral_anchored",
                     fname="inspiral_AN_alpha_beta_overlay",
                     label="inspiral-anchored, FITTED (6 coef, alpha on E)"),
}


def load_theta(model, use_global):
    d = RG.results_dir(model) / ("coeffs_global.json" if use_global else "coeffs.json")
    if not d.exists():
        sys.exit(f"no coefficients at {d} -- run fit_scaling_inspiral_regress.py "
                 f"--model {model}" + (" --global" if use_global else ""))
    j = json.loads(d.read_text())
    theta = np.concatenate([np.asarray(j["c"], float), np.asarray(j["A"], float),
                            [float(j["b"])], np.asarray(j["P"], float)])
    return theta, j


def curves(q, theta, model, drive, t_cut, t0_seed):
    """Master alpha/beta and the time map at one q; t0 from the scored window."""
    case = FL.add_flux(G.load_case(q, IN.SRC_STRIDE, IN.NR_STRIDE))
    tcase = IN.truncate(case, t_cut)
    p, p_val = RG.params_at(q, theta, model)
    e_fit, t0 = IN.mism(p, tcase, t0_seed, drive, "fit")
    e_insp = IN.mism(p, tcase, t0, drive, "full")[0]
    alpha, tau_shape, dmin = IN.model_shape(p, case["t_bhpt"], case["losses"], drive)
    if dmin <= 0:
        sys.exit(f"q={q:g}: non-monotone time map, dmin={dmin:.3e}")
    beta = p[2] * (1.0 + p[3] * case["losses"]["e_oft"])
    return dict(tau=t0 + tau_shape, alpha=alpha, beta=beta, err=e_insp,
                b_E=float(p[3]), P=p_val, anchor=(q / (1.0 + q)) ** 1.2)


def draw(model, res, t_cut, clip_at_cut, use_global):
    S = SPECS[model]
    out_dir = PLOTS / S["dirname"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4))
    panels = ((axa, "alpha"), (axb, "beta"))
    x_hi = t_cut if clip_at_cut else 80.0

    def seg(ax, which, r, color, lw, zorder, label=None):
        # scored region opaque; unscored continuation same colour, faded, over the band.
        t = r["tau"]
        m = (t >= -1000) & (t <= t_cut)
        ax.plot(t[m], r[which][m], color=color, lw=lw, zorder=zorder, label=label)
        if not clip_at_cut:
            m2 = (t >= t_cut) & (t <= 80)
            ax.plot(t[m2], r[which][m2], color=color, lw=lw, zorder=zorder, alpha=0.45)

    for q in SIDE:
        for ax, which in panels:
            seg(ax, which, res[q], "0.65", 0.8, 1, label=f"$q={q:g}$")

    for q, c in MAIN.items():
        r = res[q]
        lw = 2.2 if q == 2 else 1.3
        ext = " [extrap]" if q == 2 else ""
        for ax, which in panels:
            seg(ax, which, r, c, lw, 3,
                label=(f"$q={q:g}${ext}  $\\mathcal{{E}}$={r['err']:.2e}"
                       if which == "alpha"
                       else f"$q={q:g}${ext}  $b_E$={r['b_E']:+.2f}"))
        axb.axhline(r["anchor"], color=c, lw=0.7, ls=(0, (4, 3)), zorder=2)

    for ax, yl, tl in ((axa, r"$\alpha(t)$", r"$\alpha(t)$ — master"),
                       (axb, r"$\beta(t)$",
                        r"$\beta(t)$ — master  (dashed: $X_1^{6/5}$)")):
        ax.axvline(t_cut, color="gray", ls="--", lw=0.9)
        if not clip_at_cut:
            # shade everything past the cut: fitted to its left, extrapolation to its right
            ax.axvspan(t_cut, 80, color="0.88", zorder=0)
            ax.axvline(0, color="gray", ls=":", lw=0.8)
            ax.text(0.985, 0.04, "not fitted", transform=ax.transAxes, fontsize=7,
                    color="0.35", ha="right", va="bottom", zorder=4)
        ax.set_xlim(-1000, x_hi)
        ax.set_xlabel(r"$t_{\mathrm{NR}}\,/\,M$")
        ax.set_ylabel(yl)
        ax.set_title(tl)
        ax.grid(True)
        # 10 curves per panel against the reference's 4, so 'best' placement lands the
        # legend on the data -- open a strip below the curves and pin it there.  Without
        # the extra room the box clips the q=2 curve, which is the one worth seeing.
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0 - 0.34 * (y1 - y0), y1)
        ax.legend(fontsize=8, ncol=2, loc="lower left", framealpha=0.92)

    tag = "joint fit" if use_global else "regressed per-q"
    fig.suptitle(f"{S['label']}: master $\\alpha$, $\\beta$ across $q$  [{tag}, "
                 f"fitted on $t_{{\\mathrm{{NR}}}} < {t_cut:g}\\,M$]")
    fig.tight_layout()
    # the two routes MUST NOT share a filename: a run that produces both would have the
    # joint fit silently overwrite the seeded figure (it did, on 2026-08-23).
    stem = S["fname"] + ("_global" if use_global else "_seeded")
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=sorted(SPECS), choices=sorted(SPECS))
    ap.add_argument("--t-cut", type=float, default=IN.DEFAULT_T_CUT)
    ap.add_argument("--global", dest="use_global", action="store_true",
                    help="use coeffs_global.json (the joint fit) instead of coeffs.json")
    ap.add_argument("--clip-at-cut", action="store_true",
                    help="end the panel at t_cut, showing only the fitted region "
                         "(default: full range with the unscored part shaded)")
    args = ap.parse_args()

    t0_seeds = {k: v[4] for k, v in IN.mult_seeds().items()}
    nu_grid = np.array([RG.nu_of(q) for q in sorted(t0_seeds)])
    t0_poly = np.polyfit(nu_grid, [t0_seeds[q] for q in sorted(t0_seeds)], 3)

    for model in args.model:
        theta, j = load_theta(model, args.use_global)
        drive = IN.MODELS[model]["drive"]
        print(f"\n[inspiral_{model}] FITTED master, {j['n_coefficients']} coef, "
              f"drive={drive}, global={j['global_refit']}, t_cut={args.t_cut:g} M",
              flush=True)
        res = {}
        for q in QS:
            seed = float(np.polyval(t0_poly, float(RG.nu_of(q))))
            res[q] = curves(q, theta, model, drive, args.t_cut, seed)
            r = res[q]
            print(f"  q={q:<4g} E_insp={r['err']:.4e}  b_E={r['b_E']:+7.4f}  "
                  f"P={r['P']:+7.4f}  beta {'rises' if r['b_E'] > 0 else 'FALLS'}",
                  flush=True)
        for p in draw(model, res, args.t_cut, args.clip_at_cut, args.use_global):
            print("wrote", p)


if __name__ == "__main__":
    main()
