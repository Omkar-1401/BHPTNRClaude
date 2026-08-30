"""
Fig-8-style higher-mode waveform plots (arXiv:2204.01972) for the gwr_energy_stiff
backbone: stacked panels (2,2)/(3,3)/(4,4), Re(h_lm) vs t, NR vs the backbone model.

Format is deliberately IDENTICAL to the pn_anchored prototypes
`peaks_results/prototypes/hm_q{2,4}_fig8.py` -- same rcdefaults, figsize (11,9), 3x1
sharex, black solid NR / red dashed model with the same linewidths, xlim, grid alpha,
merger line, per-panel mode + mathcalE annotation, legend placement, suptitle pattern
and dpi -- so the two models' figures can be laid side by side.  Those two scripts
differ only in Q, the suptitle tag and the filename, so this one takes --q.

The only substantive difference from the prototypes is what is plotted: here the model
is the ZERO-PARAMETER backbone (rho = 1), and the alignment comes from this model's own
evaluator (the shared time map with t0_nr already polished on the (2,2), plus the
analytic per-mode constant phase) rather than the prototypes' separate offset search.
That keeps the mathcalE printed on each panel identical to the table in
scaling_hm_backbone.md, so the figures and the quoted numbers cannot drift apart.

Usage:  python NRBHP_hm_backbone_fig8_plots.py --q 2 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import fit_scaling_hm_backbone as hm

# rcdefaults MUST come AFTER the imports, not before as in the pn_anchored prototypes.
# Those prototypes never imported `gw_remnant`, but this model's chain does, and
# `gw_remnant/gw_utils/gw_plotter.py` sets font.size=18, font.family='STIXGeneral',
# axes.linewidth=1 and figure.figsize=(14,10) at import time.  Resetting first and
# importing second would silently produce 18pt STIX figures instead of the 10pt
# DejaVu Sans the prototypes use.  Same ordering as NRBHP_gw_remnant_energy_plots.py.
matplotlib.rcdefaults(); matplotlib.rcParams["text.usetex"] = False

PLOT_DIR = ROOT / "Agentic_plots" / "hm_backbone"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_MODES = [(2, 2), (3, 3), (4, 4)]


def build(q, coeffs, t_cut=None):
    """Backbone model curves for the panel modes, on the shared time map."""
    qm = hm.quadrupole_model(q, coeffs, t_cut=t_cut)
    data = hm.load_hm(q)
    t_src = qm["case"]["t_bhpt"]
    t_hm = data["t_bhpt"]
    tau_h = np.interp(t_hm, t_src, qm["tau"])
    beta_h = np.interp(t_hm, t_src, qm["beta"])
    a22_h = np.interp(t_hm, t_src, qm["alpha22"])

    out = {}
    for (l, m) in PANEL_MODES:
        a_lm = a22_h * hm.C_lm(l, m, q) * beta_h ** hm.beta_exponent(l, m)
        # curves over the WHOLE window so the extrapolation past t_cut is drawn;
        # the quoted mathcalE is scored on t_cut, matching the tables.
        out[(l, m)] = hm.mode_mismatch(t_hm, tau_h, a_lm, data["h_bhpt"][(l, m)],
                                       data["t_nr"], data["h_nr"][(l, m)],
                                       return_curves=True)
        if t_cut is not None:
            sc = hm.mode_mismatch(t_hm, tau_h, a_lm, data["h_bhpt"][(l, m)],
                                  data["t_nr"], data["h_nr"][(l, m)], t_cut=t_cut)
            out[(l, m)]["err_rho1_scored"] = sc["err_rho1"]
    return out, data


def save_fig8(q, coeffs, model="stiff", t_cut=None):
    res, data = build(q, coeffs, t_cut=t_cut)
    tag = "[in-range]" if 3.0 <= q <= 8.0 else "[EXTRAP]"

    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for i, lm in enumerate(PANEL_MODES):
        r = res[lm]
        ax[i].plot(data["t_nr"], np.real(data["h_nr"][lm]), "k-", lw=1.2,
                   label="NR (NRHybSur3dq8)")
        ax[i].plot(r["t_common"], np.real(r["h_model"]), "r--", lw=1.1,
                   label=f"{hm.MODEL_LABELS.get(model, model)} backbone (rho=1)")
        ax[i].set_xlim(-600, 90); ax[i].grid(alpha=0.3)
        ax[i].axvline(0, ls=":", color="gray", lw=1)
        if t_cut is not None:
            # everything right of the cut is unscored extrapolation -- same convention
            # as the inspiral_only alpha/beta overlays.
            ax[i].axvspan(t_cut, 90, color="0.88", zorder=0)
            ax[i].axvline(t_cut, ls="--", color="gray", lw=0.9)
            if i == 0:
                ax[i].text(0.985, 0.04, "not fitted", transform=ax[i].transAxes,
                           fontsize=8, color="0.35", ha="right", va="bottom")
        ax[i].set_ylabel(f"$\\mathrm{{Re}}\\,h_{{{lm[0]}{lm[1]}}}$")
        e_show = r.get("err_rho1_scored", r["err_rho1"])
        ax[i].text(0.02, 0.9,
                   f"$({lm[0]},{lm[1]})$   $\\mathcal{{E}}$ = {e_show:.2e}",
                   transform=ax[i].transAxes, fontsize=10, va="top")
        if i == 0:
            ax[i].legend(fontsize=9, loc="upper right")
    ax[2].set_xlabel(r"$t\,/\,M$")
    cut_tag = f"  [scored on t < {t_cut:g} M]" if t_cut is not None else ""
    fig.suptitle(f"q={q:g} {tag}: {hm.MODEL_LABELS.get(model, model)} backbone (rho=1) vs NR "
                 f"— (2,2),(3,3),(4,4)  (Fig.8-style){cut_tag}")
    fig.tight_layout()
    if model in ("stiff",):
        p = PLOT_DIR / f"q{q:g}_waveforms_224.png"
    elif model == "anchored":
        p = PLOT_DIR / f"q{q:g}_waveforms_224_anchored.png"
    else:
        p = PLOT_DIR / f"hm_q{q:g}_{model}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"saved {p}")
    for lm in PANEL_MODES:
        r = res[lm]
        if t_cut is not None:
            print(f"  {lm}: mathcalE(scored)={r['err_rho1_scored']:.3e}  "
                  f"full-window={r['err_rho1']:.3e}")
        else:
            print(f"  {lm}: mathcalE={r['err_rho1']:.3e}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, nargs="+", default=[2.0, 4.0])
    ap.add_argument("--model", nargs="+", default=["stiff"],
                    choices=("stiff", "anchored", "E_deg3", "flux_deg3",
                             "E_anchored", "flux_anchored",
                             "beta_monotone", "beta_monotone_E") + hm.INSPIRAL_MODELS)
    ap.add_argument("--t-cut", type=float, default=None,
                    help="shade and stop scoring past this t_NR; defaults to -200 for "
                         "the inspiral_* bases")
    ap.add_argument("--outdir", default=None,
                    help="override the plot directory (e.g. consolidated_gwremnant_hm)")
    args = ap.parse_args()
    global PLOT_DIR
    if args.outdir:
        PLOT_DIR = ROOT / "Agentic_plots" / args.outdir
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.model:
        coeffs = hm.read_stiff_coeffs(name)
        t_cut = args.t_cut
        if t_cut is None and name in hm.INSPIRAL_MODELS:
            t_cut = -200.0
        for q in args.q:
            print(f"\n--- {name}  q={q:g} ---", flush=True)
            save_fig8(q, coeffs, name, t_cut=t_cut)


if __name__ == "__main__":
    main()
