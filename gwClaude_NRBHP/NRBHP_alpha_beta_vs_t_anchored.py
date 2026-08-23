"""alpha and beta vs t for the ENERGY-ANCHORED model (gwr_energy_anchored), per q.

Same enriched style as NRBHP_alpha_beta_vs_x.py -- seven mass ratios rather than four,
with q=2/3/5/8 in the house colours (lw 2.2 for q=2), the intermediates q=2.5/4/6 thin
grey, and dashed horizontals marking X1^(6/5) in the beta panel -- but the abscissa is
t_NR, since this model is built in the time domain.

    alpha(t) = a_PP(q) * (1 + a_E(nu)*E(t))       a_PP = X1^(6/5)(1+c0 nu+c1 nu^2)
    beta (t) = b_PP(q) * (1 + b_E(q)*E(t))        b_PP = X1^(6/5)(1+b nu)
                                                  b_E  = P(nu)/b_PP,  P = P0+P1 nu
coefficients from gwr_energy_anchored_results/coeffs.json (global fit).

rcdefaults() AFTER the model imports (gw_remnant's gw_plotter sets 18pt STIX at import).

Usage:  python NRBHP_alpha_beta_vs_t_anchored.py
"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_global as GG
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_gwr_energy_anchored as AN

mpl.rcdefaults(); mpl.rcParams["text.usetex"] = False

d = json.loads((AN.RESULTS/"coeffs.json").read_text())
TH = np.concatenate([np.asarray(d["c"], float), np.asarray(d["A"], float),
                     [float(d["b"])], np.asarray(d["P"], float)])
DEG = int(d["alphaE_degree"])
PLOT_DIR = ROOT/"Agentic_plots"/"alpha_beta_of_x"; PLOT_DIR.mkdir(parents=True, exist_ok=True)
MAIN = {2.0: "tab:red", 3.0: "tab:orange", 5.0: "tab:green", 8.0: "tab:blue"}
SIDE = [2.5, 4.0, 6.0]
QS = sorted(list(MAIN) + SIDE)

def build(q):
    case = FL.add_flux(G.load_case(q, 6, 10))
    p = AN.params_at(q, TH, DEG)
    e = np.asarray(case["losses"]["e_oft"], float); t = np.asarray(case["t_bhpt"], float)
    alpha = p[0]*(1.0 + p[1]*e); beta = p[2]*(1.0 + p[3]*e)
    bc = G.creative.cumulative_trapezoid(beta, t)
    tau0 = bc - float(np.interp(G.T_ANCHOR, t, bc))
    r = minimize_scalar(lambda s: GG.err_at_t0(s, alpha, tau0, case["h_bhpt"],
                                               case["t_nr"], case["h_nr"]),
                        bounds=(-195., 45.), method="bounded",
                        options={"xatol": 1e-3, "maxiter": 80})
    return dict(tau=float(r.x)+tau0, alpha=alpha, beta=beta, err=float(r.fun),
                anchor=(q/(1+q))**1.2)

res = {q: build(q) for q in QS}
for q in QS: print(f"  q={q:<5g} mathcalE={res[q]['err']:.4e}", flush=True)

fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4))
for q in SIDE:
    r = res[q]; m = (r["tau"] >= -1000) & (r["tau"] <= 80)
    axa.plot(r["tau"][m], r["alpha"][m], color="0.65", lw=0.8, zorder=1, label=f"q={q:g}")
    axb.plot(r["tau"][m], r["beta"][m],  color="0.65", lw=0.8, zorder=1, label=f"q={q:g}")
for q, c in MAIN.items():
    r = res[q]; m = (r["tau"] >= -1000) & (r["tau"] <= 80); lw = 2.2 if q == 2 else 1.3
    axa.plot(r["tau"][m], r["alpha"][m], color=c, lw=lw, zorder=3,
             label=f"q={q:g}  E={r['err']:.2e}")
    axb.plot(r["tau"][m], r["beta"][m],  color=c, lw=lw, zorder=3, label=f"q={q:g}")
    axb.axhline(r["anchor"], color=c, lw=0.7, ls=(0, (4, 3)), zorder=2)
for ax, yl, tl in ((axa, "alpha(t)", "alpha vs t   (energy-anchored model)"),
                   (axb, "beta(t)",  "beta vs t    (dashed: X1^(6/5) anchor)")):
    ax.axvline(0, color="gray", ls=":", lw=0.8); ax.set_xlim(-1000, 80)
    ax.set_xlabel("t_NR / M"); ax.set_ylabel(yl); ax.set_title(tl)
    ax.grid(True); ax.legend(fontsize=8, ncol=2)
fig.suptitle("gwr-energy-anchored: master alpha/beta vs t across q")
fig.tight_layout()
for ext in ("pdf", "png"):
    p = PLOT_DIR/f"alpha_beta_vs_t_anchored.{ext}"; fig.savefig(p); print("wrote", p)
plt.close(fig)
