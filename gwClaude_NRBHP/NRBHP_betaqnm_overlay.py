"""alpha/beta overlay for gwr_betaqnm, in the house (stiff) format.

Format copied from NRBHP_gwr_energy_stiff_plots.save_alpha_beta_overlay: 1 x 2, one
curve per q in {2,3,5,8} coloured tab:red/orange/green/blue, figsize (12,4), curves
resampled onto the NR time grid, lw 2.2 for the q=2 extrapolation, xlim (-1000, 80),
grid on, default rcParams, PDF.

beta here is NOT of the b_pp*(1+b_c*E) form the shared evaluator assumes -- it is the
three-corner construction (B0 imposed, Bm fitted, B1 = QNM plateau) -- so this script
carries its own evaluator rather than extending NRBHP_alpha_beta_overlays.py.

rcdefaults() comes AFTER the model imports: the chain pulls in gw_remnant, whose
gw_utils/gw_plotter.py sets font.size=18 / STIXGeneral / figsize (14,10) at import.
"""
from __future__ import annotations
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
import fit_scaling_gwr_betaqnm as BQ

mpl.rcdefaults(); mpl.rcParams["text.usetex"] = False     # AFTER the imports

Q_INPUTS = [2.0, 3.0, 5.0, 8.0]
PLOT_DIR = ROOT / "Agentic_plots" / "gwr_betaqnm"; PLOT_DIR.mkdir(parents=True, exist_ok=True)
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "parfree"     # parfree | bfree | plain
FN = {"parfree": "coeffs_parfree.json", "bfree": "coeffs_bfree.json",
      "plain": "coeffs.json"}[VARIANT]
d = json.loads((ROOT/"gwr_betaqnm_results"/FN).read_text())
TH = np.asarray(d["theta"], float); BFREE = bool(d.get("bfree", False))
NFIT = 3 if d.get("parfree") else len(TH)
SUF = {"parfree": "_parfree", "bfree": "_bfree", "plain": ""}[VARIANT]
print(f"variant={VARIANT}  theta={[round(v,4) for v in TH]}  "
      f"({NFIT} fitted)  median={d['in_range']['median']:.4e}")

def build(q):
    case = FL.add_flux(G.load_case(q, 6, 10)); P = BQ.prep(q, case)
    c0, c1, A, m = TH[:4]; b = TH[4] if BFREE else 0.0
    alpha, beta, B0, Bm, gb = BQ.shape(P, c0, c1, A, m, b)
    bc = G.creative.cumulative_trapezoid(beta, P["t"])
    tau0 = bc - float(np.interp(G.T_ANCHOR, P["t"], bc))
    f = lambda s: GG.err_at_t0(s, alpha, tau0 + s*0 + 0.0, case["h_bhpt"],
                               case["t_nr"], case["h_nr"])
    r = minimize_scalar(lambda s: GG.err_at_t0(s, alpha, tau0, case["h_bhpt"],
                                               case["t_nr"], case["h_nr"]),
                        bounds=(-195., 45.), method="bounded",
                        options={"xatol": 1e-3, "maxiter": 80})
    t0 = float(r.x)
    return dict(tau=t0+tau0, alpha=alpha, beta=beta, err=float(r.fun), case=case)

res = {q: build(q) for q in Q_INPUTS}
for q in Q_INPUTS: print(f"  q={q:g}  mathcalE={res[q]['err']:.4e}", flush=True)

cols = {2.0: "tab:red", 3.0: "tab:orange", 5.0: "tab:green", 8.0: "tab:blue"}
fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4))
for q in Q_INPUTS:
    r = res[q]; tau = r["tau"]; t_nr = r["case"]["t_nr"]
    m = (t_nr >= max(t_nr[0], tau[0])) & (t_nr <= min(t_nr[-1], tau[-1]))
    t = t_nr[m]; lw = 2.2 if q == 2 else 1.3
    axa.plot(t, np.interp(t, tau, r["alpha"]), color=cols[q], lw=lw,
             label=f"q={q:.0f}{' [extrap]' if q == 2 else ''}  E={r['err']:.2e}")
    axb.plot(t, np.interp(t, tau, r["beta"]), color=cols[q], lw=lw, label=f"q={q:.0f}")
for ax, yl, tl in [(axa, "alpha(t)", "alpha(t) — master"),
                   (axb, "beta(t)", "beta(t) — master")]:
    ax.axvline(0, color="gray", ls=":", lw=0.8); ax.set_xlim(-1000, 80); ax.grid(True)
    ax.set_xlabel("t_NR / M"); ax.set_ylabel(yl); ax.set_title(tl); ax.legend(fontsize=8)
fig.suptitle(f"gwr-betaqnm{SUF.replace('_',' ')}: master alpha/beta across q"); fig.tight_layout()
for ext in ("pdf", "png"):
    p = PLOT_DIR / f"gwr_betaqnm{SUF}_alpha_beta_overlay.{ext}"; fig.savefig(p); print("wrote", p)
plt.close(fig)
