"""Is the transition q where the PHYSICAL anchor gap gb(q) is swamped by the fit's own
systematic error on beta's drift?

  gb(q)   = (B1 - B0)/B0,  B0 = X1^(6/5),  B1 = W_Schw*Mf/omega_220(chi_f)   [DERIVED,
            both anchors verified against direct omega_pp/omega_NR to <1%]
  dbeta   = b_E * E_tot,  the fitted total drift of the L2 models over the window

gb > 0 at every q, so the anchors NEVER cross -- if the fitted drift crosses zero, that is
the fit departing from the physical endpoints, not the physics changing sign.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_gwr_anchored2 as A2

BASE = HERE.parent
flux = sorted(json.load(open(str(BASE / "gwr_energy_flux_results/per_q_cache_flux.json"))
                        ).values(), key=lambda v: v["q"])
mult = sorted(json.load(open(str(BASE / "gw_remnant_energy_results/per_q_cache_mult.json"))
                        ).values(), key=lambda v: v["q"])

print(f"{'q':>5s} {'gb % (QNM anchor)':>18s} {'dbeta % (flux fit)':>19s} "
      f"{'dbeta % (E fit)':>16s} {'flux - gb':>10s} {'E - gb':>9s}")
rows = []
for rec in flux:
    q = float(rec["q"])
    case = FL.add_flux(G.load_case(q, 6, 10))
    gb = A2.anchors(q, case)[2] * 100.0
    Et = float(case["losses"]["e_oft"][-1] - case["losses"]["e_oft"][0])
    df = rec["params"][3] * Et * 100.0
    rm = min(mult, key=lambda v: abs(v["q"] - q))
    de = (rm["params"][3] * Et * 100.0
          if abs(rm["q"] - q) < 0.06 else float("nan"))
    print(f"{q:5.2f} {gb:18.3f} {df:19.3f} {de:16.3f} {df-gb:10.3f} {de-gb:9.3f}",
          flush=True)
    rows.append((q, gb, df, de))
a = np.array([[r[0], r[1], r[2]] for r in rows], float)
for j, nm in ((2, "flux"),):
    s = np.where(np.diff(np.sign(a[:, j])))[0]
    if len(s):
        i = s[0]
        print(f"\nfitted drift ({nm}) crosses zero at q = "
              f"{a[i,0] + (a[i+1,0]-a[i,0])*(-a[i,j])/(a[i+1,j]-a[i,j]):.3f}")
print(f"gb never crosses: min over the grid = {a[:,1].min():+.3f}% at q={a[a[:,1].argmin(),0]:g}")
json.dump([{"q": r[0], "gb_pct": r[1], "dbeta_flux_pct": r[2], "dbeta_E_pct": r[3]}
           for r in rows], open(str(HERE / "gb_vs_fitted_drift.json"), "w"), indent=1,
          default=float)
