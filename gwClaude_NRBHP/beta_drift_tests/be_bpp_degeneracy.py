"""Reconcile two results that appear to conflict.

  inspiral_only_sign.py   (PROFILED: all 4 params refit on t<=-200)  -> b_E > 0 at every q
  transition_q_decompose  (CONDITIONAL: other 3 held at the full optimum) -> b_E flips at ~4

If b_E and b_PP are degenerate on the inspiral -- which they should be, since the inspiral
constrains the accumulated phase int(beta dt), not beta's slope -- then the PROFILED
inspiral curve in b_E is nearly flat and its minimum's location carries little information,
while the CONDITIONAL curve is sharp.  Measure both.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_gwr_energy_global as GG
import inspiral_only_sign as IOS   # reuses truncate() and mism()

cache = sorted(json.load(open(str(HERE.parent /
    "gwr_energy_flux_results/per_q_cache_flux.json"))).values(), key=lambda v: v["q"])

for qt in (3.0, 5.3077, 8.0):
    rec = min(cache, key=lambda v: abs(v["q"] - qt)); q = float(rec["q"])
    p0 = np.array(rec["params"][:4], float); t0 = float(rec["t0"])
    case = IOS.truncate(FL.add_flux(G.load_case(q, 6, 10)), -200.0)
    print(f"\n=== q={q:g}   inspiral only (t_NR <= -200 M),  b_E(full-window) = {p0[3]:+.3f}")
    print(f"{'b_E':>7s} {'cond. err':>11s} {'prof. err':>11s} {'b_PP shift %':>13s}")
    grid = np.linspace(-3.0, 4.0, 15)
    cond, prof, shift = [], [], []
    for b in grid:
        p = p0.copy(); p[3] = b
        cond.append(IOS.mism(p, case, t0)[0])
        f = lambda s: IOS.mism(np.r_[p0[0], p0[1], p0[2] * s, b], case, t0)[0]
        r = minimize_scalar(f, bounds=(0.97, 1.03), method="bounded",
                            options={"xatol": 1e-7, "maxiter": 60})
        prof.append(float(r.fun)); shift.append((float(r.x) - 1.0) * 100.0)
        print(f"{b:7.2f} {cond[-1]:11.4e} {prof[-1]:11.4e} {shift[-1]:+13.4f}", flush=True)
    cond, prof = np.asarray(cond), np.asarray(prof)
    print(f"  conditional: min at b_E={grid[cond.argmin()]:+.2f}, "
          f"spread over the grid = x{cond.max()/cond.min():.1f}")
    print(f"  profiled   : min at b_E={grid[prof.argmin()]:+.2f}, "
          f"spread over the grid = x{prof.max()/prof.min():.1f}")
