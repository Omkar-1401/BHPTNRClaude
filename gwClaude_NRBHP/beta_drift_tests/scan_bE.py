"""q=8: penalty in mismatch for forcing b_E (=P/beta_PP) away from its optimum.
Fix b_E, re-optimise (a_PP, a_F, b_PP) with t0 the inner 1-D nuisance."""
import sys, json, warnings
import numpy as np
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL

Q = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
case = FL.add_flux(G.load_case(Q, 6, 10))
x1 = Q / (1.0 + Q)
cache = json.load(open(str(__import__("pathlib").Path(__file__).resolve().parent.parent) + "/"
                       "gwr_energy_flux_results/per_q_cache_flux.json"))
ref = None
for v in cache.values():
    if abs(v["q"] - Q) < 1e-6:
        ref = v
print(f"q={Q}  per-q optimum from cache: params={np.round(ref['params'],5).tolist()} "
      f"err={ref['err']:.4e}", flush=True)
t0_ref = ref["t0"]; p_ref = np.array(ref["params"], float)

def best_at(be, t0_seed):
    """optimise the 3 non-b_E params at fixed b_E; return (fit_err, full_err, p, t0)"""
    best = None
    for a_f in (p_ref[1], -0.15, -0.35):
        for s in (1.0, 0.98, 1.02):
            p3 = np.array([p_ref[0] * s, a_f, p_ref[2]], float)
            st = {"t0": t0_seed}
            def obj(u):
                p = np.array([u[0], u[1], u[2], be])
                e, t0 = FL.fast_mismatch(p, case, st["t0"], "fit")
                st["t0"] = t0
                return e
            r = minimize(obj, p3, method="Nelder-Mead",
                         options={"maxiter": 3000, "xatol": 1e-7, "fatol": 1e-12})
            if best is None or r.fun < best[0]:
                best = (float(r.fun), r.x.copy(), st["t0"])
    fit_e, p3, t0 = best
    p = np.array([p3[0], p3[1], p3[2], be])
    full_e, t0f = FL.fast_mismatch(p, case, t0, "full")
    return fit_e, full_e, p, t0f

grid = [-2.5, -2.0, -1.75, -1.5, p_ref[3], -1.25, -1.0, -0.75, -0.5, -0.25,
        0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
grid = sorted(set(round(g, 6) for g in grid))
print(f"\n{'b_E':>8s} {'fit E':>12s} {'full E':>12s} {'ratio/opt':>10s} "
      f"{'a_PP':>8s} {'a_F':>8s} {'b_PP':>8s} {'dbeta/beta':>11s}")
out = []
E_tot = float(case["losses"]["e_oft"][-1] - case["losses"]["e_oft"][0])
for be in grid:
    fe, fu, p, t0 = best_at(be, t0_ref)
    out.append((be, fe, fu, p.tolist()))
    print(f"{be:+8.3f} {fe:12.5e} {fu:12.5e} {'':>10s} "
          f"{p[0]:8.5f} {p[1]:+8.4f} {p[2]:8.5f} {be*E_tot*100:+10.2f}%", flush=True)
best_full = min(o[2] for o in out)
print(f"\n  E_tot={E_tot:.5f}   best full E = {best_full:.5e}")
print(f"\n{'b_E':>8s} {'full E':>12s} {'x optimum':>10s}")
for be, fe, fu, p in out:
    print(f"{be:+8.3f} {fu:12.5e} {fu/best_full:10.2f}")
json.dump(out, open(str(__import__("pathlib").Path(__file__).resolve().parent) +
                    f"/scan_bE_q{Q}.json", "w"))
