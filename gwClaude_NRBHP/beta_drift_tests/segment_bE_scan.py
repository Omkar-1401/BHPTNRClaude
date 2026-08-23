"""WHICH REGION WANTS WHICH SIGN OF beta's DRIFT?  The decisive, orthogonal experiment.

Established already (all measured, all reproducible):
  * inspiral alone (t<-200) wants b_E > 0 at every q; flipping costs x90/x44/x5
  * ringdown alone (t>+20) prefers the rising QNM-anchor beta by 2.7-5.5x
  * the FULL window wants b_E < 0 for q > 4
Those can only be consistent if the MERGER region demands b_E < 0 strongly enough to
outvote both.  That region has never been scored on its own.  This does it.

For each q and each window, scan b_E and re-optimise the other three parameters
(a_PP, a_F, b_PP) at each point; report the preferred b_E and the penalty curve.
Also report each window's share of the L2 weight, which is what sets who wins.
"""
import sys, json, warnings
import numpy as np
from scipy.optimize import minimize, minimize_scalar
warnings.filterwarnings("ignore")
HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_gwr_energy_global as GG
import fit_scaling_gwr_energy_fluxanchored as FA

fa = json.load(open(str(HERE.parent / "gwr_energy_fluxanchored_results/coeffs.json")))
TH = np.concatenate([fa["c"], fa["A"], [fa["b"]], fa["P"]]); DEG = int(fa["aF_degree"])
WINDOWS = [("inspiral t<-200", -1e9, -200.0), ("merger -200..+20", -200.0, 20.0),
           ("ringdown t>+20", 20.0, 1e9), ("full window", -1e9, 1e9)]
BE_GRID = [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0]

def cut(case, lo, hi):
    c = dict(case)
    for tk, hk in (("t_nr","h_nr"), ("fit_t_nr","fit_h_nr")):
        t = np.asarray(case[tk], float); h = np.asarray(case[hk])
        m = (t >= lo) & (t <= hi); c[tk] = t[m]; c[hk] = h[m]
    return c

def mism(p, case, t0):
    a, tau, dmin = FL.model_shape(p, case["t_bhpt"], case["losses"])
    if dmin <= 0: return 50.0
    f = lambda s: GG.err_at_t0(s, a, tau, case["h_bhpt"], case["t_nr"], case["h_nr"])
    r = minimize_scalar(f, bounds=(t0-120., t0+120.), method="bounded",
                        options={"xatol":1e-3,"maxiter":60})
    return float(r.fun)

def best_at(case, pref, be):
    """re-optimise a_PP, a_F, b_PP at fixed b_E"""
    best = np.inf
    for s in (1.0, 0.98):
        x0 = np.array([pref[0]*s, pref[1], pref[2]])
        r = minimize(lambda u: mism(np.r_[u, be], case, -75.0), x0,
                     method="Nelder-Mead",
                     options={"maxiter":600,"xatol":1e-6,"fatol":1e-12})
        best = min(best, float(r.fun))
    return best

for q in (3.0, 5.0, 8.0):
    case = FL.add_flux(G.load_case(q, 6, 10))
    pref = FA.params_at(q, TH, DEG)
    Etot = float(case["losses"]["e_oft"][-1] - case["losses"]["e_oft"][0])
    tn = np.asarray(case["t_nr"], float); hn = np.abs(np.asarray(case["h_nr"]))
    tot = float(np.sum(hn**2))
    print(f"\n===== q = {q:g}   (shipped b_E = {pref[3]:+.4f}, "
          f"dbeta = {pref[3]*Etot*100:+.2f}%) =====")
    print(f"{'window':>17s} {'L2 wt':>7s} | " +
          " ".join(f"{b:+7.1f}" for b in BE_GRID) + " | best b_E  dbeta")
    for name, lo, hi in WINDOWS:
        cc = cut(case, lo, hi)
        m = (tn >= lo) & (tn <= hi)
        wt = float(np.sum(hn[m]**2))/tot
        vals = [best_at(cc, pref, be) for be in BE_GRID]
        i = int(np.argmin(vals)); nv = np.array(vals)/vals[i]
        print(f"{name:>17s} {wt:7.4f} | " + " ".join(f"{v:7.2f}" for v in nv) +
              f" | {BE_GRID[i]:+8.2f} {BE_GRID[i]*Etot*100:+7.2f}%", flush=True)
    print("   (rows are mismatch RELATIVE to that window's own best; 1.00 = preferred)")
