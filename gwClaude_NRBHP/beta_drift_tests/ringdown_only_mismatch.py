"""Does the shipped model actually fit the RINGDOWN, or is L2 indifferent there?

The QNM anchor (parameter-free, matches the measured omega ratio to <1%) says beta must
END ABOVE where it starts.  The shipped fluxanchored model has beta FALLING for q>4.
Both cannot be right.  This test scores the mismatch on the ringdown ALONE, for
  (a) the shipped fluxanchored coefficients  (beta falls)
  (b) the same alpha, but b_E replaced by the ANCHOR value  (beta rises)
If (b) wins on the ringdown while (a) wins overall, L2 is trading ringdown accuracy away
and the anchor is right.
"""
import sys, json, warnings
import numpy as np
from scipy.optimize import minimize_scalar
warnings.filterwarnings("ignore")
HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_gwr_energy_global as GG
import fit_scaling_gwr_energy_fluxanchored as FA
import fit_scaling_gwr_anchored2 as A2

fa = json.load(open(str(HERE.parent / "gwr_energy_fluxanchored_results/coeffs.json")))
th = np.concatenate([fa["c"], fa["A"], [fa["b"]], fa["P"]]); deg = int(fa["aF_degree"])

def cut_case(case, tmin):
    c = dict(case)
    for tk, hk in (("t_nr","h_nr"), ("fit_t_nr","fit_h_nr")):
        t = np.asarray(case[tk], float); h = np.asarray(case[hk]); m = t >= tmin
        c[tk] = t[m]; c[hk] = h[m]
    return c

def mism(p, case, t0_seed):
    a, tau, dmin = FL.model_shape(p, case["t_bhpt"], case["losses"])
    if dmin <= 0: return 50.0
    f = lambda t0: GG.err_at_t0(t0, a, tau, case["h_bhpt"], case["t_nr"], case["h_nr"])
    r = minimize_scalar(f, bounds=(t0_seed-120., t0_seed+120.), method="bounded",
                        options={"xatol":1e-3,"maxiter":80})
    return float(r.fun)

print(f"{'q':>5s} {'window':>14s} | {'shipped (beta falls)':>20s} {'anchor beta (rises)':>20s} "
      f"{'ratio':>7s}  winner")
for q in (3.0,5.0,8.0,2.0):
    case = FL.add_flux(G.load_case(q,6,10))
    p_ship = FA.params_at(q, th, deg)
    A0, ga, gb, Etot, A1, B1 = A2.anchors(q, case)
    p_anch = p_ship.copy()
    p_anch[3] = gb * A0 / p_ship[2] / Etot          # beta_E from the anchor (positive)
    t0 = -75.0
    for label, tmin in (("full", -1e9), ("ringdown t>0", 0.0), ("QNM t>+20", 20.0)):
        cc = case if tmin < -1e8 else cut_case(case, tmin)
        e_s = mism(p_ship, cc, t0); e_a = mism(p_anch, cc, t0)
        win = "anchor" if e_a < e_s else "shipped"
        print(f"{q:5.1f} {label:>14s} | {e_s:20.4e} {e_a:20.4e} {e_a/e_s:7.2f}  {win}",
              flush=True)
    print()
