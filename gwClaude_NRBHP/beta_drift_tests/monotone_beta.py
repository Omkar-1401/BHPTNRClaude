"""Can beta be made non-decreasing at ALL q, with no post-merger correction and no
extra coefficient?  Four variants, same 4-param per-q model, flux alpha throughout:

  A  E drive,     b_E free            (baseline = shipped)
  B  E drive,     b_E >= 0            (price of the constraint, same coordinate)
  C  E_sat drive, b_E free            E_sat = min(E, E(t_merger)); 0 new params
  D  E_sat drive, b_E >= 0            the target: rising beta, frozen after merger
"""
import sys, json, warnings
import numpy as np
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL

SCRATCH = str(__import__("pathlib").Path(__file__).resolve().parent)

QS = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


def add_esat(case):
    """E clipped at its merger value: monotone non-decreasing, flat after merger."""
    t = np.asarray(case["t_bhpt"], float)
    e = np.asarray(case["losses"]["e_oft"], float)
    im = int(np.argmax(np.abs(case["h_bhpt"])))
    es = np.minimum(e, e[im])
    case["losses"]["e_sat"] = es
    case["fit_losses"]["e_sat"] = np.interp(case["fit_t_bhpt"], t, es)
    return case, float(e[im] - e[0]), float(e[-1] - e[0])


def shape(p, t_bhpt, losses, drive, nonneg):
    a_pp, a_f, b_pp, u = p
    b_e = u * u if nonneg else u
    alpha = a_pp * (1.0 + a_f * losses["flux_hat"])
    beta = b_pp * (1.0 + b_e * losses[drive])
    if np.any(beta <= 0):
        return alpha, None, -1.0
    bc = G.creative.cumulative_trapezoid(beta, t_bhpt)
    tau = bc - float(np.interp(G.T_ANCHOR, t_bhpt, bc))
    return alpha, tau, float(np.min(np.diff(tau)))


def mism(p, case, t0_seed, drive, nonneg, key="fit"):
    from scipy.optimize import minimize_scalar
    import fit_scaling_gwr_energy_global as GG
    if key == "fit":
        tb, hb, tn, hn, los = (case["fit_t_bhpt"], case["fit_h_bhpt"], case["fit_t_nr"],
                               case["fit_h_nr"], case["fit_losses"])
    else:
        tb, hb, tn, hn, los = (case["t_bhpt"], case["h_bhpt"], case["t_nr"],
                               case["h_nr"], case["losses"])
    a, tau, dmin = shape(p, tb, los, drive, nonneg)
    if dmin <= 0:
        return 50.0 + 1e3 * (-dmin), t0_seed
    f = lambda t0: GG.err_at_t0(t0, a, tau, hb, tn, hn)
    r = minimize_scalar(f, bounds=(t0_seed - 60.0, t0_seed + 60.0), method="bounded",
                        options={"xatol": 1e-3, "maxiter": 50})
    return float(r.fun), float(r.x)


def fit(case, pref, t0_ref, drive, nonneg):
    best = None
    seeds = ([0.05, 0.4, 1.0] if nonneg
             else [pref[3], -1.0, 1.0])
    for u0 in seeds:
        for s in (1.0, 0.98):
            p0 = np.array([pref[0] * s, pref[1], pref[2], u0], float)
            st = {"t0": t0_ref}
            def obj(p):
                e, t0 = mism(p, case, st["t0"], drive, nonneg, "fit")
                st["t0"] = t0
                return e
            r = minimize(obj, p0, method="Nelder-Mead",
                         options={"maxiter": 3000, "xatol": 1e-7, "fatol": 1e-12})
            if best is None or r.fun < best[0]:
                best = (float(r.fun), r.x.copy(), st["t0"])
    _, p, t0 = best
    full, _ = mism(p, case, t0, drive, nonneg, "full")
    b_e = p[3] ** 2 if nonneg else p[3]
    return full, b_e, p


cache = json.load(open(str(__import__("pathlib").Path(__file__).resolve().parent.parent) + "/"
                       "gwr_energy_flux_results/per_q_cache_flux.json"))
res = {}
print(f"{'q':>5s} | {'A base':>10s} {'b_E':>7s} | {'B E,b>=0':>10s} {'x':>5s} "
      f"| {'C Esat':>10s} {'b_E':>7s} | {'D Esat,b>=0':>11s} {'x':>5s} | {'dbeta%':>7s}")
for q in QS:
    ref = min(cache.values(), key=lambda v: abs(v["q"] - q))
    pref = np.array(ref["params"], float); t0r = ref["t0"]
    case = FL.add_flux(G.load_case(q, 6, 10))
    case, E_m, E_tot = add_esat(case)
    A, beA, _ = fit(case, pref, t0r, "e_oft", False)
    B, beB, _ = fit(case, pref, t0r, "e_oft", True)
    C, beC, _ = fit(case, pref, t0r, "e_sat", False)
    D, beD, _ = fit(case, pref, t0r, "e_sat", True)
    res[q] = dict(A=A, beA=beA, B=B, beB=beB, C=C, beC=beC, D=D, beD=beD,
                  E_m=E_m, E_tot=E_tot)
    print(f"{q:5.1f} | {A:10.4e} {beA:+7.3f} | {B:10.4e} {B/A:5.2f} "
          f"| {C:10.4e} {beC:+7.3f} | {D:11.4e} {D/A:5.2f} | {beD*E_m*100:+6.2f}%",
          flush=True)

print()
for k, lab in (("A", "A  E, free   (shipped)"), ("B", "B  E, b_E>=0          "),
               ("C", "C  Esat, free         "), ("D", "D  Esat, b_E>=0       ")):
    v = np.array([res[q][k] for q in QS])
    print(f"{lab}  median {np.median(v):.4e}   max {v.max():.4e}")
json.dump({str(k): v for k, v in res.items()},
          open(f"{SCRATCH}/monotone_beta.json", "w"), indent=1)
