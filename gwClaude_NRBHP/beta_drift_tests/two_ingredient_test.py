"""Test the two proposed ingredients SEPARATELY.

(A) the full-window metric pins beta's MEAN (phase coherence / window matching), so a
    rise must be paid for by lowering the level.
(B) the rigid E profile crams the rise into the last ~2% of the window, making that
    payment unaffordable because the inspiral carries ~84% of the L2 weight.

TEST A -- no new model.  Scan b_E; at each point re-optimise (a_PP, a_F, b_PP) on the
FULL window; record mismatch, b_PP, and mean beta = int(beta dt)/T, plus beta at the
window start and end.  If (A) holds, mean beta stays ~constant while b_PP slides.

TEST B -- placeable rise.  beta = b_PP*(1 + b_E*Ehat^p), b_E > 0 FORCED, scan p.
If (B) holds, some p < 1 gives a RISING beta at much better mismatch than p = 1.
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
QS = (3.0, 5.0, 8.0)
BE = [-0.08, -0.04, -0.02, 0.0, 0.02, 0.04, 0.08]   # = fractional drift
PS = [0.15, 0.3, 0.5, 1.0, 2.0]
SEEDS_B = (0.01, 0.04, 0.10)


def beta_of(case, b_pp, b_e, p=1.0):
    e = np.asarray(case["losses"]["e_oft"], float)
    Eh = (e - e[0]) / (e[-1] - e[0])
    return b_pp * (1.0 + b_e * Eh ** p)


def mism(a_pp, a_f, b_pp, b_e, case, p=1.0):
    t = np.asarray(case["t_bhpt"], float)
    alpha = a_pp * (1.0 + a_f * case["losses"]["flux_hat"])
    beta = beta_of(case, b_pp, b_e, p)
    if np.any(beta <= 0): return 50.0, np.nan
    bc = G.creative.cumulative_trapezoid(beta, t)
    tau = bc - float(np.interp(G.T_ANCHOR, t, bc))
    if np.min(np.diff(tau)) <= 0: return 50.0, np.nan
    f = lambda s: GG.err_at_t0(s, alpha, tau, case["h_bhpt"], case["t_nr"], case["h_nr"])
    r = minimize_scalar(f, bounds=(-195., 45.), method="bounded",
                        options={"xatol": 1e-3, "maxiter": 60})
    mean_b = float(np.trapezoid(beta, t) / (t[-1] - t[0]))
    return float(r.fun), mean_b


def opt3(case, pref, b_e, p=1.0):
    best = (np.inf, None)
    for s in (1.0, 0.98):
        x0 = np.array([pref[0] * s, pref[1], pref[2]])
        r = minimize(lambda u: mism(u[0], u[1], u[2], b_e, case, p)[0], x0,
                     method="Nelder-Mead",
                     options={"maxiter": 300, "xatol": 1e-6, "fatol": 1e-13})
        if r.fun < best[0]: best = (float(r.fun), r.x.copy())
    e, m = mism(*best[1], b_e, case, p)
    return e, best[1], m


for q in QS:
    case = FL.add_flux(G.load_case(q, 6, 10))
    pref = FA.params_at(q, TH, DEG)
    e = np.asarray(case["losses"]["e_oft"], float); Etot = float(e[-1] - e[0])
    print(f"\n########## q = {q:g}  (shipped: drift={pref[3]*Etot*100:+.2f}%, "
          f"b_PP={pref[2]:.5f}) ##########")
    print("TEST A: scan b_E, re-optimise a_PP/a_F/b_PP on the FULL window")
    print(f"{'b_E':>6s} {'dbeta':>8s} {'mismatch':>11s} {'rel':>6s} {'b_PP fit':>9s} "
          f"{'mean beta':>10s} {'b_start':>8s} {'b_end':>8s}")
    rowsA = []
    for be in BE:
        er, x, mb = opt3(case, pref, be)
        bs = x[2] * 1.0; bend = x[2] * (1.0 + be)
        rowsA.append((be, er, x[2], mb))
        print(f"{be:+6.3f} {be*100:+7.2f}% {er:11.4e} {'':>6s} {x[2]:9.5f} "
              f"{mb:10.6f} {bs:8.5f} {bend:8.5f}", flush=True)
    best = min(r[1] for r in rowsA)
    mbs = np.array([r[3] for r in rowsA])
    print(f"   mean beta across the whole b_E scan: {mbs.mean():.6f} "
          f"+- {mbs.std()/mbs.mean()*100:.3f}%   (rel mismatch "
          f"{'/'.join(f'{r[1]/best:.1f}' for r in rowsA)})")

    print("TEST B: beta = b_PP(1 + b_E*Ehat^p), b_E > 0, GRID (no nested optimiser)")
    BEP = [0.01, 0.02, 0.04, 0.08]
    print(f"{'p':>6s} | " + " ".join(f"drift{b*100:+5.1f}%" for b in BEP)
          + " | best drift  mismatch  mean beta")
    for pw in PS:
        vals, means, bpps = [], [], []
        for be in BEP:
            er, x, mb = opt3(case, pref, be, pw)
            vals.append(er); means.append(mb); bpps.append(x[2])
        j = int(np.argmin(vals))
        print(f"{pw:6.2f} | " + " ".join(f"{v:11.4e}" for v in vals)
              + f" | {BEP[j]*100:+9.1f}% {vals[j]:9.4e} {means[j]:10.6f}", flush=True)
