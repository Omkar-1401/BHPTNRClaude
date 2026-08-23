"""gwr_anchored2 -- alpha and beta as anchor-to-anchor interpolations in the radiated
energy fraction, with ONE fitted coefficient.

    Ehat(t) = Eoft(t)/E_tot                        in [0,1], from gw_remnant

    alpha(t,q) = X1^(6/5) * [ 1 + w_a(nu) * ga(q) * Ehat ]
    beta (t,q) = X1^(6/5) * [ 1 + w_b     * gb(q) * Ehat ]

DERIVED anchors (nothing fitted):
    A0 = B0 = X1^(6/5)              = (nu/eta_pp)^(3/5)   -- Newtonian chirp/amplitude
    A1 = max|h_NR| / max|h_pp|       -- peak amplitude ratio (measured; RemS L_peak,
                                        omega_peak would predict it)
    B1 = omega_Schw / omega_220(Mf, chif)   -- ringdown QNM ratio, matches the measured
                                        ringdown beta to <1% (see beta_drift_tests/
                                        ringdown_beta_vs_qnm.py)
    ga = (A1-A0)/A0   (negative: alpha FALLS)     gb = (B1-B0)/B0   (positive: beta RISES)

FITTED:  w_a = 1 + c*nu^2      one coefficient.   w_b = +1 held fixed.

Why w_b is held: gb > 0 at every q in [2,8] (checked), so w_b > 0 makes beta rise by
construction -- the requirement is satisfied without a constraint.  A freely fitted
w_b(nu) would have to cross zero near q~4 to match the old L2 fits, reproducing exactly
the `P`-crosses-zero pathology that wrecks low-q extrapolation.  If w_b = 1 proves too
costly, the fallback is w_b = 1 + c_b*nu^2 held POSITIVE.

Everything is PP-anchored for free: ga, gb -> 0 as nu -> 0, so alpha, beta -> X1^(6/5) -> 1.

Usage:  python fit_scaling_gwr_anchored2.py [--wb 1.0] [--fit-wb]
"""
from __future__ import annotations

import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar, minimize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_global as GG
import fit_scaling_gwr_energy_flux as FL
import BHPTNRPNAnchored as PNA

RESULTS = ROOT / "gwr_anchored2_results"; RESULTS.mkdir(exist_ok=True)
IN_Q = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
LOW_Q = (2.75, 2.5, 2.25, 2.0)
W_SCHW = 0.3737                    # Schwarzschild (2,2,0); measured omega_pp = 0.3740

REF = {"fluxanchored (7c)": {"median": 4.7516e-04, "max": 7.1503e-04, 2.75: 6.3465e-04,
                             2.5: 5.9369e-04, 2.25: 6.8673e-04, 2.0: 1.3056e-03},
       "anchored (6c)":     {"median": 6.2551e-04, "max": 9.7884e-04, 2.75: 9.4774e-04,
                             2.5: 9.2793e-04, 2.25: 1.0163e-03, 2.0: 1.5293e-03}}


_AN = None
def _prefac(q, mode):
    """a_pp, b_pp.  'bare' = X1^(6/5) exactly; 'anchored' = with the fitted corrections
    from gwr_energy_anchored, so the couplings can be tested in isolation."""
    global _AN
    X1 = q / (1.0 + q); base = X1 ** 1.2; nu = q / (1.0 + q) ** 2
    if mode == "bare":
        return base, base
    if _AN is None:
        _AN = json.loads((ROOT / "gwr_energy_anchored_results" / "coeffs.json").read_text())
    c0, c1 = _AN["c"]; b = _AN["b"]
    return base * (1.0 + c0 * nu + c1 * nu ** 2), base * (1.0 + b * nu)


def anchors(q, case):
    """A0=B0, ga, gb, E_tot -- all derived or measured, nothing fitted."""
    X1 = q / (1.0 + q)
    A0 = X1 ** 1.2
    A1 = float(np.max(np.abs(case["h_nr"])) / np.max(np.abs(case["h_bhpt"])))
    mf, chif = PNA._remnant(q)
    qn = PNA._cfg()["qnm"]
    w_nr = (qn["F1"] + qn["F2"] * (1.0 - chif) ** qn["F3"]) / mf
    B1 = W_SCHW / w_nr
    e = np.asarray(case["losses"]["e_oft"], float)
    return A0, (A1 - A0) / A0, (B1 - A0) / A0, float(e[-1] - e[0]), A1, B1


def params_at(q, case, c, w_b, prefac="bare"):
    A0, ga, gb, E_tot, _, _ = anchors(q, case)
    nu = q / (1.0 + q) ** 2
    w_a = 1.0 + c * nu ** 2
    a_pp, b_pp = _prefac(q, prefac)
    # couplings are relative to the ANCHOR baseline A0, so rescale onto the prefactor
    # actually in use; the evaluator couples to RAW e_oft, hence the /E_tot.
    return np.array([a_pp, w_a * ga * A0 / a_pp / E_tot,
                     b_pp, w_b * gb * A0 / b_pp / E_tot])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--fit-wb", action="store_true",
                    help="also fit w_b = 1 + c_b*nu^2, held positive (the fallback)")
    ap.add_argument("--prefac", default="bare", choices=("bare", "anchored"),
                    help="bare = X1^(6/5); anchored = keep the fitted prefactor corrections")
    ap.add_argument("--src", type=int, default=6)
    ap.add_argument("--nr", type=int, default=10)
    args = ap.parse_args()

    mult = json.loads(G.RESULTS_DIR.joinpath("per_q_cache_mult.json").read_text())
    rows = sorted(mult.values(), key=lambda r: r["q"])
    t0_poly = np.polyfit(G.get_nu(np.array([r["q"] for r in rows])),
                         np.array([r["params"] for r in rows])[:, 4], 3)

    cases = {}
    def case_for(q):
        if q not in cases:
            cases[q] = FL.add_flux(G.load_case(q, args.src, args.nr))
        return cases[q]

    print("anchors (derived / measured, nothing fitted):")
    print(f"{'q':>5s} {'A0=B0':>8s} {'A1':>8s} {'ga':>9s} {'B1':>8s} {'gb':>9s} {'E_tot':>8s}")
    for q in list(IN_Q) + list(LOW_Q):
        A0, ga, gb, Et, A1, B1 = anchors(q, case_for(q))
        print(f"{q:5.2f} {A0:8.5f} {A1:8.5f} {ga:+9.4f} {B1:8.5f} {gb:+9.4f} {Et:8.5f}",
              flush=True)

    def err_at(q, c, w_b):
        p = params_at(q, case_for(q), c, w_b, args.prefac)
        seed = float(np.polyval(t0_poly, G.get_nu(q)))
        e, t0 = GG.fast_mismatch(p, case_for(q), seed, "fit")
        return GG.fast_mismatch(p, case_for(q), t0, "full")[0]

    if args.fit_wb:
        def obj(v):
            c, cb = v
            wb = lambda q: max(1e-6, 1.0 + cb * (q / (1 + q) ** 2) ** 2)
            return float(np.mean([err_at(q, c, wb(q)) for q in IN_Q]))
        r = minimize(obj, [7.51, 0.0], method="Nelder-Mead",
                     options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-10})
        c_fit, cb_fit = r.x
        print(f"\nfitted: c = {c_fit:.3f}   c_b = {cb_fit:.3f}  (w_b = 1 + c_b nu^2)")
        wb_of = lambda q: max(1e-6, 1.0 + cb_fit * (q / (1 + q) ** 2) ** 2)
    else:
        def obj1(c):
            return float(np.mean([err_at(q, c, args.wb) for q in IN_Q]))
        r = minimize_scalar(obj1, bounds=(-20.0, 60.0), method="bounded",
                            options={"xatol": 1e-3})
        c_fit, cb_fit = float(r.x), None
        print(f"\nfitted: c = {c_fit:.3f}   (w_a = 1 + c nu^2),  w_b = {args.wb} held")
        wb_of = lambda q: args.wb

    final = {q: err_at(q, c_fit, wb_of(q)) for q in list(IN_Q) + list(LOW_Q)}
    med = float(np.median([final[q] for q in IN_Q]))
    mx = float(np.max([final[q] for q in IN_Q]))
    print(f"\n{'q':>6s} {'anchored2':>12s}  |  " + "  ".join(f"{k:>18}" for k in REF))
    for q in IN_Q:
        print(f"{q:>6g} {final[q]:>12.4e}")
    print(f"{'median':>6s} {med:>12.4e}  |  "
          + "  ".join(f"{REF[k]['median']:>18.4e}" for k in REF))
    print(f"{'max':>6s} {mx:>12.4e}  |  " + "  ".join(f"{REF[k]['max']:>18.4e}" for k in REF))
    print("  --- q < 3 (held out) ---")
    for q in LOW_Q:
        print(f"{q:>6g} {final[q]:>12.4e}  |  "
              + "  ".join(f"{REF[k][q]:>15.4e} {'B' if final[q]<REF[k][q] else 'w'}"
                          for k in REF))

    # verify beta rises at every q
    print("\n[verify] beta drift over the window (should be POSITIVE at every q):")
    for q in list(IN_Q) + list(LOW_Q):
        p = params_at(q, case_for(q), c_fit, wb_of(q), args.prefac)
        Et = anchors(q, case_for(q))[3]
        print(f"   q={q:<5g} dbeta = {p[3]*Et*100:+7.3f}%    dalpha = {p[1]*Et*100:+8.3f}%")

    (RESULTS / "coeffs.json").write_text(json.dumps(
        {"model": "gwr_anchored2", "prefac": args.prefac, "n_fitted": 1 if cb_fit is None else 2,
         "c_w_a": c_fit, "c_w_b": cb_fit, "w_b_held": None if cb_fit is not None else args.wb,
         "form": {"alpha": "X1^(6/5)*(1 + (1+c*nu^2)*ga(q)*Ehat)",
                  "beta": "X1^(6/5)*(1 + w_b*gb(q)*Ehat)",
                  "ga": "(A1-A0)/A0, A1 = peak amplitude ratio",
                  "gb": "(B1-A0)/A0, B1 = omega_Schw/omega_220(Mf,chif)"},
         "in_range": {"median": med, "max": mx},
         "low_q": {f"{q:g}": final[q] for q in LOW_Q},
         "reference": REF}, indent=2, default=float))
    print(f"\nwrote {RESULTS/'coeffs.json'}")


if __name__ == "__main__":
    main()
