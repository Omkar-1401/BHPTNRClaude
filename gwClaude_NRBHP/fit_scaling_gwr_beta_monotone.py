"""gwr_beta_monotone -- fluxanchored with beta GUARANTEED NON-DECREASING at every q.

Requirement (user, 2026-08-11): beta(t) must never decrease at any q; E(t) must remain
beta's drive; no post-merger correction / gate.

Since E(t) is monotone increasing, `sign(dbeta/dt) = sign(P)` with
`P = beta_PP * b_E = dbeta/dE`.  So "beta never decreases" is exactly `P(nu) >= 0`, and
the whole job is to impose that in P's nu-form.

Measured target.  Constrained per-q optima are `max(0, b_E_free)` (the free values are
already positive below q~3.8 and the `b_E >= 0` fit lands on exactly zero above q=4 --
see RESUME_gwremnant.md).  Over the four positive training points, P is astonishingly
linear in (nu - nu_c):  P/(nu - nu_c) = 40.30 / 40.33 / 40.27 / 40.33, constant to 0.15%.
So the form is a RECTIFIED LINE, not a polynomial:

    P(nu) = |a| * max(0, nu - nu_c)          a ~ 40.3,  nu_c ~ 0.16479  (q_c = 3.806)

  * `P >= 0` by construction -> beta non-decreasing at EVERY q, for free.
  * 2 coefficients, the same count as the shipped `P = P0 + P1*nu`, so this is a swap,
    not an extra parameter: still 7 coefficients total.
  * PP-anchored for free: nu -> 0 lies below nu_c, so P -> 0 in the test-mass limit.
  * The kink at q_c = 3.81 is INSIDE the training range; q < 3 extrapolates along the
    linear branch (larger nu), so the kink cannot affect the low-q behaviour.

Everything else is fluxanchored, unchanged:

    F(t) = Edot/max(Edot)
    alpha(t) = alpha_PP(q) * (1 + alpha_F(q) * F(t))
    beta (t) = beta_PP(q)  * (1 + (P(nu)/beta_PP) * E(t))
    alpha_PP = X1^1.2 * (1 + c0*nu + c1*nu^2)   [2]
    alpha_F  = A0*nu + A1*nu^2                  [2]
    beta_PP  = X1^1.2 * (1 + b*nu)              [1]
    P        = |a| * max(0, nu - nu_c)          [2]   <- the only change

Cost expected from the per-q constrained fits: in-range median 4.69e-04 -> ~7.1e-04.
Above q_c beta is FLAT (not rising); a rise at every q needs the gate (route C).

Usage:  python fit_scaling_gwr_beta_monotone.py [--global] [--maxiter 80]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.optimize import minimize

import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_global as GG
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_gwr_energy_fluxanchored as FA

from pathlib import Path
from pathlib import Path as pathlib_Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "gwr_beta_monotone_results"
RESULTS.mkdir(exist_ok=True)
COEFFS = RESULTS / "coeffs.json"

LOW_Q = (2.75, 2.5, 2.25, 2.0)
IN_Q = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0)

# The two bases this can sit on.  Only P's nu-form differs from the parent in each case,
# so the measured cost is attributable to `P >= 0` alone.
#   flux : alpha = alpha_PP*(1 + alpha_F*F),  F = Edot/max(Edot)   parent fluxanchored, 7c
#   E    : alpha = alpha_PP*(1 + alpha_E*E)                        parent anchored,     6c
BASES = {
    "flux": dict(parent="fluxanchored (7c, global)", mm=lambda: FL.fast_mismatch,
                 perq=lambda: FL.RESULTS / "per_q_cache_flux.json", adeg=2,
                 coeffs="coeffs.json",
                 ref={"median": 4.7516e-04, "max": 7.1503e-04, 2.75: 6.3465e-04,
                      2.5: 5.9369e-04, 2.25: 6.8673e-04, 2.0: 1.3056e-03}),
    "E": dict(parent="anchored (6c, global)", mm=lambda: GG.fast_mismatch,
              perq=lambda: G.RESULTS_DIR / "per_q_cache_mult.json", adeg=0,
              coeffs="coeffs_Ebase.json",
              ref={"median": 6.2551e-04, "max": 9.7884e-04, 2.75: 9.4774e-04,
                   2.5: 9.2793e-04, 2.25: 1.0163e-03, 2.0: 1.5293e-03}),
}

nu_of = lambda q: np.asarray(q, float) / (1.0 + np.asarray(q, float)) ** 2
base_of = lambda q: (np.asarray(q, float) / (1.0 + np.asarray(q, float))) ** 1.2
NU_C_BOUNDS = (0.02, 0.245)


def P_of(nu, a, nu_c):
    """Rectified line.  |a| and the max() together guarantee P >= 0 for every nu."""
    nu_c = float(np.clip(nu_c, *NU_C_BOUNDS))
    return abs(float(a)) * max(0.0, float(nu) - nu_c)


def params_at(q, theta, adeg=2, base="flux"):
    """[a_pp, a_c, b_pp, b_e].  a_c is alpha_F (flux base) or alpha_E (E base);
    beta is identical in both, with P >= 0 by construction."""
    na = adeg if base == "flux" else adeg + 1
    c = theta[0:2]
    A = theta[2:2 + na]
    b = theta[2 + na]
    a_P, nu_c = theta[3 + na], theta[4 + na]
    nu = float(nu_of(q)); bs = float(base_of(q))
    a_pp = float(np.clip(bs * (1.0 + c[0] * nu + c[1] * nu ** 2), 0.05, 2.5))
    b_pp = float(np.clip(bs * (1.0 + b * nu), 0.2, 1.6))
    if base == "flux":
        a_c = float(sum(A[k] * nu ** (k + 1) for k in range(len(A))))   # -> 0 as nu->0
    else:
        a_c = float(sum(A[k] * nu ** k for k in range(len(A)))) / nu    # alpha_E ~ 1/nu
    b_e = P_of(nu, a_P, nu_c) / b_pp
    return np.array([a_pp, a_c, b_pp, b_e])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global", dest="do_global", action="store_true")
    ap.add_argument("--base", default="flux", choices=list(BASES),
                    help="flux = Edot alpha (parent fluxanchored, 7c); "
                         "E = E alpha (parent anchored, 6c)")
    ap.add_argument("--aF-deg", type=int, default=None,
                    help="alpha term degree; default 2 for flux base, 0 for E base")
    ap.add_argument("--ntrain", type=int, default=12)
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--src", type=int, default=6)
    ap.add_argument("--nr", type=int, default=10)
    args = ap.parse_args()
    B = BASES[args.base]
    adeg = B["adeg"] if args.aF_deg is None else args.aF_deg
    MM = B["mm"]()
    REF = {B["parent"]: B["ref"]}
    coeffs_path = RESULTS / B["coeffs"]
    na = adeg if args.base == "flux" else adeg + 1

    mult = json.loads(G.RESULTS_DIR.joinpath("per_q_cache_mult.json").read_text())
    rows = sorted(mult.values(), key=lambda r: r["q"])
    q_all = np.array([r["q"] for r in rows], float)
    pa = np.array([r["params"] for r in rows], float)
    t0_poly = np.polyfit(G.get_nu(q_all), pa[:, 4], 3)

    cases = {}
    def case_for(q):
        if q not in cases:
            cases[q] = FL.add_flux(G.load_case(q, args.src, args.nr))
        return cases[q]

    # ---- seed from the flux per-q cache -----------------------------------------
    cache = json.loads(pathlib_Path(B["perq"]()).read_text())
    pq = sorted(cache.values(), key=lambda r: r["q"])
    nus = np.array([nu_of(r["q"]) for r in pq])
    V = np.array([r["params"] for r in pq], float)
    bs = base_of(np.array([r["q"] for r in pq]))
    P_meas = V[:, 2] * V[:, 3]

    c_seed = np.linalg.lstsq(np.vstack([nus, nus ** 2]).T, V[:, 0] / bs - 1.0,
                             rcond=None)[0]
    if args.base == "flux":
        A_basis = np.vstack([nus ** (k + 1) for k in range(na)]).T
        A_seed = np.linalg.lstsq(A_basis, V[:, 1], rcond=None)[0]
    else:                                    # alpha_E ~ A(nu)/nu  ->  fit A = alpha_E*nu
        A_basis = np.vstack([nus ** k for k in range(na)]).T
        A_seed = np.linalg.lstsq(A_basis, V[:, 1] * nus, rcond=None)[0]
    b_seed = np.linalg.lstsq(nus[:, None], V[:, 2] / bs - 1.0, rcond=None)[0]
    # rectified-line seed: line through the POSITIVE-P points, slope and root
    m = P_meas > 0
    sl, ic = np.polyfit(nus[m], P_meas[m], 1)
    theta = np.concatenate([c_seed, A_seed, b_seed, [abs(sl), -ic / sl]])
    ncoef = len(theta)
    print(f"[beta_monotone] base={args.base}  {ncoef} coefficients "
          f"(alpha_PP 2, alpha_{'F' if args.base == 'flux' else 'E'} {na}, "
          f"beta_PP 1, P 2 = |a|*max(0,nu-nu_c))")
    q_c = (lambda n: (-(2 * n - 1) + np.sqrt(max((2 * n - 1) ** 2 - 4 * n * n, 0)))
           / (2 * n))(-ic / sl)
    print(f"[beta_monotone] P seed from {m.sum()} positive per-q points: "
          f"a={abs(sl):.3f}  nu_c={-ic/sl:.5f}  (q_c={q_c:.3f})", flush=True)

    def run_all(th):
        out = {}
        for q in list(IN_Q) + list(LOW_Q):
            seed = float(np.polyval(t0_poly, G.get_nu(q)))
            p = params_at(q, th, adeg, args.base)
            e, t0 = MM(p, case_for(q), seed, "fit")
            out[q] = MM(p, case_for(q), t0, "full")[0]
        return out

    seeded = run_all(theta)
    theta_seeded = theta.copy()

    if args.do_global:
        tg = GG.even_nu_targets(q_all, args.ntrain)
        train_q = sorted({float(q_all[np.argmin(np.abs(q_all - t))]) for t in tg})
        tcases = {q: case_for(q) for q in train_q}
        nuis = {q: float(np.polyval(t0_poly, G.get_nu(q))) for q in train_q}

        def obj(th):
            errs = []
            for q in train_q:
                e, t0 = MM(params_at(q, th, adeg, args.base),
                           tcases[q], nuis[q], "fit")
                nuis[q] = t0
                errs.append(e)
            errs = np.array(errs)
            obj.last = (float(errs.mean()), float(errs.max()))
            return errs.mean()
        obj.last = (np.nan, np.nan)

        obj(theta)
        print(f"\n[global] train q: {[round(q, 2) for q in train_q]}")
        print(f"[global] seed: mean={obj.last[0]:.4e} max={obj.last[1]:.4e}", flush=True)
        t_s = time.time()
        res = minimize(obj, theta, method="Powell",
                       options={"maxiter": args.maxiter, "xtol": 1e-4, "ftol": 1e-6})
        print(f"[global] opt {time.time() - t_s:.0f}s: mean={obj.last[0]:.4e} "
              f"max={obj.last[1]:.4e}", flush=True)
        theta = res.x

    final = run_all(theta)
    med = float(np.median([final[q] for q in IN_Q]))
    mx = float(np.max([final[q] for q in IN_Q]))

    print(f"\n{'q':>6} {'seeded':>12} {'monotone-beta':>14}  |  "
          + "  ".join(f"{k:>26}" for k in REF))
    for q in IN_Q:
        print(f"{q:>6g} {seeded[q]:>12.4e} {final[q]:>14.4e}")
    print(f"{'median':>6} {'':>12} {med:>14.4e}  |  "
          + "  ".join(f"{REF[k]['median']:>26.4e}" for k in REF))
    print(f"{'max':>6} {'':>12} {mx:>14.4e}  |  "
          + "  ".join(f"{REF[k]['max']:>26.4e}" for k in REF))
    print("  --- q < 3 (held out) ---")
    for q in LOW_Q:
        marks = "  ".join(f"{REF[k][q]:>19.4e} "
                          f"{'B' if final[q] < REF[k][q] else 'w'}" for k in REF)
        print(f"{q:>6g} {seeded[q]:>12.4e} {final[q]:>14.4e}  |  " + marks)

    # ---- the requirement itself: verify beta is non-decreasing everywhere --------
    a_P, nu_c = theta[3 + na], theta[4 + na]
    print(f"\n[verify] fitted P(nu) = {abs(a_P):.4f} * max(0, nu - "
          f"{float(np.clip(nu_c, *NU_C_BOUNDS)):.5f})")
    qs = np.concatenate([np.arange(2.0, 10.01, 0.25), [12.0, 20.0, 100.0, 1e4]])
    Ps = np.array([P_of(float(nu_of(q)), a_P, nu_c) for q in qs])
    bad = qs[Ps < 0]
    print(f"[verify] min P over q in [2, 1e4]: {Ps.min():.3e}   "
          f"violations: {len(bad)}  -> beta is "
          f"{'NON-DECREASING at every q' if len(bad) == 0 else 'DECREASING somewhere!'}")
    qz = qs[Ps > 0]
    print(f"[verify] beta strictly rises for q <= {qz.max():.2f} and is flat above")

    coeffs_path.write_text(json.dumps(
        {"model": "gwr_beta_monotone", "n_coefficients": int(ncoef),
         "base": args.base, "alpha_deg": na, "global_refit": bool(args.do_global),
         "requirement": "beta non-decreasing at every q; E(t) is beta's drive; no gate",
         "form": {"alpha": "alpha_PP(q)*(1+alpha_F(q)*F(t)), F=Edot/max(Edot)",
                  "alpha_PP": "X1^(6/5)*(1+c0*nu+c1*nu^2)",
                  "alpha_term": ("A0*nu(+A1*nu^2)" if args.base == "flux"
                                else "A(nu)/nu"),
                  "beta_PP": "X1^(6/5)*(1+b*nu)",
                  "P": "|a|*max(0, nu-nu_c)  -> P>=0 BY CONSTRUCTION"},
         "c": theta[0:2].tolist(), "A": theta[2:2 + na].tolist(),
         "b": float(theta[2 + na]),
         "P_a": float(abs(theta[3 + na])),
         "P_nu_c": float(np.clip(theta[4 + na], *NU_C_BOUNDS)),
         "in_range": {"median": med, "max": mx},
         "low_q": {f"{q:g}": final[q] for q in LOW_Q},
         "min_P_over_q_2_to_1e4": float(Ps.min()),
         "seeded": {"c": theta_seeded[0:2].tolist(),
                    "A": theta_seeded[2:2 + na].tolist(),
                    "b": float(theta_seeded[2 + na]),
                    "P_a": float(abs(theta_seeded[3 + na])),
                    "P_nu_c": float(theta_seeded[4 + na]),
                    "in_range": {"median": float(np.median([seeded[q] for q in IN_Q])),
                                 "max": float(np.max([seeded[q] for q in IN_Q]))},
                    "low_q": {f"{q:g}": seeded[q] for q in LOW_Q}},
         "reference": REF}, indent=2, default=float))
    print(f"\nwrote {coeffs_path}")


if __name__ == "__main__":
    main()
