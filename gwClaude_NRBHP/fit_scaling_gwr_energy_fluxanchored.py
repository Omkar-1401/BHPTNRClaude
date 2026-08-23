"""
Flux-coupled alpha ON the anchored regression layer -- the combination.

The two previous tests separated cleanly:
  * `gwr_energy_flux`     (flux alpha, uniform deg-3, 14 coef): in-range median
    4.70e-04 vs 6.26e-04 for the E-coupled equivalent -- alpha's COORDINATE controls
    in-band accuracy -- but low-q unchanged/worse, because the deg-3 layer governs it.
  * `gwr_energy_anchored` (E alpha, anchored, 6 coef): q=2 1.53e-03 vs 3.79e-03 --
    the REGRESSION LAYER controls extrapolation -- but in-range unchanged at 6.26e-04.

They act on different failure modes, so this pairs them:

    F(t) = Edot / max(Edot)                       peak-normalised instantaneous flux

    alpha(t) = alpha_PP(q) * (1 + alpha_F(q) * F(t))
    beta (t) = beta_PP(q)  +  P(nu) * E(t)

    alpha_PP(q) = X1**1.2 * (1 + c0*nu + c1*nu**2)     2 free   [derived base]
    alpha_F(q)  = A0*nu (+ A1*nu**2)                   1-2 free [PP-anchored by form]
    beta_PP(q)  = X1**1.2 * (1 + b*nu)                 1 free   [derived base]
    P(nu)       = P0 + P1*nu                           2 free   [empirical]

Note what the flux coordinate buys structurally, beyond accuracy.  Because F is
peak-normalised, `alpha_F` IS essentially `d ln alpha` across the window -- the physical
quantity measured earlier to scale as nu^1.635.  The per-q fits confirm it: alpha_F runs
-0.3213 (q=3) to -0.1206 (q=8), i.e. ~nu^1.53.  So `alpha_F = A0*nu + A1*nu^2` vanishes
as nu -> 0 **by construction**, giving the point-particle limit for free.  The E-coupled
`alpha_E` had to carry nu^-1 to do the same job, a power no polynomial represents and
whose justification did not survive measurement (see scaling_gwr_energy_stiff.md).  So
this is also the more honest parameterisation.

Usage:  python fit_scaling_gwr_energy_fluxanchored.py --global --maxiter 80
        python fit_scaling_gwr_energy_fluxanchored.py --global --aF-deg 1   # 6 coef
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_global as GG
import fit_scaling_gwr_energy_flux as FL

RESULTS = ROOT / "gwr_energy_fluxanchored_results"
RESULTS.mkdir(exist_ok=True)
COEFFS = RESULTS / "coeffs.json"
PERQ_CACHE = FL.RESULTS / "per_q_cache_flux.json"
SEED_CACHE = G.RESULTS_DIR / "per_q_cache_mult.json"

LOW_Q = (2.75, 2.5, 2.25, 2.0)
IN_Q = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0)

# the two parents
REF = {
    "anchored (E alpha, 6c)": {"median": 6.2551e-04, "max": 9.7884e-04,
                               2.75: 9.4774e-04, 2.5: 9.2793e-04,
                               2.25: 1.0163e-03, 2.0: 1.5293e-03},
    "flux (deg-3, 14c)":      {"median": 4.6993e-04, "max": 7.1604e-04,
                               2.75: 7.0893e-04, 2.5: 1.7208e-03,
                               2.25: 8.7024e-03, 2.0: 4.6005e-02},
}

nu_of = lambda q: np.asarray(q, float) / (1.0 + np.asarray(q, float)) ** 2
base_of = lambda q: (np.asarray(q, float) / (1.0 + np.asarray(q, float))) ** 1.2


def params_at(q, theta, aF_deg):
    c = theta[0:2]
    A = theta[2:2 + aF_deg]
    b = theta[2 + aF_deg]
    P = theta[3 + aF_deg:5 + aF_deg]
    nu = float(nu_of(q)); bs = float(base_of(q))
    a_pp = float(np.clip(bs * (1.0 + c[0] * nu + c[1] * nu ** 2), 0.05, 2.5))
    b_pp = float(np.clip(bs * (1.0 + b * nu), 0.2, 1.6))
    a_f = float(sum(A[k] * nu ** (k + 1) for k in range(len(A))))
    b_e = float(P[0] + P[1] * nu) / b_pp
    return np.array([a_pp, a_f, b_pp, b_e])


def build_perq(q_list, case_for, t0_poly, be_poly, force=False):
    cache = {} if force or not PERQ_CACHE.exists() else json.loads(PERQ_CACHE.read_text())
    changed = False
    for q in q_list:
        k = f"{q:.10f}"
        if k in cache:
            continue
        nu = G.get_nu(q)
        e, p, t0 = FL.perq_fit(q, case_for(q), float(np.polyval(t0_poly, nu)),
                               float(np.polyval(be_poly, nu)))
        cache[k] = {"q": q, "params": p.tolist(), "err": e, "t0": t0}
        changed = True
        print(f"   q={q:<6.3g} err={e:.4e}  a_PP={p[0]:.4f} a_F={p[1]:+.4f} "
              f"b_PP={p[2]:.4f} b_E={p[3]:+.4f}", flush=True)
    if changed:
        PERQ_CACHE.write_text(json.dumps(cache, indent=2))
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global", dest="do_global", action="store_true")
    ap.add_argument("--aF-deg", type=int, default=2, choices=(1, 2))
    ap.add_argument("--ntrain", type=int, default=12)
    ap.add_argument("--nperq", type=int, default=16)
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--src", type=int, default=6)
    ap.add_argument("--nr", type=int, default=10)
    args = ap.parse_args()

    mult = json.loads(SEED_CACHE.read_text())
    rows = sorted(mult.values(), key=lambda r: r["q"])
    q_all = np.array([r["q"] for r in rows], float)
    pa = np.array([r["params"] for r in rows], float)
    t0_poly = np.polyfit(G.get_nu(q_all), pa[:, 4], 3)
    be_poly = np.polyfit(G.get_nu(q_all), pa[:, 3], 3)

    cases = {}
    def case_for(q):
        if q not in cases:
            cases[q] = FL.add_flux(G.load_case(q, args.src, args.nr))
        return cases[q]

    tg = GG.even_nu_targets(q_all, args.nperq)
    perq_q = sorted({float(q_all[np.argmin(np.abs(q_all - t))]) for t in tg})
    print(f"[fluxanchored] per-q fits (flux alpha) at {len(perq_q)} mass ratios ...",
          flush=True)
    cache = build_perq(perq_q, case_for, t0_poly, be_poly)

    nus = np.array([G.get_nu(q) for q in perq_q])
    V = np.array([cache[f"{q:.10f}"]["params"] for q in perq_q], float)
    bs = base_of(np.array(perq_q))

    # seed each anchored form by least squares against the per-q values
    c_seed = np.linalg.lstsq(np.vstack([nus, nus ** 2]).T, V[:, 0] / bs - 1.0,
                             rcond=None)[0]
    A_basis = np.vstack([nus ** (k + 1) for k in range(args.aF_deg)]).T
    A_seed = np.linalg.lstsq(A_basis, V[:, 1], rcond=None)[0]
    b_seed = np.linalg.lstsq(nus[:, None], V[:, 2] / bs - 1.0, rcond=None)[0]
    P_seed = np.polyfit(nus, V[:, 2] * V[:, 3], 1)[::-1]
    theta = np.concatenate([c_seed, A_seed, b_seed, P_seed])
    ncoef = len(theta)
    print(f"[fluxanchored] {ncoef} coefficients "
          f"(alpha_PP 2, alpha_F {args.aF_deg}, beta_PP 1, P 2)", flush=True)

    def run_all(th):
        out = {}
        for q in list(IN_Q) + list(LOW_Q):
            seed = float(np.polyval(t0_poly, G.get_nu(q)))
            p = params_at(q, th, args.aF_deg)
            e, t0 = FL.fast_mismatch(p, case_for(q), seed, "fit")
            out[q] = FL.fast_mismatch(p, case_for(q), t0, "full")[0]
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
                e, t0 = FL.fast_mismatch(params_at(q, th, args.aF_deg),
                                         tcases[q], nuis[q], "fit")
                nuis[q] = t0
                errs.append(e)
            errs = np.array(errs)
            obj.last = (float(errs.mean()), float(errs.max()))
            return errs.mean()
        obj.last = (np.nan, np.nan)

        obj(theta)
        print(f"\n[global] train q: {[round(q, 2) for q in train_q]}", flush=True)
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

    print(f"\n{'q':>6} {'seeded':>12} {'flux+anchored':>14}  |  "
          + "  ".join(f"{k:>22}" for k in REF))
    for q in IN_Q:
        print(f"{q:>6g} {seeded[q]:>12.4e} {final[q]:>14.4e}")
    print(f"{'median':>6} {'':>12} {med:>14.4e}  |  "
          + "  ".join(f"{REF[k]['median']:>22.4e}" for k in REF))
    print(f"{'max':>6} {'':>12} {mx:>14.4e}  |  "
          + "  ".join(f"{REF[k]['max']:>22.4e}" for k in REF))
    print("  --- q < 3 (held out) ---")
    wins = {k: 0 for k in REF}
    for q in LOW_Q:
        marks = []
        for k in REF:
            better = final[q] < REF[k][q]
            wins[k] += better
            marks.append(f"{REF[k][q]:>15.4e} {'B' if better else 'w'}")
        print(f"{q:>6g} {seeded[q]:>12.4e} {final[q]:>14.4e}  |  " + "  ".join(marks))

    COEFFS.write_text(json.dumps(
        {"model": "gwr_energy_fluxanchored", "n_coefficients": int(ncoef),
         "aF_degree": args.aF_deg, "global_refit": bool(args.do_global),
         "form": {"alpha": "alpha_PP(q)*(1+alpha_F(q)*F(t)), F=Edot/max(Edot)",
                  "alpha_PP": "X1^(6/5)*(1+c0*nu+c1*nu^2)",
                  "alpha_F": "A0*nu(+A1*nu^2)  -> 0 as nu->0 BY CONSTRUCTION",
                  "beta_PP": "X1^(6/5)*(1+b*nu)", "P": "P0+P1*nu"},
         "c": theta[0:2].tolist(), "A": theta[2:2 + args.aF_deg].tolist(),
         "b": float(theta[2 + args.aF_deg]),
         "P": theta[3 + args.aF_deg:5 + args.aF_deg].tolist(),
         "in_range": {"median": med, "max": mx},
         "low_q": {f"{q:g}": final[q] for q in LOW_Q},
         "seeded": {
             "c": theta_seeded[0:2].tolist(),
             "A": theta_seeded[2:2 + args.aF_deg].tolist(),
             "b": float(theta_seeded[2 + args.aF_deg]),
             "P": theta_seeded[3 + args.aF_deg:5 + args.aF_deg].tolist(),
             "in_range": {"median": float(np.median([seeded[q] for q in IN_Q])),
                          "max": float(np.max([seeded[q] for q in IN_Q]))},
             "low_q": {f"{q:g}": seeded[q] for q in LOW_Q}},
         "reference": {k: {kk if isinstance(kk, str) else f"{kk:g}": vv
                           for kk, vv in v.items()} for k, v in REF.items()},
         "low_q_wins": {k: int(v) for k, v in wins.items()}}, indent=2))
    print(f"\nwrote {COEFFS}")
    for k, v in wins.items():
        print(f"GATE vs {k}: {v}/4 low-q better")


if __name__ == "__main__":
    main()
