"""
Physically-anchored reduction of gwr_energy_stiff: 9 coefficients -> 6.

Motivation and evidence are in `scaling_gwr_energy_stiff.md`, sections "Which
coefficient orders are physically motivated" and "Three negative results".  Summary:

  * `alpha_PP` and `beta_PP` both measure as the Newtonian chirp factor X1^(6/5)
    (beta_PP to 0.03%), which is derivable from matching the leading chirp rate and
    the mass-unit conversion.  Putting that base in explicitly turns 3 + 2 unstructured
    coefficients into 2 + 1 on a derived base, and their residual nu-expansions are
    well-ordered (2PA/1PA ~ 0.15 and smaller), so those orders mean something.
  * `alpha_E` keeps ONE coefficient.  Its 1/nu is a normalisation convention, not
    physics (E is the ppBHPT's radiated energy in the surrogate's own units, so only
    alpha_E*E is physical).  Higher orders are neither identifiable -- alpha_E is the
    soft direction of a -0.974-correlated degeneracy with alpha_PP -- nor meaningful,
    since the 2PA term of d ln alpha is 1.8-3.4x the 1PA term over the training range.
  * `P` is unchanged at 2 coefficients.  Its Bondi-mass account holds only at the
    bottom of the range and inverts by q=8, so it has no derivable order.

    alpha_PP(q) = X1^(6/5) * (1 + c0*nu + c1*nu^2)        2 free   [derived base]
    alpha_E(q)  = A(nu) / nu,  A = A0 (+ A1*nu + ...)     1 free   [convention]
    beta_PP(q)  = X1^(6/5) * (1 + b*nu)                   1 free   [derived base]
    P(q)        = P0 + P1*nu                              2 free   [empirical]

The per-q model is untouched (`--form mult`), so `per_q_cache_mult.json` is reused and
the per-q optima are identical to `gwr_energy_stiff`; only the regression layer changes.
The global joint refit reuses `fit_scaling_gwr_energy_global`'s fast evaluator exactly
as `fit_scaling_gwr_energy_stiff` does (Powell, analytic phi0, t0_nr a per-q nuisance,
training points sampled even in nu over [3,8], no q<3 data anywhere).

NOTE: this puts X1^(6/5) INTO the model, so it is no longer "no PN expressions
anywhere" and it no longer serves as independent confirmation of that anchor.  Keep the
shipped 9-coefficient `gwr_energy_stiff` frozen as the evidence run.

Usage:  python fit_scaling_gwr_energy_anchored.py --global --maxiter 80
        python fit_scaling_gwr_energy_anchored.py --global --alphaE-deg 2   # 8 coeffs
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

FORM = "mult"
RESULTS = ROOT / "gwr_energy_anchored_results"
RESULTS.mkdir(exist_ok=True)
COEFFS = RESULTS / "coeffs.json"
SEED_CACHE = G.RESULTS_DIR / "per_q_cache_mult.json"

LOW_Q = (2.75, 2.5, 2.25, 2.0)
IN_Q = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0)

# the shipped gwr_energy_stiff numbers we must beat below q=3
REF_LOW = {2.75: 9.8291e-04, 2.5: 1.1240e-03, 2.25: 1.7272e-03, 2.0: 3.7850e-03}
REF_IN = {"median": 6.2366e-04, "max": 9.7359e-04}

nu_of = lambda q: np.asarray(q, float) / (1.0 + np.asarray(q, float)) ** 2
X1_of = lambda q: np.asarray(q, float) / (1.0 + np.asarray(q, float))
base_of = lambda q: X1_of(q) ** 1.2          # the Newtonian chirp factor


# ---------------------------------------------------------------------------

def load_per_q():
    cache = json.loads(SEED_CACHE.read_text())
    rows = sorted(cache.values(), key=lambda r: r["q"])
    q = np.array([r["q"] for r in rows], float)
    P4 = np.array([r["params"] for r in rows], float)
    return {"q": q, "nu": nu_of(q), "alpha_PP": P4[:, 0], "alpha_E": P4[:, 1],
            "beta_PP": P4[:, 2], "P": P4[:, 2] * P4[:, 3], "t0_nr": P4[:, 4]}


def seed_theta(d, aE_deg):
    nu, b = d["nu"], base_of(d["q"])
    c = np.linalg.lstsq(np.vstack([nu, nu ** 2]).T, d["alpha_PP"] / b - 1.0,
                        rcond=None)[0]
    A = np.polyfit(nu, d["alpha_E"] * nu, aE_deg)[::-1]          # A0, A1, ...
    bb = np.linalg.lstsq(nu[:, None], d["beta_PP"] / b - 1.0, rcond=None)[0]
    P = np.polyfit(nu, d["P"], 1)[::-1]                          # P0, P1
    return np.concatenate([c, A, bb, P])


def split(theta, aE_deg):
    k = 0
    c = theta[k:k + 2]; k += 2
    A = theta[k:k + aE_deg + 1]; k += aE_deg + 1
    b = theta[k]; k += 1
    P = theta[k:k + 2]
    return c, A, b, P


def params_at(q, theta, aE_deg):
    c, A, b, P = split(theta, aE_deg)
    nu = float(nu_of(q)); bs = float(base_of(q))
    a_pp = float(np.clip(bs * (1.0 + c[0] * nu + c[1] * nu ** 2), 0.05, 2.5))
    b_pp = float(np.clip(bs * (1.0 + b * nu), 0.2, 1.6))
    a_e = float(sum(A[k] * nu ** k for k in range(len(A)))) / nu
    b_e = float(P[0] + P[1] * nu) / b_pp
    return np.array([a_pp, a_e, b_pp, b_e, 0.0, 0.0])


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global", dest="do_global", action="store_true")
    ap.add_argument("--alphaE-deg", type=int, default=0)
    ap.add_argument("--ntrain", type=int, default=12)
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--src", type=int, default=6)
    ap.add_argument("--nr", type=int, default=10)
    args = ap.parse_args()

    d = load_per_q()
    theta = seed_theta(d, args.alphaE_deg)
    ncoef = len(theta)
    print(f"anchored model: {ncoef} coefficients "
          f"(alpha_PP 2, alpha_E {args.alphaE_deg + 1}, beta_PP 1, P 2)", flush=True)

    t0_poly = np.polyfit(d["nu"], d["t0_nr"], 3)
    cases = {}
    def case_for(q):
        if q not in cases:
            cases[q] = G.load_case(q, args.src, args.nr)
        return cases[q]

    def run_all(th):
        out = {}
        for q in list(IN_Q) + list(LOW_Q):
            seed = float(np.polyval(t0_poly, float(nu_of(q))))
            p = params_at(q, th, args.alphaE_deg)
            e, t0 = GG.fast_mismatch(p[:4], case_for(q), seed, "fit")
            out[q] = GG.fast_mismatch(p[:4], case_for(q), t0, "full")[0]
        return out

    seeded = run_all(theta)

    if args.do_global:
        targets = GG.even_nu_targets(d["q"], args.ntrain)
        train_q = sorted({float(d["q"][np.argmin(np.abs(d["q"] - t))]) for t in targets})
        tcases = {q: case_for(q) for q in train_q}
        nuis = {q: float(np.polyval(t0_poly, float(nu_of(q)))) for q in train_q}

        def obj(th):
            errs = []
            for q in train_q:
                e, t0 = GG.fast_mismatch(params_at(q, th, args.alphaE_deg)[:4],
                                         tcases[q], nuis[q], "fit")
                nuis[q] = t0
                errs.append(e)
            errs = np.array(errs)
            obj.last = (float(errs.mean()), float(errs.max()))
            return errs.mean()
        obj.last = (np.nan, np.nan)

        obj(theta)
        print(f"[global] train q: {[round(q, 2) for q in train_q]}", flush=True)
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

    print(f"\n{'q':>6} {'seeded':>12} {'anchored':>12} {'stiff (9 coef)':>15} {'':>10}")
    for q in IN_Q:
        print(f"{q:>6g} {seeded[q]:>12.4e} {final[q]:>12.4e} {'':>15}")
    print(f"{'median':>6} {'':>12} {med:>12.4e} {REF_IN['median']:>15.4e}")
    print(f"{'max':>6} {'':>12} {mx:>12.4e} {REF_IN['max']:>15.4e}")
    print("  --- q < 3 (held out) ---")
    better = 0
    for q in LOW_Q:
        gain = REF_LOW[q] / final[q]
        flag = "BETTER" if final[q] < REF_LOW[q] else "worse"
        better += final[q] < REF_LOW[q]
        print(f"{q:>6g} {seeded[q]:>12.4e} {final[q]:>12.4e} {REF_LOW[q]:>15.4e}"
              f"  {flag} ({gain:.2f}x)")

    c, A, b, P = split(theta, args.alphaE_deg)
    out = {"model": "gwr_energy_anchored", "n_coefficients": int(ncoef),
           "alphaE_degree": args.alphaE_deg, "global_refit": bool(args.do_global),
           "form": {"alpha_PP": "X1^(6/5)*(1+c0*nu+c1*nu^2)",
                    "alpha_E": "A(nu)/nu", "beta_PP": "X1^(6/5)*(1+b*nu)",
                    "P": "P0+P1*nu"},
           "c": c.tolist(), "A": np.asarray(A).tolist(), "b": float(b),
           "P": P.tolist(),
           "in_range": {"median": med, "max": mx},
           "low_q": {f"{q:g}": final[q] for q in LOW_Q},
           "reference_stiff9": {"in_range": REF_IN,
                                "low_q": {f"{q:g}": REF_LOW[q] for q in LOW_Q}},
           "low_q_better_than_stiff9": int(better)}
    COEFFS.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {COEFFS}")
    print(f"GATE: {better}/4 low-q points better than the shipped 9-coefficient model")


if __name__ == "__main__":
    main()
