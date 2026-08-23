"""
Flux-coupled alpha, with the SAME uniform degree-3 regression layer as
`gwr_energy_global` (14 coefficients).  A controlled test of the COORDINATE change
alone: everything else -- degrees, anchors, joint-fit machinery, training set -- is held
fixed, so any difference is attributable to alpha's coordinate.

    E(t)  = gw_remnant Eoft                       (cumulative radiated energy)
    F(t)  = Edot / max(Edot)                      (peak-normalised instantaneous flux)

    alpha(t, nu) = alpha_PP(nu) * (1 + alpha_F(nu) * F(t))     <-- CHANGED, was E
    beta (t, nu) = beta_PP (nu) * (1 + beta_E (nu) * E(t))     <-- unchanged

Motivation.  Regressing the *empirical* alpha(t) (the amplitude ratio |h_NR(tau)| /
|h_BHPT(t)| at the per-q time map) on each coordinate gives RMS residuals, as % of alpha:

    q      on E (cumulative)     on -Edot (flux)     on both
    3            2.279%              1.372%           1.369%
    5            2.011%              1.278%           1.265%
    8            1.448%              0.934%           0.933%

The flux is ~1.6x better at every mass ratio, and E adds essentially nothing on top of
it (0.2%), so the flux is the better single coordinate rather than a complement.  The
reason is that alpha's departure from its PP value is localised near merger, whereas E
is the integral of the flux and therefore carries inspiral memory and starts growing far
too early.  Measured on the same window: alpha leaves its plateau at t = -31 / -20 / -7 M
for q = 3 / 5 / 8, against |Edot| peaks at -7 / -9 / +7 M.

Prior art and the known risk.  `wf_nu_fluxes` already coupled alpha to instantaneous
fluxes and produced the best per-q median in this workspace (6.86e-5, beating the gated
8.76e-5) but FAILED as a master (median 6.31e-4, q=2 = 0.11) because it used TWO flux
terms that are degenerate (F_e ~ omega * F_j, both peaking at merger).  This script uses
a SINGLE flux term, which is the follow-up CLAUDE.md records as never having been run.

KNOWN RISK: unlike E, F is NOT monotonic -- it peaks at merger and falls through
ringdown -- so alpha would return toward alpha_PP after merger, whereas the empirical
alpha keeps dropping.  The evidence above was measured on t in [-800, 20] M; the fit
window extends further.  If flux-only underperforms, that asymmetry is the first
suspect, and the fix is a two-term alpha (E for the ringdown floor, F for the merger
ramp) rather than abandoning the coordinate.

Usage:  python fit_scaling_gwr_energy_flux.py --global --maxiter 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_global as GG

RESULTS = ROOT / "gwr_energy_flux_results"
RESULTS.mkdir(exist_ok=True)
COEFFS = RESULTS / "coeffs.json"
SEED_CACHE = G.RESULTS_DIR / "per_q_cache_mult.json"

NAMES = ["alpha_PP", "alpha_F", "beta_PP", "beta_E"]
ANCH = [1.0, None, 1.0, None]
LOW_Q = (2.75, 2.5, 2.25, 2.0)
IN_Q = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0)

# gwr_energy_global, degree 3, lam 0 -- the model this is a controlled variant of
REF_IN = {"median": 6.26e-04, "max": 9.70e-04}
REF_LOW = {2.75: 9.9e-04, 2.5: 1.40e-03, 2.25: 4.9e-03, 2.0: 2.8e-02}


# ---------------------------------------------------------------------------

def add_flux(case):
    """Peak-normalised instantaneous flux, computed once at full resolution."""
    t = case["t_bhpt"]
    e = case["losses"]["e_oft"]
    ed = np.gradient(e, t)
    f = ed / float(np.max(ed))
    case["losses"]["flux_hat"] = f
    case["fit_losses"]["flux_hat"] = np.interp(case["fit_t_bhpt"], t, f)
    return case


def model_shape(p, t_bhpt, losses):
    a_pp, a_f, b_pp, b_e = p
    alpha = a_pp * (1.0 + a_f * losses["flux_hat"])
    beta = b_pp * (1.0 + b_e * losses["e_oft"])
    if np.any(beta <= 0):
        return alpha, None, -1.0
    bc = G.creative.cumulative_trapezoid(beta, t_bhpt)
    tau_shape = bc - float(np.interp(G.T_ANCHOR, t_bhpt, bc))
    return alpha, tau_shape, float(np.min(np.diff(tau_shape)))


def fast_mismatch(p, case, t0_seed, key="fit"):
    if key == "fit":
        tb, hb, tn, hn, los = (case["fit_t_bhpt"], case["fit_h_bhpt"],
                               case["fit_t_nr"], case["fit_h_nr"], case["fit_losses"])
    else:
        tb, hb, tn, hn, los = (case["t_bhpt"], case["h_bhpt"],
                               case["t_nr"], case["h_nr"], case["losses"])
    alpha, tau_shape, dmin = model_shape(p, tb, los)
    if dmin <= 0:
        return 50.0 + 1e3 * (-dmin), t0_seed
    f = lambda t0: GG.err_at_t0(t0, alpha, tau_shape, hb, tn, hn)
    r = minimize_scalar(f, bounds=(t0_seed - 60.0, t0_seed + 60.0),
                        method="bounded", options={"xatol": 1e-3, "maxiter": 50})
    return float(r.fun), float(r.x)


def perq_fit(q, case, t0_seed, be_seed):
    """4-parameter per-q fit with the flux alpha.  t0 is the inner 1-D nuisance."""
    x1 = q / (1.0 + q)
    best = None
    for a_f in (-0.15, -0.25, -0.35, -0.5):
        for s in (1.0, 0.97, 1.03):
            p0 = np.array([x1 ** 1.2 * s, a_f, x1 ** 1.2, be_seed])
            st = {"t0": t0_seed}

            def obj(p):
                e, t0 = fast_mismatch(p, case, st["t0"], "fit")
                st["t0"] = t0
                return e
            r = minimize(obj, p0, method="Nelder-Mead",
                         options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-10})
            if best is None or r.fun < best[0]:
                best = (float(r.fun), r.x.copy(), st["t0"])
    return best


def layout(degree):
    return [(i, degree if ANCH[i] is not None else degree + 1) for i in range(len(NAMES))]


def unpack(theta, degree):
    coeffs, k = {}, 0
    for i, nc in layout(degree):
        blk = np.asarray(theta[k:k + nc], float); k += nc
        coeffs[NAMES[i]] = (np.concatenate([[ANCH[i]], blk]) if ANCH[i] is not None
                            else blk)
    return coeffs


def pack(coeffs, degree):
    out = []
    for i, nc in layout(degree):
        c = np.asarray(coeffs[NAMES[i]], float)
        out.extend(c[1:1 + nc] if ANCH[i] is not None else c[:nc])
    return np.array(out, float)


def params_at(q, coeffs):
    p = np.array([G.eval_poly(q, coeffs[n]) for n in NAMES], float)
    p[0] = np.clip(p[0], 0.05, 2.5)
    p[2] = np.clip(p[2], 0.2, 1.6)
    return p


def fit_one(x, y, degree, anchor):
    if anchor is None:
        X = np.vstack([x ** k for k in range(degree + 1)]).T
        return np.linalg.lstsq(X, y, rcond=None)[0]
    X = np.vstack([x ** k for k in range(1, degree + 1)]).T
    c = np.linalg.lstsq(X, y - anchor, rcond=None)[0]
    return np.concatenate([[anchor], c])


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global", dest="do_global", action="store_true")
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--ntrain", type=int, default=12)
    ap.add_argument("--nperq", type=int, default=16)
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--src", type=int, default=6)
    ap.add_argument("--nr", type=int, default=10)
    args = ap.parse_args()

    cache = json.loads(SEED_CACHE.read_text())
    rows = sorted(cache.values(), key=lambda r: r["q"])
    q_all = np.array([r["q"] for r in rows], float)
    pa = np.array([r["params"] for r in rows], float)
    t0_poly = np.polyfit(G.get_nu(q_all), pa[:, 4], 3)
    be_poly = np.polyfit(G.get_nu(q_all), pa[:, 3], 3)

    cases = {}
    def case_for(q):
        if q not in cases:
            cases[q] = add_flux(G.load_case(q, args.src, args.nr))
        return cases[q]

    # ---- per-q fits with the flux alpha, to measure the nu-structure and seed ----
    tg = GG.even_nu_targets(q_all, args.nperq)
    perq_q = sorted({float(q_all[np.argmin(np.abs(q_all - t))]) for t in tg})
    print(f"[flux] per-q fits at {len(perq_q)} mass ratios ...", flush=True)
    P = {}
    for q in perq_q:
        nu = G.get_nu(q)
        e, p, t0 = perq_fit(q, case_for(q), float(np.polyval(t0_poly, nu)),
                            float(np.polyval(be_poly, nu)))
        P[q] = (p, e)
        print(f"   q={q:<6.3g} err={e:.4e}  a_PP={p[0]:.4f} a_F={p[1]:+.4f} "
              f"b_PP={p[2]:.4f} b_E={p[3]:+.4f}", flush=True)
    pq_err = np.array([P[q][1] for q in perq_q])
    print(f"[flux] per-q floor: median={np.median(pq_err):.4e} max={pq_err.max():.4e}",
          flush=True)

    nus = np.array([G.get_nu(q) for q in perq_q])
    vals = np.array([P[q][0] for q in perq_q])
    coeffs = {n: fit_one(nus, vals[:, i], args.degree, ANCH[i])
              for i, n in enumerate(NAMES)}

    def run_all(cf):
        out = {}
        for q in list(IN_Q) + list(LOW_Q):
            seed = float(np.polyval(t0_poly, G.get_nu(q)))
            p = params_at(q, cf)
            e, t0 = fast_mismatch(p, case_for(q), seed, "fit")
            out[q] = fast_mismatch(p, case_for(q), t0, "full")[0]
        return out

    seeded = run_all(coeffs)

    if args.do_global:
        tg = GG.even_nu_targets(q_all, args.ntrain)
        train_q = sorted({float(q_all[np.argmin(np.abs(q_all - t))]) for t in tg})
        tcases = {q: case_for(q) for q in train_q}
        nuis = {q: float(np.polyval(t0_poly, G.get_nu(q))) for q in train_q}

        def obj(theta):
            cf = unpack(theta, args.degree)
            errs = []
            for q in train_q:
                e, t0 = fast_mismatch(params_at(q, cf), tcases[q], nuis[q], "fit")
                nuis[q] = t0
                errs.append(e)
            errs = np.array(errs)
            obj.last = (float(errs.mean()), float(errs.max()))
            return errs.mean()
        obj.last = (np.nan, np.nan)

        theta0 = pack(coeffs, args.degree)
        obj(theta0)
        print(f"\n[global] train q: {[round(q, 2) for q in train_q]}", flush=True)
        print(f"[global] seed: mean={obj.last[0]:.4e} max={obj.last[1]:.4e}", flush=True)
        t_s = time.time()
        res = minimize(obj, theta0, method="Powell",
                       options={"maxiter": args.maxiter, "xtol": 1e-4, "ftol": 1e-6})
        print(f"[global] opt {time.time() - t_s:.0f}s: mean={obj.last[0]:.4e} "
              f"max={obj.last[1]:.4e}", flush=True)
        coeffs = unpack(res.x, args.degree)

    final = run_all(coeffs)
    med = float(np.median([final[q] for q in IN_Q]))
    mx = float(np.max([final[q] for q in IN_Q]))

    print(f"\n{'q':>6} {'seeded':>12} {'flux (14c)':>12} {'global E (14c)':>15}")
    for q in IN_Q:
        print(f"{q:>6g} {seeded[q]:>12.4e} {final[q]:>12.4e}")
    print(f"{'median':>6} {'':>12} {med:>12.4e} {REF_IN['median']:>15.4e}")
    print(f"{'max':>6} {'':>12} {mx:>12.4e} {REF_IN['max']:>15.4e}")
    print("  --- q < 3 (held out) ---")
    better = 0
    for q in LOW_Q:
        flag = "BETTER" if final[q] < REF_LOW[q] else "worse"
        better += final[q] < REF_LOW[q]
        print(f"{q:>6g} {seeded[q]:>12.4e} {final[q]:>12.4e} {REF_LOW[q]:>15.4e}  {flag}")

    COEFFS.write_text(json.dumps(
        {"model": "gwr_energy_flux", "degree": args.degree,
         "n_coefficients": int(sum(nc for _, nc in layout(args.degree))),
         "alpha_coordinate": "peak-normalised Edot", "beta_coordinate": "E",
         "coeffs": {n: coeffs[n].tolist() for n in NAMES},
         "per_q_floor": {"median": float(np.median(pq_err)), "max": float(pq_err.max())},
         "in_range": {"median": med, "max": mx},
         "low_q": {f"{q:g}": final[q] for q in LOW_Q},
         "reference_gwr_energy_global": {"in_range": REF_IN,
                                         "low_q": {f"{q:g}": v for q, v in REF_LOW.items()}},
         "low_q_better": int(better)}, indent=2))
    print(f"\nwrote {COEFFS}")
    print(f"GATE: {better}/4 low-q better than gwr_energy_global (E-coupled, 14 coef)")


if __name__ == "__main__":
    main()
