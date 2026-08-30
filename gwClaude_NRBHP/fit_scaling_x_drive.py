"""
x-driven per-q calibration: replace the energy drive E(t) with the orbital-frequency
coordinate x(t), for BOTH alpha and beta.

    s(t)     = x(t) - x(t_start)              x = (0.5 * omega_GW,pp)^(2/3)
    alpha(t) = a_PP * (1 + sum_{k=1..Ka} a_k * s^k)
    beta (t) = b_PP * (1 + sum_{k=1..Kb} b_k * s^k)
    tau(t)   = t0_nr + integral_{T_ANCHOR}^{t} beta dt'        phi0 analytic

This is alpha(x, q) and beta(x, q): polynomials in x with the q-dependence carried by
the coefficients.  `s` vanishes at the window start, so a_PP and b_PP ARE the low-x
limits and the X1^(6/5) anchor applies to them directly -- no shifted basis, no
constraint to impose.  Ka=Kb=1 is the exact structural analogue of the energy models
(4 parameters, one shape coefficient each) with x in place of E, so the comparison at
fixed parameter count is clean.

WHY x RATHER THAN E.  Both are monotone in t, so an E-driven model is already an
x-model with its shape fixed to E(x) -- the question is only whether that shape is the
right one.  Measured (NRBHP_alpha_beta_model_vs_x.py): Ehat(x) = E/E_tot is NOT
q-universal, spreading ~2x across q at mid-inspiral, whereas x itself has a nearly
q-universal merger value (0.327-0.330 over q in [2,8]).  So E hides q-dependence in its
shape that x exposes as a fixed coordinate; that is what this model tests.

DEGREES.  Chosen from a residual study against the parameter-free measured curves
(beta_drift_tests/alpha_beta_of_omega_v2.json, RMS of a degree-K fit):

    alpha   K=1 2.0e-02 (q=2) .. 4.0e-03 (q=8)     beta   K=1 6.5e-03 .. 4.5e-03
            K=2 8.0e-03      .. 3.4e-03                   K=2 4.9e-03 .. 3.1e-03
            K=3 3.2e-03      .. 3.2e-03                   K=3 3.0e-03 .. 2.8e-03

Everything flattens at ~2-3e-03, which is ~2% of the curves' span and matches the
measurement's own smoothing noise -- so Ka=2..3, Kb=1..2, and beyond that the fit is
chasing savgol.  Note the required degree is q-DEPENDENT: q=8 is nearly linear already,
q=2 needs K=3.  Raw powers of s are used, not Chebyshev: at K<=3 the design matrix
conditions at 1.0 / 3.1e1 / 8.2e2, so centring buys nothing (it only matters from K~5,
where the peaks model sits).

NO CLIPPING IS APPLIED.  x self-saturates: after merger omega_GW approaches the QNM
value, so x plateaus (measured max 0.327-0.330 at every q) and s stops driving on its
own.  That is the honest handling of "where x loses validity" -- the coordinate stops
moving rather than being cut off by hand.

Usage:
    python fit_scaling_x_drive.py --check --nproc 3        # q = 3,5,8
    python fit_scaling_x_drive.py --nproc 24               # full 64-q grid
    python fit_scaling_x_drive.py --ka 2 --kb 1 --nproc 24
    python fit_scaling_x_drive.py --t-cut -200 --nproc 24  # score the inspiral only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_gwr_energy_global as GG

RESULTS = ROOT / "x_drive_results"
RESULTS.mkdir(exist_ok=True)
SEED_CACHE = G.RESULTS_DIR / "per_q_cache_mult.json"

SRC_STRIDE, NR_STRIDE = 6, 10
WINDOW_M = 20.0          # savgol window for omega, in M (v2 shows 10/20/40 agree to 0.3%)


# ---------------------------------------------------------------------------

def add_x(case, window_M=WINDOW_M):
    """Inject s = x - x(t_start) into losses, mirroring FL.add_flux.

    x = (0.5 * omega_GW)^(2/3) from the unwrapped ppBHPT phase, with savgol's own
    analytic derivative and the window set in PHYSICAL M -- the ppBHPT cache is
    dt = 0.2 M while NR is 0.1 M, and a fixed SAMPLE window smooths the two sides over
    different durations (the flaw that invalidated alpha_beta_of_omega.py v1).
    """
    t = np.asarray(case["t_bhpt"], float)
    dt = float(np.median(np.diff(t)))
    n = int(round(window_M / dt))
    n = max(n + 1 - (n % 2), 7)                     # force odd, at least 7
    psi = np.unwrap(np.angle(case["h_bhpt"]))
    w = np.abs(savgol_filter(psi, n, 3, deriv=1, delta=dt))
    x = (0.5 * w) ** (2.0 / 3.0)
    s = x - float(x[0])
    case["losses"]["x_hat"] = s
    case["fit_losses"]["x_hat"] = np.interp(case["fit_t_bhpt"], t, s)
    case["_x_raw"] = x
    return case


def model_shape(p, t_bhpt, losses, ka, kb):
    """alpha and beta as polynomials in s, and the resulting time-map shape."""
    a_pp = p[0]
    a_k = p[1:1 + ka]
    b_pp = p[1 + ka]
    b_k = p[2 + ka:2 + ka + kb]
    s = losses["x_hat"]

    alpha = a_pp * (1.0 + sum(a_k[j] * s ** (j + 1) for j in range(ka)))
    beta = b_pp * (1.0 + sum(b_k[j] * s ** (j + 1) for j in range(kb)))
    if np.any(beta <= 0) or not np.all(np.isfinite(np.r_[alpha, beta])):
        return alpha, None, -1.0
    bc = G.creative.cumulative_trapezoid(beta, t_bhpt)
    tau_shape = bc - float(np.interp(G.T_ANCHOR, t_bhpt, bc))
    return alpha, tau_shape, float(np.min(np.diff(tau_shape)))


def mism(p, case, t0_ref, ka, kb, key="fit", span=120.0):
    if key == "fit":
        tb, hb, tn, hn, los = (case["fit_t_bhpt"], case["fit_h_bhpt"],
                               case["fit_t_nr"], case["fit_h_nr"], case["fit_losses"])
    else:
        tb, hb, tn, hn, los = (case["t_bhpt"], case["h_bhpt"],
                               case["t_nr"], case["h_nr"], case["losses"])
    a, tau, dmin = model_shape(p, tb, los, ka, kb)
    if dmin <= 0:
        return 50.0 + 1e3 * (-dmin), t0_ref
    r = minimize_scalar(lambda t0: GG.err_at_t0(t0, a, tau, hb, tn, hn),
                        bounds=(t0_ref - span, t0_ref + span), method="bounded",
                        options={"xatol": 1e-3, "maxiter": 80})
    return float(r.fun), float(r.x)


def truncate(case, t_cut):
    """NR-side arrays cut at t_cut; the BHPT side is untouched."""
    c = dict(case)
    for tk, hk in (("t_nr", "h_nr"), ("fit_t_nr", "fit_h_nr")):
        t = np.asarray(case[tk], float)
        m = t <= t_cut
        c[tk] = t[m]
        c[hk] = np.asarray(case[hk])[m]
    return c


def seed_from_energy(case, pm, ka, kb):
    """Least-squares the ENERGY model's alpha(t), beta(t) onto the s-basis.

    Both drives are monotone in t, so the E solution is a curve in s and projecting it
    gives a seed already in the right basin -- far better than starting the shape
    coefficients at zero, and it makes the comparison against the energy model a
    refinement rather than a fresh search.
    """
    los = case["losses"]
    e, s = np.asarray(los["e_oft"], float), np.asarray(los["x_hat"], float)
    alpha_e = pm[0] * (1.0 + pm[1] * e)
    beta_e = pm[2] * (1.0 + pm[3] * e)

    def project(y, K):
        y0 = float(y[0])
        V = np.vstack([s ** (j + 1) for j in range(K)]).T
        c, *_ = np.linalg.lstsq(V, y / y0 - 1.0, rcond=None)
        return y0, c

    a_pp, a_k = project(alpha_e, ka)
    b_pp, b_k = project(beta_e, kb)
    return np.r_[a_pp, a_k, b_pp, b_k]


def fit_one_q(q, ka, kb, pm, t0_seed, t_cut):
    case = add_x(FL.add_flux(G.load_case(q, SRC_STRIDE, NR_STRIDE)))
    scase = case if t_cut is None else truncate(case, t_cut)
    p0 = seed_from_energy(case, pm, ka, kb)

    # 3 starts, not 9: seed_from_energy projects the CONVERGED energy solution onto the
    # s-basis, so the seed is already in the right basin.  (perq_fit_energy needs 9
    # because it seeds from crude scalings of a cached solution.)  Tolerances are set
    # relative to the mismatch scale ~1e-4, not absurdly below it.
    cands = []
    for sa, sh in ((1.0, 1.0), (0.98, 1.15), (1.02, 0.85)):
        if True:
            p = p0.copy()
            p[0] *= sa
            p[1:1 + ka] *= sh
            st = {"t0": t0_seed}

            def obj(pp):
                e, t0 = mism(pp, scase, st["t0"], ka, kb, "fit")
                st["t0"] = t0
                return e
            e0 = obj(p)
            cands.append((e0, p.copy()))
            r = minimize(obj, p, method="Nelder-Mead",
                         options={"maxiter": 2500, "xatol": 1e-6, "fatol": 1e-10})
            cands.append((float(r.fun), np.asarray(r.x, float).copy()))
    best_seed = min(c[0] for c in cands[::2])
    e_fit, p_best = min(cands, key=lambda c: c[0])

    # rescore on the full-resolution arrays, and report both windows
    e_scored, t0 = mism(p_best, scase, t0_seed, ka, kb, "full")
    e_full = mism(p_best, case, t0, ka, kb, "full")[0]

    s = np.asarray(case["losses"]["x_hat"], float)
    x = np.asarray(case["_x_raw"], float)
    alpha, _, _ = model_shape(p_best, case["t_bhpt"], case["losses"], ka, kb)
    beta = p_best[1 + ka] * (1.0 + sum(p_best[2 + ka + j] * s ** (j + 1)
                                       for j in range(kb)))
    return {
        "q": q, "nu": G.get_nu(q), "ka": ka, "kb": kb, "t_cut": t_cut,
        "params": p_best.tolist(), "err_fitgrid": e_fit,
        "err_scored": e_scored, "err_full": e_full,
        "a_PP": float(p_best[0]), "b_PP": float(p_best[1 + ka]),
        "a_k": p_best[1:1 + ka].tolist(),
        "b_k": p_best[2 + ka:2 + ka + kb].tolist(),
        "a_PP_over_X1_6_5": float(p_best[0]) / (q / (1.0 + q)) ** 1.2,
        "b_PP_over_X1_6_5": float(p_best[1 + ka]) / (q / (1.0 + q)) ** 1.2,
        "x_start": float(x[0]), "x_max": float(np.nanmax(x)), "s_max": float(s[-1]),
        "alpha_change_pct": float((alpha[-1] / alpha[0] - 1.0) * 100.0),
        "beta_change_pct": float((beta[-1] / beta[0] - 1.0) * 100.0),
        # tolerant test: x comes from a savgol derivative and carries ~0.3% numerical
        # wiggle, so an exact all(diff<=0) is a test of the smoothing, not the model.
        # Report the offending fraction and flag on a 2% threshold.
        "alpha_frac_rising": float(np.mean(np.diff(alpha) > 0)),
        "beta_frac_falling": float(np.mean(np.diff(beta) < 0)),
        "alpha_monotone_falling": bool(np.mean(np.diff(alpha) > 0) < 0.02),
        "beta_monotone_rising": bool(np.mean(np.diff(beta) < 0) < 0.02),
        "seed_warning": bool(e_fit > best_seed + 1e-15),
    }


_CTX = {}


def _init(ka, kb, pm, t0s, t_cut):
    _CTX.update(ka=ka, kb=kb, pm=pm, t0s=t0s, t_cut=t_cut)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def _worker(q):
    k = round(q, 10)
    return fit_one_q(q, _CTX["ka"], _CTX["kb"], _CTX["pm"][k], _CTX["t0s"][k],
                     _CTX["t_cut"])


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ka", type=int, default=2, help="alpha shape terms in s")
    ap.add_argument("--kb", type=int, default=1, help="beta shape terms in s")
    ap.add_argument("--t-cut", type=float, default=None,
                    help="score only t_nr <= t_cut (default: full window)")
    ap.add_argument("--q", type=float, nargs="+", default=None)
    ap.add_argument("--check", action="store_true", help="q = 3, 5, 8 only")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seeds = json.loads(SEED_CACHE.read_text())
    grid = sorted(r["q"] for r in seeds.values())
    pm = {round(float(v["q"]), 10): v["params"][:4] for v in seeds.values()}
    t0s = {round(float(v["q"]), 10): v["params"][4] for v in seeds.values()}

    if args.check:
        q_list = [min(grid, key=lambda g: abs(g - t)) for t in (3.0, 5.0, 8.0)]
    elif args.q is not None:
        q_list = sorted({min(grid, key=lambda g: abs(g - t)) for t in args.q})
    else:
        q_list = grid

    tag = f"ka{args.ka}_kb{args.kb}" + ("" if args.t_cut is None
                                        else f"_tcut{abs(args.t_cut):.0f}")
    cache_path = RESULTS / f"per_q_cache_{tag}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    key_of = lambda q: f"{q:.10f}"
    todo = q_list if args.force else [q for q in q_list if key_of(q) not in cache]

    print(f"[x_drive] alpha: {args.ka} shape term(s), beta: {args.kb}  "
          f"({2 + args.ka + args.kb} params/q)")
    print(f"[x_drive] window: {'full' if args.t_cut is None else f't_nr <= {args.t_cut:g} M'}"
          f",  {len(q_list)} q ({len(q_list)-len(todo)} cached, {len(todo)} to fit), "
          f"nproc={args.nproc}", flush=True)

    t_start = time.time()
    if todo:
        def emit(n, row):
            cache[key_of(row["q"])] = row
            cache_path.write_text(json.dumps(cache, indent=1))
            print(f"  [{n:>2}/{len(todo)}] q={row['q']:<7.4f} "
                  f"E={row['err_scored']:.4e}  a_k={np.array2string(np.array(row['a_k']), precision=3)}  "
                  f"b_k={np.array2string(np.array(row['b_k']), precision=3)}  "
                  f"dalpha={row['alpha_change_pct']:+6.2f}% "
                  f"dbeta={row['beta_change_pct']:+6.2f}%", flush=True)

        if args.nproc > 1:
            with Pool(args.nproc, initializer=_init,
                      initargs=(args.ka, args.kb, pm, t0s, args.t_cut)) as pool:
                for n, row in enumerate(pool.imap_unordered(_worker, todo), 1):
                    emit(n, row)
        else:
            _init(args.ka, args.kb, pm, t0s, args.t_cut)
            for n, q in enumerate(todo, 1):
                emit(n, _worker(q))
    print(f"[x_drive] fitting took {time.time()-t_start:.0f}s", flush=True)

    rows = [cache[key_of(q)] for q in q_list]
    err = np.array([r["err_scored"] for r in rows])
    print(f"\n{'q':>8} {'E scored':>11} {'E full':>11} {'a_PP/X1':>8} {'b_PP/X1':>8} "
          f"{'d alpha':>9} {'d beta':>9} {'a mono':>7} {'b mono':>7}")
    for r in rows:
        print(f"{r['q']:>8.4f} {r['err_scored']:>11.4e} {r['err_full']:>11.4e} "
              f"{r['a_PP_over_X1_6_5']:>8.4f} {r['b_PP_over_X1_6_5']:>8.4f} "
              f"{r['alpha_change_pct']:>+8.2f}% {r['beta_change_pct']:>+8.2f}% "
              f"{'yes' if r['alpha_monotone_falling'] else 'NO':>7} "
              f"{'yes' if r['beta_monotone_rising'] else 'NO':>7}")

    n_af = sum(r["alpha_monotone_falling"] for r in rows)
    n_br = sum(r["beta_monotone_rising"] for r in rows)
    print(f"\n=== x_drive  ka={args.ka} kb={args.kb}  {len(rows)} q ===")
    print(f"scored-window mismatch: median {np.median(err):.4e}  "
          f"min {err.min():.4e}  max {err.max():.4e}")
    print(f"alpha falls monotonically at {n_af}/{len(rows)} q")
    print(f"beta rises monotonically at {n_br}/{len(rows)} q")
    ra = np.array([r["a_PP_over_X1_6_5"] for r in rows])
    rb = np.array([r["b_PP_over_X1_6_5"] for r in rows])
    print(f"a_PP/X1^(6/5): {ra.min():.4f} .. {ra.max():.4f}   "
          f"b_PP/X1^(6/5): {rb.min():.4f} .. {rb.max():.4f}")
    nw = sum(r["seed_warning"] for r in rows)
    if nw:
        print(f"!! {nw} q where the optimiser never beat its seed -- inspect")

    summary = {
        "model": "x_drive", "ka": args.ka, "kb": args.kb, "t_cut": args.t_cut,
        "n_q": len(rows), "n_params_per_q": 2 + args.ka + args.kb,
        "err_scored": {"median": float(np.median(err)), "min": float(err.min()),
                       "max": float(err.max())},
        "n_alpha_monotone_falling": int(n_af),
        "n_beta_monotone_rising": int(n_br),
        "a_PP_over_X1_6_5": {"min": float(ra.min()), "max": float(ra.max())},
        "b_PP_over_X1_6_5": {"min": float(rb.min()), "max": float(rb.max())},
        "n_seed_warnings": int(nw),
    }
    (RESULTS / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {cache_path}\nwrote {RESULTS / f'summary_{tag}.json'}")


if __name__ == "__main__":
    main()
