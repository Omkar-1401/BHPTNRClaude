"""TEST, NOT A MODEL: refit pn_anchored + spin's 13 coefficients JOINTLY on the 2d grid.

This does not replace anything.  The shipped spin model stays
`fit_pn_anchored_spin.py` + `pn_anchored_spin_results/coeffs_spin.json` (9 frozen from a
1d fit + 4 fitted, median 7.686e-04).  Outputs here go to `test_joint2d_results/`.

WHAT IT IS FOR.  The three-way spin comparison is not matched on source:

    model                    fitted on 2d   transplanted from 1d   median
    pn_anchored + spin             4               9               7.686e-04
    fluxanchored + spin            9               0               1.411e-03
    energy_anchored + spin         8               0               1.731e-03

pn_anchored's 9 were fitted against BHPTNRSur1dq1e4 and carried onto BHPTNRSur2dq1e3.
Those two surrogates differ from EACH OTHER at chi1 = 0 by a mismatch of 8.7e-05 to
1.2e-04 -- ~25% of the residual being fitted -- and the transplant landed FAVOURABLY for
pn_anchored (6.148e-04 -> 5.543e-04 at chi1 = 0) and UNFAVOURABLY for fluxanchored
(4.694e-04 -> 7.453e-04).  So some unknown part of the 1.84x gap could be that coin
toss rather than anything structural.

This run removes the confound: all 13 fitted on the same 96-point grid, chi1 = 0
included, exactly the treatment the other two models got.

  * if pn_anchored's advantage SURVIVES, the gap is real and worth explaining -- and the
    next test is whether `beta_r`'s spin dependence is what buys it (freeze beta_r at
    chi1 = 0 and refit; that is a mechanism test, and it does NOT make the models
    equivalent, since the switch and both merger branches remain);
  * if it SHRINKS, we have been comparing transplant luck.

Judge it on the chi1 < 0 columns, not the median: the gap is 3-4x at chi1 = -0.4/-0.6
and only 1.1x at chi1 = 0, so it is a retrograde-spin deficit, not a general one.

theta = [b1, b2, a2, p00, p01, w0, r0, m0, m1, s11, s12, s21, s22]   -- all 13 free.
Model, drive, objective and grid are otherwise IDENTICAL to fit_pn_anchored_spin.py;
see `scaling_pn_anchored_spin.md` for the definitions and sources.

SPEED NOTE.  `fit_pn_anchored_spin.mismatch` runs an 81-point coarse t0 scan on EVERY
call, which costs ~28 s per 96-point objective evaluation -- fine for 4 parameters,
hopeless for 13.  Here t0 is carried forward per case (coarse scan once, then a bounded
refinement), as `fit_fluxanchored_spin.py` already does.  Same optimum, ~7x cheaper.

Usage:
    python test_pn_spin_joint2d.py --refit --budget 100
    python test_pn_spin_joint2d.py --refit --resume --budget 100
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, minimize_scalar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")

import fit_pn_anchored_spin as S

RESULTS = HERE / "test_joint2d_results"; RESULTS.mkdir(exist_ok=True)
COEFFS = RESULTS / "coeffs_joint2d.json"
PARTIAL = RESULTS / "partial_joint2d.json"
NAMES = ["b1", "b2", "a2", "p00", "p01", "w0", "r0", "m0", "m1",
         "s11", "s12", "s21", "s22"]
REF = {"pn_anchored+spin, 9 frozen from 1d": 7.6858e-04,
       "fluxanchored+spin, 9 fitted on 2d": 1.4110e-03,
       "energy_anchored+spin, 8 fitted on 2d": 1.7311e-03}


def seed_theta():
    """The shipped 9 (1d-fitted) + the shipped 4 spin (2d-fitted).  A starting point."""
    th9 = S.shipped9()
    ths = np.array(json.loads(
        (HERE / "pn_anchored_spin_results" / "coeffs_spin.json").read_text()
    )["theta_spin"], float)
    return np.r_[th9, ths]


def mismatch(th, q, chi1, case, t0_seed, key="full"):
    th9, ths = th[:9], th[9:]
    p = "fit_" if key == "fit" else ""
    t, h = case[p + "t_bhpt"], case[p + "h_bhpt"]
    tn, hn = case[p + "t_nr"], case[p + "h_nr"]
    x, pl = case[p + "x"], case[p + "p_loss"]
    alpha, beta = S.model_ab(th9, ths, q, chi1, x, pl)
    if np.any(beta <= 0) or not np.all(np.isfinite(np.r_[alpha, beta])):
        return 50.0, t0_seed
    bc = S.creative.cumulative_trapezoid(beta, t)
    ts = bc - float(np.interp(S.T_ANCHOR, t, bc))
    d = float(np.min(np.diff(ts)))
    if d <= 0:
        return 50.0 + 1e3 * (-d), t0_seed
    f = lambda x0: S._err_at_t0(x0, alpha, ts, t, h, tn, hn)
    if t0_seed is None:                       # once per case, not per evaluation
        g = np.linspace(-250.0, 150.0, 81)
        t0_seed = float(g[int(np.argmin([f(v) for v in g]))])
    r = minimize_scalar(f, bounds=(t0_seed - 60.0, t0_seed + 60.0), method="bounded",
                        options={"xatol": 1e-3, "maxiter": 60})
    return float(r.fun), float(r.x)


def main():
    import fit_pn_anchored_spin as _S0; _S0.refuse_frozen9_protocol(__file__)   # 2026-09-27
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--maxiter", type=int, default=600)
    args = ap.parse_args()

    print("TEST: pn_anchored + spin, all 13 coefficients refitted JOINTLY on the 2d grid")
    print("      (does NOT replace the shipped model; outputs -> test_joint2d_results/)")
    print(f"grid: {len(S.TRAIN_Q)} q x {len(S.CHIS)} chi1 = "
          f"{len(S.TRAIN_Q)*len(S.CHIS)} points, chi1=0 INCLUDED\n", flush=True)

    t0 = time.time(); cases = {}
    for q in S.TRAIN_Q:
        for c in S.CHIS:
            cases[(q, c)] = S.build_case(q, c)
        print(f"  cached q={q:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"cache ready in {time.time()-t0:.0f}s", flush=True)

    th = (np.array(json.loads(COEFFS.read_text())["theta"], float)
          if args.resume and COEFFS.exists() else seed_theta())
    print("seed: " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(NAMES, th)), flush=True)

    nuis = {k: None for k in cases}
    st = {"n": 0, "best": np.inf, "bx": th.copy(), "t0": time.time()}

    class _Budget(Exception):
        pass

    def score(x):
        out = []
        for k, case in cases.items():
            e, t0_ = mismatch(x, k[0], k[1], case, nuis[k])
            nuis[k] = t0_
            out.append(e)
        return np.array(out)

    def obj(x):
        errs = score(x)
        m, md = float(np.mean(errs)), float(np.median(errs))
        st["n"] += 1; el = time.time() - st["t0"]; tag = ""
        if m < st["best"]:
            st["best"] = m; st["bx"] = np.array(x, float); tag = "  <-- best"
            PARTIAL.write_text(json.dumps({"n_eval": st["n"], "elapsed_min": el/60,
                                           "names": NAMES, "theta": st["bx"].tolist(),
                                           "mean": m, "median": md}, indent=2))
        print(f"  [{st['n']:4d}] {el/60:6.1f}m mean={m:.4e} median={md:.4e} "
              f"max={errs.max():.4e}{tag}", flush=True)
        if el > args.budget * 60:
            raise _Budget
        return m

    obj(th)
    st["t0"] = time.time(); st["n"] = 0
    try:
        r = minimize(obj, th, method="Powell",
                     options={"maxiter": args.maxiter, "xtol": 1e-4, "ftol": 1e-6})
        th = st["bx"]
        print(f"  CONVERGED / {st['n']} evals: {r.message}", flush=True)
    except _Budget:
        th = st["bx"]
        print(f"  STOPPED ON BUDGET / {st['n']} evals (best mean {st['best']:.4e}); "
              f"rerun with --resume", flush=True)

    print("\nfitted: " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(NAMES, th)))
    errs = score(th)
    allm = {f"{q:.4f}_{c:+.1f}": float(e) for (q, c), e in zip(cases.keys(), errs)}
    print(f"\n{'q':>8} " + " ".join(f"{c:>+9.1f}" for c in S.CHIS))
    for q in S.TRAIN_Q:
        print(f"{q:>8.4f} " + " ".join(
            f"{allm[f'{q:.4f}_{c:+.1f}']:9.2e}" for c in S.CHIS))
    print(f"\nALL 96: median {np.median(errs):.4e}  max {errs.max():.4e}")
    print("\nper chi1, against the SHIPPED (transplanted) pn_anchored + spin:")
    ship = json.loads((HERE / "pn_anchored_spin_results" / "coeffs_spin.json").read_text())
    for c in S.CHIS:
        new = np.median([allm[f"{q:.4f}_{c:+.1f}"] for q in S.TRAIN_Q])
        old = np.median([ship["mismatches"][f"{q:.4f}_{c:+.1f}"] for q in S.TRAIN_Q])
        print(f"  chi1={c:+.1f}: joint-2d {new:.4e}   shipped {old:.4e}   "
              f"{new/old:5.2f}x")
    print("\nreference: " + ",  ".join(f"{k} {v:.4e}" for k, v in REF.items()))
    COEFFS.write_text(json.dumps({
        "purpose": "TEST of the source confound; NOT a replacement model",
        "model": "pn_anchored + spin, all 13 coefficients fitted on the 2d grid",
        "n_coefficients": 13, "names": NAMES, "theta": [float(x) for x in th],
        "grid": {"q": S.TRAIN_Q, "chi1": S.CHIS, "chi0_included": True},
        "median": float(np.median(errs)), "max": float(errs.max()),
        "mismatches": allm, "reference": REF}, indent=2))
    print(f"\nwrote {COEFFS}")


if __name__ == "__main__":
    main()
