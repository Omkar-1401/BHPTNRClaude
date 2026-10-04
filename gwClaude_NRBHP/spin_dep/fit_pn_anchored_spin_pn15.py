"""pn_anchored + spin at the CORRECT PN ORDERS: 2 spin coefficients, not 4.

    beta_insp/base = 1 + nu*( b1*xc + b2*xc**2                     # nonspinning
                              + sSO*chi1   *xc**1.5                # 1.5PN spin-orbit
                              + sSS*chi1**2*xc**2 )                # 2PN   spin-spin
    beta_mr        = beta_r(q, chi1) * r0
    alpha          = unchanged, and carries NO spin

    theta = [b1, b2, a2, p00, p01, w0, r0, m0, m1, sSO, sSS]   -> 11 coefficients
    ALL 11 fitted jointly on the 2d grid, chi1 = 0 included.

WHY.  x = (M*omega/2)^(2/3) ~ v^2, so x^n corresponds to nPN.  The shipped spin form is
`nu*[(s11*chi1 + s12*chi1^2)*x + (s21*chi1 + s22*chi1^2)*x^2]`, which spends its four
coefficients as:

    s11*chi1  *x     chi1   at 1PN   -- WRONG: spin-orbit enters the flux at 1.5PN
    s12*chi1^2*x     chi1^2 at 1PN   -- WRONG: spin-spin enters at 2PN
    s21*chi1  *x^2   chi1   at 2PN   -- legitimate, but SUBLEADING spin-orbit
    s22*chi1^2*x^2   chi1^2 at 2PN   -- correct (spin-spin)

**There is no x^1.5 term anywhere in the model**, so the LEADING spin-orbit effect has
nowhere to live.  Measured consequence in the shipped fit: s21 = +13.32 dominates
s11 = -0.086 by 4-27x, and the ratio GROWS with x (3.9x at x=0.06, 16.8x at x=0.26).
So the fit is approximating x^1.5 with x^2, which drags the whole spin effect to the
merger plateau where the x^2 basis has 5-6x more leverage.  That is a candidate cause of
the q-U-shape that survived every other test (drive quarantine, chi1 domain split,
chi_eff, eps = 1/q -- all rejected; see scaling_pn_anchored_spin.md).

This form puts each spin term at a derivable PN order instead of fitting an arbitrary
polynomial, which is the same discipline the rest of the workspace runs on (X1^(6/5),
the 55/42 alpha slope, beta_r).  It also PARAMETER-MATCHES the spin sector to
fluxanchored + spin and energy_anchored + spin, which have 2 spin coefficients each --
the shipped 4 made that comparison unfair.

x^1.5 is safe: x = (M*omega/2)^(2/3) > 0 everywhere by construction.

WHAT IS NOT CHANGED: the drive (x from a savgol-smoothed phase, p_loss cumulative), the
switch, beta_r, alpha (still spin-blind), the grid, the objective.  See
`scaling_pn_anchored_spin.md` for definitions and sources.

NOT A REPLACEMENT until it wins.  Outputs -> pn15_results/.

Comparison targets, all on the same 96-point grid:
    pn_anchored + spin,  13 coef, 9 frozen from 1d      7.686e-04
    pn_anchored + spin,  13 coef, all fitted on 2d      (job 3512450, running)
    fluxanchored + spin,  9 coef, all fitted on 2d      1.411e-03
    energy_anchored+spin, 8 coef, all fitted on 2d      1.731e-03

Usage:
    python fit_pn_anchored_spin_pn15.py --refit --budget 100
    python fit_pn_anchored_spin_pn15.py --refit --resume --budget 100
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

RESULTS = HERE / "pn15_results"; RESULTS.mkdir(exist_ok=True)
COEFFS = RESULTS / "coeffs_pn15.json"
PARTIAL = RESULTS / "partial_pn15.json"
NAMES = ["b1", "b2", "a2", "p00", "p01", "w0", "r0", "m0", "m1", "sSO", "sSS"]
REF = {"pn_anchored+spin 13c, 9 frozen from 1d": 7.6858e-04,
       "fluxanchored+spin 9c on 2d": 1.4110e-03,
       "energy_anchored+spin 8c on 2d": 1.7311e-03}


def model_ab(th, q, chi1, x, p_loss):
    """Identical to fit_pn_anchored_spin.model_ab except for the two spin terms."""
    b1, b2, a2, p00, p01, w0, r0, m0, m1, sSO, sSS = th
    X1 = q / (1 + q); nu = q / (1 + q) ** 2; base = X1 ** 1.2
    xc = np.minimum(x, S.XCLIP)
    Sw = S.creative.sigmoid((p_loss - (p00 + p01 * nu)) / max(w0, 1e-3))
    beta = ((1 - Sw) * base * (1 + nu * (b1 * xc + b2 * xc ** 2
                                         + sSO * chi1 * xc ** 1.5
                                         + sSS * chi1 ** 2 * xc ** 2))
            + Sw * S.beta_r_phys(q, chi1) * r0)
    alpha = ((1 - Sw) * base * (1 + S.PN_SLOPE * nu * xc + nu * a2 * xc ** 2)
             + Sw * base * (m0 + m1 * nu))
    return alpha, beta


def seed_theta():
    """Shipped 9, plus sSO/sSS projected to reproduce the shipped 4-coefficient spin
    correction at mid-inspiral x = 0.12, so the fit starts from an equivalent effect:
        s11*x + s21*x^2  ->  sSO*x^1.5        (the chi1-linear part)
        s12*x + s22*x^2  ->  sSS*x^2          (the chi1^2 part)
    """
    th9 = S.shipped9()
    s11, s12, s21, s22 = json.loads(
        (HERE / "pn_anchored_spin_results" / "coeffs_spin.json").read_text())["theta_spin"]
    x = 0.12
    sSO = (s11 * x + s21 * x ** 2) / x ** 1.5
    sSS = (s12 * x + s22 * x ** 2) / x ** 2
    print(f"  seed projection at x={x}: sSO={sSO:+.4f}  sSS={sSS:+.4f}")
    return np.r_[th9, sSO, sSS]


def mismatch(th, q, chi1, case, t0_seed):
    t, h = case["t_bhpt"], case["h_bhpt"]
    tn, hn = case["t_nr"], case["h_nr"]
    alpha, beta = model_ab(th, q, chi1, case["x"], case["p_loss"])
    if np.any(beta <= 0) or not np.all(np.isfinite(np.r_[alpha, beta])):
        return 50.0, t0_seed
    bc = S.creative.cumulative_trapezoid(beta, t)
    ts = bc - float(np.interp(S.T_ANCHOR, t, bc))
    d = float(np.min(np.diff(ts)))
    if d <= 0:
        return 50.0 + 1e3 * (-d), t0_seed
    f = lambda v: S._err_at_t0(v, alpha, ts, t, h, tn, hn)
    if t0_seed is None:
        g = np.linspace(-250.0, 150.0, 81)
        t0_seed = float(g[int(np.argmin([f(v) for v in g]))])
    r = minimize_scalar(f, bounds=(t0_seed - 60.0, t0_seed + 60.0), method="bounded",
                        options={"xatol": 1e-3, "maxiter": 60})
    return float(r.fun), float(r.x)


def main():
    import fit_pn_anchored_spin as _S0; _S0.refuse_frozen9_protocol(__file__)   # 2026-09-27
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="no longer required: a run AUTO-RESUMES whenever "
                         "coeffs_pn15.json exists. Kept for compatibility.")
    ap.add_argument("--fresh", action="store_true",
                    help="override the auto-resume and restart from the "
                         "(poor) x=0.12 seed projection")
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--maxiter", type=int, default=600)
    args = ap.parse_args()

    print("pn_anchored + spin at CORRECT PN ORDERS: 1.5PN spin-orbit + 2PN spin-spin")
    print("11 coefficients, ALL fitted on the 2d grid, chi1 = 0 INCLUDED")
    print(f"grid: {len(S.TRAIN_Q)} q x {len(S.CHIS)} chi1 = "
          f"{len(S.TRAIN_Q)*len(S.CHIS)} points\n", flush=True)

    t0 = time.time(); cases = {}
    for q in S.TRAIN_Q:
        for c in S.CHIS:
            cases[(q, c)] = S.build_case(q, c)
    print(f"cache ready in {time.time()-t0:.0f}s", flush=True)

    # AUTO-RESUME.  The seed projection is poor (mean 6.06e-03 vs the predecessor's
    # 1.28e-03), so a pass-2 run that silently restarted from it would throw away the
    # whole of pass 1.  Resuming is therefore the default whenever a previous run has
    # written coefficients; --fresh is the explicit opt-out.
    resume = COEFFS.exists() and not args.fresh
    th = (np.array(json.loads(COEFFS.read_text())["theta"], float)
          if resume else seed_theta())
    print(f"start: {'RESUMED from ' + COEFFS.name if resume else 'FRESH seed'}")
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
    ship = json.loads((HERE / "pn_anchored_spin_results" / "coeffs_spin.json").read_text())
    print(f"\n{'chi1':>6} {'pn15 (11c)':>12} {'shipped (13c)':>14} {'ratio':>7}")
    for c in S.CHIS:
        new = np.median([allm[f"{q:.4f}_{c:+.1f}"] for q in S.TRAIN_Q])
        old = np.median([ship["mismatches"][f"{q:.4f}_{c:+.1f}"] for q in S.TRAIN_Q])
        print(f"{c:+6.1f} {new:12.4e} {old:14.4e} {new/old:6.2f}x")
    print("\nreference: " + ",  ".join(f"{k} {v:.4e}" for k, v in REF.items()))
    # Read the baseline from the shipped file rather than hardcoding it.  The literal
    # that used to sit here (1.09e-2 / 2.40e-3 / 7.43e-3) did NOT match
    # coeffs_spin.json and was almost certainly carried over from the superseded
    # quarantine run; the q=3 entry below is exactly that file's recorded `max`.
    print("\nis the q-U-shape flatter?  chi1 = +0.8 column, against the shipped form:")
    print(f"{'q':>6} {'pn15':>12} {'shipped':>12} {'ratio':>7}")
    for q in (3.0, 5.0, 8.0):
        k = f"{q:.4f}_+0.8"
        print(f"{q:6.1f} {allm[k]:12.4e} {ship['mismatches'][k]:12.4e} "
              f"{allm[k]/ship['mismatches'][k]:6.2f}x")
    sp = lambda d: max(d) / min(d)
    print(f"  spread across q:  pn15 {sp([allm[f'{q:.4f}_+0.8'] for q in (3.,5.,8.)]):.2f}x"
          f"   shipped {sp([ship['mismatches'][f'{q:.4f}_+0.8'] for q in (3.,5.,8.)]):.2f}x")
    COEFFS.write_text(json.dumps({
        "model": "pn_anchored + spin, 1.5PN spin-orbit + 2PN spin-spin",
        "form": {"beta_spin": "nu*(sSO*chi1*xc**1.5 + sSS*chi1**2*xc**2)",
                 "alpha_spin": "NONE"},
        "n_coefficients": 11, "names": NAMES, "theta": [float(x) for x in th],
        "grid": {"q": S.TRAIN_Q, "chi1": S.CHIS, "chi0_included": True},
        "median": float(np.median(errs)), "max": float(errs.max()),
        "mismatches": allm, "reference": REF}, indent=2))
    print(f"\nwrote {COEFFS}")


if __name__ == "__main__":
    main()
