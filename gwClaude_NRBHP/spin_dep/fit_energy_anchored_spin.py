"""gwr_energy_anchored extended to aligned primary spin, fitted on the 2d grid.

    alpha(t) = alpha_PP(q) * (1 + alpha_E(q) * E(t))       # E-driven, NO spin
    beta (t) = beta_PP(q)  * (1 + b_E(q,chi1) * E(t))      # ALL the spin lives here

    alpha_PP(q)  = X1**1.2 * (1 + c0*nu + c1*nu**2)                2
    alpha_E(q)   = A0 / nu                                         1
    beta_PP(q)   = X1**1.2 * (1 + b*nu)                            1
    P(nu,chi1)   = P0 + P1*nu + W*(s1*chi1 + s2*chi1**2)           2 + 2
    b_E(q,chi1)  = P(nu,chi1) / beta_PP(q)                  ---->  8 coefficients

WHAT THIS ISOLATES.  It differs from `fit_fluxanchored_spin.py` in EXACTLY ONE THING:
alpha reads E(t) instead of F(t) (and alpha_E carries A0/nu rather than A0*nu+A1*nu^2).
Everything else is shared -- switchless, no remnant anchor, beta driven by E with the
spin on P, one spin shape term.  So the comparison measures what alpha's drive is worth
under spin, with nothing else varying.

EXPECTATION, stated before running.  fluxanchored + spin lost to pn_anchored + spin
(1.411e-03 vs 7.686e-04) for two structural reasons, and **energy_anchored has both**:
  * no remnant anchor -- pn_anchored's `beta_r(q,chi1)` supplies ringdown spin
    dependence from RemS + the Kerr QNM fit for ZERO fitted parameters; a switchless
    E-driven model has nothing equivalent, and that is where the retrograde columns went
    (4.11e-03 vs 1.32e-03 at chi1 = -0.6);
  * one spin shape term -- `P*E` locks the spin correction's time PROFILE to E(t), only
    its amplitude moves, while the step-1 measurement showed R_beta's spin ratio varies
    with x by 0.17% to 3.35%.
So ~1.4e-03 is the honest prior.  A materially better number would mean alpha's drive
mattered more than either of those, which would itself be worth knowing.

WHY EVERYTHING IS REFITTED rather than freezing the nonspinning 6: BHPTNRSur1dq1e4 and
BHPTNRSur2dq1e3 at chi1 = 0 differ from each other by a mismatch of 8.7e-05 to 1.2e-04,
~25% of the residual these models fit, so coefficients are not portable between sources.
chi1 = 0 is INCLUDED in the objective because the nonspinning coefficients are free here
and that column constrains them.

ALPHA CARRIES NO SPIN, on the measurement that settled it for pn_anchored: |R_alpha - 1|
<= 0.7% against a within-case x-spread of 0.66-2.4%, and R_alpha - 1 CHANGES SIGN across
q at chi1 = +0.3/+0.6/+0.8 -- no power law in any mass-ratio variable can do that.  Both
checks are INSPIRAL ONLY; alpha's merger spin dependence remains untested.

Usage:
    python fit_energy_anchored_spin.py --weight nu  --refit --budget 100
    python fit_energy_anchored_spin.py --weight nu  --refit --resume --budget 100
    python fit_energy_anchored_spin.py --weight eps --refit --budget 100
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, minimize_scalar

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(PARENT))
warnings.filterwarnings("ignore")

import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_global as GG
import fit_pn_anchored_spin as S
import fit_fluxanchored_spin as FS      # reuses its (q,chi1) energy cache

RESULTS = HERE / "energy_anchored_spin_results"; RESULTS.mkdir(exist_ok=True)
NAMES = ["c0", "c1", "A0", "b", "P0", "P1", "s1", "s2"]
TRAIN_Q, CHIS = S.TRAIN_Q, S.CHIS
WEIGHT = "nu"
REF = {"pn_anchored+spin (13c)": 7.686e-04, "fluxanchored+spin (9c)": 1.4110e-03}


def seed_theta():
    """The shipped nonspinning 6 (1d fit) + zero spin.  A STARTING POINT ONLY -- the
    1d coefficients are not valid on this source (see module docstring)."""
    d = json.loads((PARENT / "gwr_energy_anchored_results" / "coeffs.json").read_text())
    return np.array(list(d["c"]) + list(d["A"]) + [d["b"]] + list(d["P"]) + [0.0, 0.0])


def params_at(q, chi1, th):
    c0, c1, A0, b, P0, P1, s1, s2 = th
    nu = q / (1 + q) ** 2; bs = (q / (1 + q)) ** 1.2
    w = (1.0 / q) if WEIGHT == "eps" else nu
    a_pp = float(np.clip(bs * (1 + c0 * nu + c1 * nu ** 2), 0.05, 2.5))
    b_pp = float(np.clip(bs * (1 + b * nu), 0.2, 1.6))
    a_e = float(A0) / nu
    P = float(P0 + P1 * nu + w * (s1 * chi1 + s2 * chi1 ** 2))
    return a_pp, a_e, b_pp, P / b_pp


def mismatch(th, q, chi1, case, t0_seed):
    a_pp, a_e, b_pp, b_e = params_at(q, chi1, th)
    t = np.asarray(case["t_bhpt"], float)
    E = np.asarray(case["losses"]["e_oft"], float)
    alpha = a_pp * (1.0 + a_e * E)
    beta = b_pp * (1.0 + b_e * E)
    if np.any(beta <= 0) or not np.all(np.isfinite(np.r_[alpha, beta])):
        return 50.0, t0_seed
    bc = G.creative.cumulative_trapezoid(beta, t)
    tau_shape = bc - float(np.interp(G.T_ANCHOR, t, bc))
    if np.min(np.diff(tau_shape)) <= 0:
        return 50.0, t0_seed
    f = lambda t0: GG.err_at_t0(t0, alpha, tau_shape, case["h_bhpt"],
                                case["t_nr"], case["h_nr"])
    if t0_seed is None:                       # first pass: coarse scan
        g = np.linspace(-250.0, 150.0, 81)
        t0_seed = float(g[int(np.argmin([f(x) for x in g]))])
    r = minimize_scalar(f, bounds=(t0_seed - 60.0, t0_seed + 60.0), method="bounded",
                        options={"xatol": 1e-3, "maxiter": 60})
    return float(r.fun), float(r.x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", choices=["nu", "eps"], default="nu")
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--maxiter", type=int, default=400)
    args = ap.parse_args()

    global WEIGHT
    WEIGHT = args.weight
    assert WEIGHT == args.weight
    coeffs = RESULTS / f"coeffs_{WEIGHT}.json"
    partial = RESULTS / f"partial_{WEIGHT}.json"
    print(f"energy_anchored + spin on the 2d grid: {len(TRAIN_Q)} q x {len(CHIS)} chi1 "
          f"= {len(TRAIN_Q)*len(CHIS)} points, chi1=0 INCLUDED")
    print(f"spin weight W = {'1/q' if WEIGHT=='eps' else 'nu'};  8 coefficients, "
          f"ALL fitted\n", flush=True)

    t0 = time.time(); cases = {}
    for q in TRAIN_Q:
        for c in CHIS:
            cases[(q, c)] = FS.build_case(q, c)    # reuses flux_loss_cache/
        print(f"  cached q={q:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"cache ready in {time.time()-t0:.0f}s", flush=True)

    th = (np.array(json.loads(coeffs.read_text())["theta"], float)
          if args.resume and coeffs.exists() else seed_theta())
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
            partial.write_text(json.dumps({"n_eval": st["n"], "elapsed_min": el/60,
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
    print(f"\n{'q':>8} " + " ".join(f"{c:>+9.1f}" for c in CHIS))
    for q in TRAIN_Q:
        print(f"{q:>8.4f} " + " ".join(f"{allm[f'{q:.4f}_{c:+.1f}']:9.2e}" for c in CHIS))
    print(f"\nALL 96: median {np.median(errs):.4e}  max {errs.max():.4e}")
    for c in CHIS:
        sub = np.array([allm[f"{q:.4f}_{c:+.1f}"] for q in TRAIN_Q])
        print(f"  chi1={c:+.1f}: median {np.median(sub):.4e}  max {sub.max():.4e}")
    print("\nreference, same grid: " + ",  ".join(f"{k} {v:.4e}" for k, v in REF.items()))
    coeffs.write_text(json.dumps({
        "model": "gwr_energy_anchored + spin, fitted on the 2d grid",
        "weight": WEIGHT, "n_coefficients": 8, "names": NAMES,
        "theta": [float(x) for x in th],
        "grid": {"q": TRAIN_Q, "chi1": CHIS, "chi0_included": True},
        "median": float(np.median(errs)), "max": float(errs.max()),
        "mismatches": allm, "reference": REF}, indent=2))
    print(f"\nwrote {coeffs}")


if __name__ == "__main__":
    main()
