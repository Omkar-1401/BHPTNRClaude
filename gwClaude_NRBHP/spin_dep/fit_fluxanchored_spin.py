"""fluxanchored fitted from scratch on the 2d (spinning) grid, chi1 = 0 included.

    E(t) = gw_remnant Eoft        F(t) = despike(dE/dt)/max      # regularised drive
    alpha(t) = alpha_PP(q) * (1 + alpha_F(q) * F(t))             # NO spin
    beta (t) = beta_PP(q)  +  P(nu, chi1) * E(t)
    tau(t)   = t0_nr + integral_{T_ANCHOR}^{t} beta dt'

    alpha_PP(q)     = X1**1.2 * (1 + c0*nu + c1*nu**2)          2
    alpha_F(q)      = A0*nu + A1*nu**2                          2
    beta_PP(q)      = X1**1.2 * (1 + b*nu)                      1
    P(nu, chi1)     = P0 + P1*nu + W*(s1*chi1 + s2*chi1**2)      2 + 2
                                                          ---> 9 coefficients

WHY EVERYTHING IS REFITTED rather than freezing the nonspinning 7 (2026-09-09).  The
freeze-then-add-spin pattern used for pn_anchored assumed the nonspinning coefficients
transfer between waveform sources.  They do not, and the reason is quantitative: the
BHPTNRSur1dq1e4 and BHPTNRSur2dq1e3 waveforms at chi1 = 0 differ from EACH OTHER by a
mismatch of 8.7e-05 to 1.2e-04 (2.8e-05 in the inspiral alone).  Small absolutely, but
the models' own mismatches are 4.7e-04 to 6.1e-04, so the source difference is ~25% of
the residual being fitted.  Carrying 1d-fitted coefficients onto 2d waveforms therefore
moves the answer by tens of percent with a sign that depends only on how the two
residuals happen to align -- fluxanchored went 4.69e-04 -> 7.45e-04, pn_anchored went
6.148e-04 -> 5.543e-04.  Neither is evidence about the models; both are the same ~1e-04
coin landing differently.  Fitting on the grid the model is scored on removes it.

chi1 = 0 IS INCLUDED IN THE OBJECTIVE, unlike the pn_anchored spin fits, which excluded
it because the nonspinning coefficients were frozen there and its gradient w.r.t. the
spin coefficients is identically zero.  Here the nonspinning coefficients are free, so
that column carries real information about them.

ALPHA CARRIES NO SPIN.  Measured twice: |R_alpha - 1| <= 0.7% against a within-case
x-spread of 0.66-2.4% (a non-detection), and R_alpha - 1 CHANGES SIGN across q at
chi1 = +0.3, +0.6, +0.8 -- no power law in any mass-ratio variable can do that, so it is
not a mis-weighted physical term.  Both checks are inspiral-only.

W is the mass-ratio weight on the SPIN terms and is a FLAG, not a decision: the direct
measurement favours eps = 1/q (exponent k = 0.95 +- 0.24 against 1.45 +- 0.37 for nu),
but the pn_anchored eps run did not flatten the q-U-shape it was meant to fix, so the
question is open.  Both are run.

Note the despike is a NO-OP on this cache (0.03% on the null check) -- BHPTNRSur2dq1e3
carries no analogue of the 1d surrogate's 4-sample glitch at the one place it was
checked.  It is kept for symmetry with the nonspinning model, not because it does work.

Usage:
    python fit_fluxanchored_spin.py --cache-only
    python fit_fluxanchored_spin.py --weight nu  --refit --budget 100
    python fit_fluxanchored_spin.py --weight eps --refit --budget 100
    python fit_fluxanchored_spin.py --weight nu  --refit --resume --budget 100
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(PARENT))
warnings.filterwarnings("ignore")

import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_gwr_fluxanchored_reg as REG
import fit_pn_anchored_spin as S            # 2d case builder + the 96-point grid

RESULTS = HERE / "fluxanchored_spin_results"; RESULTS.mkdir(exist_ok=True)
LOSSCACHE = HERE / "flux_loss_cache"; LOSSCACHE.mkdir(exist_ok=True)
NAMES = ["c0", "c1", "A0", "A1", "b", "P0", "P1", "s1", "s2"]
TRAIN_Q, CHIS = S.TRAIN_Q, S.CHIS
WEIGHT = "nu"


def seed_theta():
    d = json.loads((PARENT / "gwr_fluxanchored_reg_results" / "coeffs_aF2.json").read_text())
    return np.array(list(d["c"]) + list(d["A"]) + [d["b"]] + list(d["P"]) + [0.0, 0.0])


def params_at(q, chi1, th):
    c0, c1, A0, A1, b, P0, P1, s1, s2 = th
    nu = q / (1 + q) ** 2; bs = (q / (1 + q)) ** 1.2
    w = (1.0 / q) if WEIGHT == "eps" else nu
    a_pp = float(np.clip(bs * (1 + c0 * nu + c1 * nu ** 2), 0.05, 2.5))
    b_pp = float(np.clip(bs * (1 + b * nu), 0.2, 1.6))
    a_f = float(A0 * nu + A1 * nu ** 2)
    P = float(P0 + P1 * nu + w * (s1 * chi1 + s2 * chi1 ** 2))
    return np.array([a_pp, a_f, b_pp, P / b_pp])


def build_case(q, chi1):
    """2d waveform pair + the energy/flux loss coordinates, cached to disk."""
    case = S.build_case(q, chi1)
    p = LOSSCACHE / f"loss_q{q:.6f}_chi{chi1:+.3f}.npz"
    t = np.asarray(case["t_bhpt"], float)
    # The spin energy/flux models use E per unit TOTAL mass (the nonspinning family uses per
    # PRIMARY mass; both valid, each better for its own models -- audit 2026-09-26), so the
    # units are requested explicitly rather than inherited from G's flag.
    # Drive (2026-09-26): physics_common.energy_drive with the 2d surrogate's join window --
    # E per unit TOTAL mass, E = cumtrapz of the despiked E_dot (was: raw), despike limited to
    # the join window (was: whole waveform, which rewrote real retrograde-ringdown structure).
    # Cache files record the drive string; a file built with any other drive is recomputed.
    import physics_common as PC
    z = np.load(p) if p.exists() else None
    if z is not None and "drive" in z.files and str(z["drive"]) == PC.ENERGY_DRIVE_2D:
        e, f = z["e"], z["f"]
    else:
        co = G.gwr_energy_coordinates(t, case["h_bhpt"], q, surrogate="2d", e_units="M")
        e = np.asarray(co["e_oft"], float)
        ed = np.asarray(co["e_dot"], float)                  # already despiked
        f = ed / float(np.max(ed))
        np.savez_compressed(p, e=e, f=f, E_units="M", drive=PC.ENERGY_DRIVE_2D)
    case["losses"] = {"e_oft": e, "flux_hat": f}
    tf = case["fit_t_bhpt"]
    case["fit_losses"] = {"e_oft": np.interp(tf, t, e), "flux_hat": np.interp(tf, t, f)}
    return case


def score(th, cases, nuis):
    errs = []
    for k, case in cases.items():
        q, c = k
        p = params_at(q, c, th)
        if nuis[k] is None:                      # first pass: coarse t0 scan
            g = np.linspace(-250.0, 150.0, 81)
            e0 = [FL.fast_mismatch(p, case, t0, "fit")[0] for t0 in g]
            nuis[k] = float(g[int(np.argmin(e0))])
        e, t0 = FL.fast_mismatch(p, case, nuis[k], "fit")
        nuis[k] = t0
        errs.append(FL.fast_mismatch(p, case, t0, "full")[0])
    return np.array(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", choices=["nu", "eps"], default="nu")
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--maxiter", type=int, default=400)
    args = ap.parse_args()

    global WEIGHT
    WEIGHT = args.weight
    assert WEIGHT == args.weight
    coeffs = RESULTS / f"coeffs_{WEIGHT}.json"
    partial = RESULTS / f"partial_{WEIGHT}.json"
    print(f"fluxanchored on the 2d grid: {len(TRAIN_Q)} q x {len(CHIS)} chi1 = "
          f"{len(TRAIN_Q)*len(CHIS)} points, chi1=0 INCLUDED")
    print(f"spin weight W = {'1/q' if WEIGHT == 'eps' else 'nu'};  9 coefficients, "
          f"ALL fitted\n", flush=True)

    t0 = time.time(); cases = {}
    for q in TRAIN_Q:
        for c in CHIS:
            cases[(q, c)] = build_case(q, c)
        print(f"  cached q={q:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"cache ready in {time.time()-t0:.0f}s", flush=True)
    if args.cache_only:
        return

    th = (np.array(json.loads(coeffs.read_text())["theta"], float)
          if args.resume and coeffs.exists() else seed_theta())
    print(f"seed: " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(NAMES, th)), flush=True)

    nuis = {k: None for k in cases}
    st = {"n": 0, "best": np.inf, "bx": th.copy(), "t0": time.time()}

    class _Budget(Exception):
        pass

    def obj(x):
        errs = score(x, cases, nuis)
        m = float(np.mean(errs)); md = float(np.median(errs))
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

    print(f"\nfitted: " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(NAMES, th)))
    errs = score(th, cases, nuis)
    allm = {f"{q:.4f}_{c:+.1f}": float(e)
            for (q, c), e in zip(cases.keys(), errs)}
    print(f"\n{'q':>8} " + " ".join(f"{c:>+9.1f}" for c in CHIS))
    for q in TRAIN_Q:
        print(f"{q:>8.4f} " + " ".join(f"{allm[f'{q:.4f}_{c:+.1f}']:9.2e}" for c in CHIS))
    print(f"\nALL 96: median {np.median(errs):.4e}  max {errs.max():.4e}")
    for c in CHIS:
        sub = np.array([allm[f"{q:.4f}_{c:+.1f}"] for q in TRAIN_Q])
        print(f"  chi1={c:+.1f}: median {np.median(sub):.4e}  max {sub.max():.4e}")
    print("\nreference, pn_anchored+spin on the SAME grid: median 7.863e-04 (13 coef)"
          "\n           and the 8-coef chi1-split:          median 7.245e-04")
    coeffs.write_text(json.dumps({
        "model": "gwr_energy_fluxanchored + spin, fitted on the 2d grid",
        "weight": WEIGHT, "n_coefficients": 9, "names": NAMES,
        "theta": [float(x) for x in th],
        "grid": {"q": TRAIN_Q, "chi1": CHIS, "chi0_included": True},
        "median": float(np.median(errs)), "max": float(errs.max()),
        "mismatches": allm}, indent=2))
    print(f"\nwrote {coeffs}")


if __name__ == "__main__":
    main()
