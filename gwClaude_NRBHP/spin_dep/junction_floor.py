"""junction_floor -- PER-POINT FLOOR test: is the junction model's q=3 failure
UNDERFITTING (too few coefficients) or MISSPECIFICATION (wrong shape)?

The master junction fit (job 3530305) is worst at q=3: 5.5e-03 to 1.3e-02 across chi1,
against 7.1e-04 at its q=4.6 best.  Two possible causes, with different fixes:

  * UNDERFITTING / regression.  The 5 nu-flat coefficients cannot simultaneously serve
    q=3 and q=8, so the joint fit trades q=3 away.  Then the per-point optimum at q=3
    is MUCH better than the master's q=3 row, and the fix is in the regression layer
    (more degrees of freedom, or a different nu-form).
  * MISSPECIFICATION.  The junction SHAPE is wrong at q=3.  Then the per-point optimum
    is close to the master's q=3 row -- freeing the coefficients buys nothing -- and no
    amount of extra parameters helps.  The fix is structural.

The discriminator is the gap between the master row and the per-point optimum, read
against the same quantity for the 11-coefficient switched model (`pn_anc_spin_physical`)
fitted the same way.  That reference is what says whether ANY architecture does better
at q=3, i.e. whether q=3 is hard for the data or hard for the junction.

TWO MODES, and they answer different questions:

  --mode perq  (THE DISCRIMINATOR).  ONE coefficient set fitted to all 8 chi1 at this q.
      This is "per-q" in the workspace's sense: it frees the q-dependence ONLY, leaving
      the coefficients to serve the spin row exactly as the master must.  Its gap
      against the master row is q=3's share of the regression burden, which is the
      quantity the underfitting-vs-misspecification question turns on.

  --mode perpoint  (ABSOLUTE BOUND).  A SEPARATE coefficient set per (q, chi1).  Frees
      the q- AND chi1-dependence together, so its gap against the master absorbs the
      whole 96-point regression burden and cannot localise anything to q=3.  Useful only
      as the lower bound nothing can beat.

Usage:
    python junction_floor.py --q 3.0                  # both architectures, 8 chi1
    python junction_floor.py --q 3.0 --model junction # junction only (fast)
"""
from __future__ import annotations

import argparse, json, multiprocessing as mp, os, sys, time as _time, warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")

import fit_pn_anchored_spin as S
import fit_junction_spin as JF
import fit_pn_anc_spin_physical as PH

RESULTS = HERE / "junction_floor_results"; RESULTS.mkdir(exist_ok=True)
_CASES: dict = {}


def _score_one(arg):
    """One case's mismatch at a given theta -- the perq objective's inner loop."""
    model, th, q, c = arg
    if model == "junction":
        return JF.full_mm(np.asarray(th, float), q, c, _CASES[(q, c)])
    e, _ = PH.mismatch(np.asarray(th, float), q, c, _CASES[(q, c)], None)
    return e


def _fit_perq_serial(arg):
    """Per-q fit with a SERIAL inner loop, so the OUTER pool can run one q per worker.
    Used by --mode traj, where the deliverable is how each coefficient moves with nu."""
    model, q, CS, maxit = arg
    th0 = (np.array(json.loads((JF.RESULTS / "coeffs_junction.json").read_text())
                    ["theta"], float) if model == "junction" else PH.shipped11()[0])
    f = lambda th: float(np.mean([_score_one((model, th, q, c)) for c in CS]))
    e0 = f(th0)
    r = minimize(f, th0, method="Powell",
                 options={"maxiter": maxit, "xtol": 1e-3, "ftol": 1e-5})
    errs = [_score_one((model, r.x, q, c)) for c in CS]
    return dict(model=model, q=q, nu=q / (1 + q) ** 2, mode="traj", n_coef=len(th0),
                start=float(e0), mean=float(r.fun),
                per_chi={f"{c:+.1f}": float(e) for c, e in zip(CS, errs)},
                theta=np.asarray(r.x).tolist())


def _fit_perq(model, q, CS, maxit, pool):
    """ONE theta for all chi1 at this q.  Objective parallel across the spin row."""
    th0 = (np.array(json.loads((JF.RESULTS / "coeffs_junction.json").read_text())
                    ["theta"], float) if model == "junction" else PH.shipped11()[0])
    n = [0]

    def f(th):
        n[0] += 1
        return float(np.mean(pool.map(_score_one, [(model, th, q, c) for c in CS])))

    e0 = f(th0)
    r = minimize(f, th0, method="Powell",
                 options={"maxiter": maxit, "xtol": 1e-3, "ftol": 1e-5})
    errs = pool.map(_score_one, [(model, r.x, q, c) for c in CS])
    print(f"  {model:>9} perq: {n[0]} evals  start {e0:.3e} -> mean {r.fun:.3e}",
          flush=True)
    return dict(model=model, q=q, mode="perq", n_coef=len(th0), start=float(e0),
                mean=float(r.fun), per_chi={f"{c:+.1f}": float(e)
                                            for c, e in zip(CS, errs)},
                theta=np.asarray(r.x).tolist())


def _fit_one(arg):
    """One independent fit at a single (q, chi1)."""
    model, q, c, maxit = arg
    case = _CASES[(q, c)]
    if model == "junction":
        th0 = np.array(json.loads(
            (JF.RESULTS / "coeffs_junction.json").read_text())["theta"], float)
        f = lambda th: JF.full_mm(np.asarray(th, float), q, c, case)
    else:
        th0 = PH.shipped11()[0]
        nuis = {"t0": None}

        def f(th):
            e, t0 = PH.mismatch(np.asarray(th, float), q, c, case, nuis["t0"])
            nuis["t0"] = t0
            return e
    e0 = f(th0)
    r = minimize(f, th0, method="Powell",
                 options={"maxiter": maxit, "xtol": 1e-3, "ftol": 1e-5})
    return dict(model=model, q=q, chi1=c, n_coef=len(th0),
                start=float(e0), floor=float(r.fun), theta=np.asarray(r.x).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, nargs="+", default=[3.0])
    ap.add_argument("--chi", type=float, nargs="*", default=None)
    ap.add_argument("--model", nargs="*", default=["junction", "physical"])
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--nproc", type=int, default=0)
    ap.add_argument("--mode", default="perq",
                    choices=["perq", "perpoint", "traj"])
    a = ap.parse_args()

    CS = a.chi if a.chi else S.CHIS
    QS = a.q
    q = QS[0]
    for qq in QS:
        for c in CS:
            _CASES[(qq, c)] = S.build_case(qq, c)
            JF.der(qq, c)

    if a.mode == "traj":
        jobs = [(m, qq, CS, a.maxiter) for m in a.model for qq in QS]
        np_ = a.nproc or max(1, min(24, (os.cpu_count() or 4) - 2))
        np_ = min(np_, len(jobs))
        print(f"TRAJECTORY: {len(QS)} q x {len(CS)} chi1, one theta per q, "
              f"{len(jobs)} fits on {np_} workers", flush=True)
        t0 = _time.time()
        with mp.get_context("fork").Pool(np_) as pool:
            rows = pool.map(_fit_perq_serial, jobs)
        print(f"done in {_time.time()-t0:.0f}s\n", flush=True)
        master = json.loads((JF.RESULTS / "coeffs_junction.json").read_text())
        mth = master["theta"]; pc = master["per_case"]
        for m in a.model:
            rs = sorted([r for r in rows if r["model"] == m], key=lambda r: r["q"])
            nm = JF.NAMES if m == "junction" else PH.NAMES
            print(f"=== {m} ({rs[0]['n_coef']} coef): per-q optimum vs nu ===")
            print(f"{'q':>6} {'nu':>7} {'floor':>10} {'master':>10} {'gain':>6} " +
                  " ".join(f"{n:>9}" for n in nm))
            for r in rs:
                mm = np.mean([pc.get(f"{r['q']:.4f}_{c:+.1f}", np.nan) for c in CS])
                print(f"{r['q']:>6.3f} {r['nu']:>7.4f} {r['mean']:>10.3e} "
                      f"{mm:>10.3e} {mm/r['mean']:>6.2f} " +
                      " ".join(f"{v:>9.4f}" for v in r["theta"]))
            if m == "junction":
                print(f"\n{'coef':>6} {'master':>10} {'min':>10} {'max':>10} "
                      f"{'spread%':>9} {'lin R^2':>8} {'monotone':>9}")
                nuv = np.array([r["nu"] for r in rs])
                for i, n in enumerate(nm):
                    v = np.array([r["theta"][i] for r in rs])
                    A = np.polyfit(nuv, v, 1); res = v - np.polyval(A, nuv)
                    ss = 1 - res.var() / v.var() if v.var() > 0 else 1.0
                    d = np.diff(v)
                    mono = "yes" if (np.all(d > 0) or np.all(d < 0)) else "NO"
                    sp = 100 * (v.max() - v.min()) / max(abs(np.mean(v)), 1e-12)
                    print(f"{n:>6} {mth[i]:>10.4f} {v.min():>10.4f} {v.max():>10.4f} "
                          f"{sp:>9.1f} {ss:>8.3f} {mono:>9}")
                print("\n  spread% large + monotone + high R^2 -> a linear nu term will"
                      " capture it")
                print("  monotone NO                          -> a line tilts the wrong"
                      " way; do NOT add one (cf. alpha_E in scaling_gwr_energy_stiff.md)")
            print()
        out = RESULTS / "traj.json"
        out.write_text(json.dumps({"q": QS, "chi1": CS, "rows": rows}, indent=1))
        print(f"wrote {out}")
        return
    nproc = a.nproc or max(1, min(24, (os.cpu_count() or 4) - 2))
    t0 = _time.time()
    if a.mode == "perq":
        nproc = min(nproc, len(CS))
        print(f"q={q:g}: PER-Q fit ({len(CS)} chi1 share one theta) on {nproc} workers",
              flush=True)
        with mp.get_context("fork").Pool(nproc) as pool:
            rows = [_fit_perq(m, q, CS, a.maxiter, pool) for m in a.model]
    else:
        jobs = [(m, q, c, a.maxiter) for m in a.model for c in CS]
        nproc = min(nproc, len(jobs))
        print(f"q={q:g}: PER-POINT floor, {len(jobs)} independent fits on {nproc} "
              f"workers", flush=True)
        with mp.get_context("fork").Pool(nproc) as pool:
            rows = pool.map(_fit_one, jobs)
    print(f"done in {_time.time()-t0:.0f}s\n", flush=True)

    # master q-row for reference
    master = json.loads((JF.RESULTS / "coeffs_junction.json").read_text())["per_case"]
    mrow = {c: master.get(f"{q:.4f}_{c:+.1f}", float("nan")) for c in CS}

    if a.mode == "perq":
        print(f"{'chi1':>6} {'master(5c)':>11} " +
              " ".join(f"{m+'_perq':>14}" for m in a.model) + f" {'gain':>7}")
        R = {r["model"]: r for r in rows}
        for c in CS:
            k = f"{c:+.1f}"
            line = f"{c:>+6.2f} {mrow[c]:>11.3e} "
            for m in a.model:
                line += f"{R[m]['per_chi'][k]:>14.3e} "
            g = mrow[c] / R["junction"]["per_chi"][k]
            print(line + f"{g:>7.2f}")
        mv = np.array([mrow[c] for c in CS])
        jm = np.mean([R["junction"]["per_chi"][f"{c:+.1f}"] for c in CS])
        print(f"\nmaster q={q:g} mean {mv.mean():.3e} | " +
              " | ".join(f"{m} per-q mean {R[m]['mean']:.3e} ({R[m]['n_coef']}c)"
                         for m in a.model))
        print("\n--- VERDICT ---")
        print(f"master / junction-per-q = {mv.mean()/jm:.2f}x")
        print("  >3x -> the 5 coefficients CAN serve q=3; the joint fit trades it away"
              " => REGRESSION problem")
        print("  ~1x -> freeing all 5 AT q=3 buys nothing => junction SHAPE wrong at"
              " q=3 => MISSPECIFICATION")
        if "physical" in R:
            print(f"junction-per-q / physical-per-q = "
                  f"{R['junction']['mean']/R['physical']['mean']:.2f}x"
                  "   (>1 => the junction architecture itself is the limit at q=3)")
        out = RESULTS / f"perq_q{q:g}.json"
        out.write_text(json.dumps({"q": q, "chi1": CS, "master_row": mrow,
                                   "rows": rows}, indent=1))
        print(f"\nwrote {out}")
        return

    by = {m: {r["chi1"]: r for r in rows if r["model"] == m} for m in a.model}
    print(f"{'chi1':>6} {'master(5c)':>11} " +
          " ".join(f"{m+'_floor':>15}" for m in a.model) + f" {'gain':>7}")
    for c in CS:
        line = f"{c:>+6.2f} {mrow[c]:>11.3e} "
        for m in a.model:
            line += f"{by[m][c]['floor']:>15.3e} "
        g = mrow[c] / by["junction"][c]["floor"] if "junction" in by else float("nan")
        print(line + f"{g:>7.2f}")

    print()
    for m in a.model:
        v = np.array([by[m][c]["floor"] for c in CS])
        print(f"{m:>10} ({by[m][CS[0]]['n_coef']} coef): "
              f"median {np.median(v):.3e}  min {v.min():.3e}  max {v.max():.3e}")
    mv = np.array([mrow[c] for c in CS])
    print(f"{'master':>10} (5 coef): median {np.median(mv):.3e}")

    jv = np.array([by["junction"][c]["floor"] for c in CS]) if "junction" in by else None
    print("\n--- VERDICT ---")
    if jv is not None:
        gain = np.median(mv) / np.median(jv)
        print(f"master/junction-floor = {gain:.2f}x")
        print("  >3x  -> the 5 coefficients CAN fit q=3; the joint fit trades it away"
              "  => REGRESSION problem")
        print("  ~1x  -> freeing all 5 at q=3 buys nothing => the junction SHAPE is"
              " wrong at q=3 (MISSPECIFICATION)")
    if "physical" in by and jv is not None:
        pv = np.array([by["physical"][c]["floor"] for c in CS])
        print(f"junction-floor / physical-floor = {np.median(jv)/np.median(pv):.2f}x"
              "   (>1 means the junction architecture itself is the limit at q=3)")

    out = RESULTS / f"floor_q{q:g}.json"
    out.write_text(json.dumps({"q": q, "chi1": CS, "master_row": mrow, "rows": rows},
                              indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
