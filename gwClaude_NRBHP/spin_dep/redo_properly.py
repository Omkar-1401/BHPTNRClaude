"""Redo the units fix properly, one change at a time, with honest comparison.

PLAIN SUMMARY.  The ppBHPT and NR waveforms live on different clocks (primary-mass vs
total-mass units) and beta_r was missing the conversion between them.  Fixing it made
every model look worse, for two reasons that were BOTH errors in how I measured, not in
the physics:
  (a) the "before" column I compared against was not the published configuration;
  (b) each refit was a single Powell run from a seed that the fix had invalidated.

This script removes both problems:
  STEP 0  reproduce the published numbers EXACTLY.  If the per-case values do not match
          the ones stored in the coefficient file to ~1e-3 relative, everything stops --
          without a verified baseline no comparison means anything.
  STEP 1  apply ONLY the clock conversion (A1), multi-start refit.
  STEP 2  apply the clock conversion AND the constant correction (A2), multi-start refit.
Each step reports MEAN and MEDIAN, because the published references mix the two
(pn_anc_spin_physical records a mean of 1.1408e-03; the 13-coef predecessor's quoted
7.6858e-04 is a MEDIAN) and comparing one to the other is how this went wrong.

Nothing here writes to a shipped coefficients file.
"""
from __future__ import annotations
import argparse, json, multiprocessing as mp, os, sys, time, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")
import fit_pn_anchored_spin as S
import fit_pn_anc_spin_physical as PH

OUT = HERE / "redo_properly_results"; OUT.mkdir(exist_ok=True)
_FIXED = S.beta_r_phys           # source is already patched: A1 on, A2 per RESCALE_QNM
_CASES: dict = {}

#        b1     b2      a2     p00    p01     w0     r0     m0     m1    sSO    sSS
LO = np.array([-5, -30,  -60,   0,  -20,  0.30, 0.30,  0.0, -10, -20, -20], float)
HI = np.array([10,  30,   20,  10,   20,  6.00, 2.00,  3.0,  15,  20,  20], float)


def _set_mode(mode):
    """0 = published (no fix), 1 = clock fix only, 2 = clock + constant."""
    S.RESCALE_QNM = (mode == 2)
    if mode == 0:
        S.beta_r_phys = lambda q, c: _FIXED(q, c) / (q / (1.0 + q))
    else:
        S.beta_r_phys = _FIXED
    PH.S.beta_r_phys = S.beta_r_phys


def _score(arg):
    th, q, c, mode = arg
    _set_mode(mode)
    e, _ = PH.mismatch(np.asarray(th, float), q, c, _CASES[(q, c)], None)
    return e


def _fit_one(arg):
    idx, th0, mode, maxiter = arg
    _set_mode(mode)
    keys = list(_CASES)
    f = lambda th: float(np.mean([PH.mismatch(np.clip(th, LO, HI), q, c,
                                              _CASES[(q, c)], None)[0]
                                  for q, c in keys]))
    r = minimize(f, np.clip(th0, LO, HI),
                 method="Powell", bounds=list(zip(LO, HI)),
                 options={"maxiter": maxiter, "xtol": 1e-3, "ftol": 1e-5})
    th = np.clip(r.x, LO, HI)
    e = np.array([PH.mismatch(th, q, c, _CASES[(q, c)], None)[0] for q, c in keys])
    return dict(start=idx, mode=mode, mean=float(e.mean()),
                median=float(np.median(e)), max=float(e.max()), theta=th.tolist())


def main():
    # Budget is measured from PROCESS START, not from after case-building.
    # Job 3534807 set it after the ~20 min build, so budget 95 + setup blew
    # through the 2 h wall and mode 2 never ran.
    _T0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=10)
    ap.add_argument("--maxiter", type=int, default=25)
    ap.add_argument("--nproc", type=int, default=0)
    ap.add_argument("--budget", type=float, default=95.0)
    a = ap.parse_args()

    d = json.load(open(HERE / "pn_anc_spin_physical_results" / "coeffs_physical.json"))
    th_pub = np.array(d["theta"], float); pub = d["mismatches"]
    PUB_MEAN, PUB_MED = float(d["mean"]), float(d["median"])

    for k in pub:
        q, c = k.split("_"); _CASES[(float(q), float(c))] = S.build_case(float(q), float(c))
    keys = list(_CASES)
    nproc = a.nproc or max(1, min(24, (os.cpu_count() or 4) - 2))
    pool = mp.get_context("fork").Pool(nproc)
    print(f"{len(keys)} cases on {nproc} workers\n", flush=True)

    # ---- STEP 0: the baseline MUST reproduce ------------------------------
    e0 = np.array(pool.map(_score, [(th_pub, q, c, 0) for q, c in keys]))
    ratio = np.array([e / pub[f"{q:.4f}_{c:+.1f}"] for e, (q, c) in zip(e0, keys)])
    print(f"STEP 0  published config reproduced: mean {e0.mean():.4e} "
          f"(stored {PUB_MEAN:.4e}), median {np.median(e0):.4e} (stored {PUB_MED:.4e})")
    print(f"        per-case ratio  median {np.median(ratio):.5f}  "
          f"worst {np.abs(ratio-1).max()+1:.5f}")
    if np.abs(ratio - 1).max() > 2e-3:
        print("        *** BASELINE DOES NOT REPRODUCE -- STOPPING ***"); sys.exit(1)
    print("        baseline verified\n", flush=True)

    rng = np.random.default_rng(20260924)
    starts = [th_pub] + [np.clip(th_pub * rng.uniform(0.6, 1.5, th_pub.size), LO, HI)
                         for _ in range(a.starts - 1)]

    # PARALLELISE OVER CASES, NOT STARTS (fixed 2026-09-24).  The first version used
    # pool.map over starts with a SERIAL 96-case objective inside each: with 10 starts on
    # 24 workers that idled 14 cores and made one objective evaluation ~29 s, so Powell on
    # 11 parameters could not finish -- job 3534070 hit the 2 h wall having produced only
    # step 0.  Here each objective evaluation is spread over all 24 workers (~1.2 s) and
    # the starts run sequentially, with a wall-clock budget and a save after every start
    # so a kill can never cost more than one start.
    def fit_seq(s0, mode, maxiter):
        f = lambda t: float(np.mean(pool.map(
            _score, [(np.clip(t, LO, HI), q, c, mode) for q, c in keys])))
        r = minimize(f, np.clip(s0, LO, HI), method="Powell",
                     bounds=list(zip(LO, HI)),
                     options={"maxiter": maxiter, "xtol": 1e-3, "ftol": 1e-5})
        th = np.clip(r.x, LO, HI)
        e = np.array(pool.map(_score, [(th, q, c, mode) for q, c in keys]))
        return dict(mean=float(e.mean()), median=float(np.median(e)),
                    max=float(e.max()), theta=th.tolist())

    res = {0: dict(mean=float(e0.mean()), median=float(np.median(e0)))}
    for mode, name in ((1, "clock fix only"), (2, "clock + constant")):
        t0 = time.time(); rows = []
        for i, s0 in enumerate(starts):
            lim = a.budget * (0.5 if mode == 1 else 1.0)   # half each
            if (time.time() - _T0) / 60.0 > lim:
                print(f"        budget reached after {i} starts", flush=True); break
            rows.append(fit_seq(s0, mode, a.maxiter))
            rows.sort(key=lambda r: r["mean"])
            print(f"        start {i}: best so far mean {rows[0]['mean']:.4e}  "
                  f"median {rows[0]['median']:.4e}  ({time.time()-t0:.0f}s)", flush=True)
            (OUT / "physical_partial.json").write_text(json.dumps(
                {"mode": mode, "n_done": len(rows), "rows": rows}, indent=1, default=float))
        if not rows:
            print(f"        STEP {mode}: no starts completed", flush=True); break
        b = rows[0]
        print(f"STEP {mode}  {name}   ({time.time()-t0:.0f}s, {len(rows)} starts)")
        print(f"        best  mean {b['mean']:.4e}   median {b['median']:.4e}   "
              f"max {b['max']:.4e}")
        print(f"        spread over starts: mean "
              f"{min(r['mean'] for r in rows):.4e} .. {max(r['mean'] for r in rows):.4e}")
        print(f"        vs published        mean {PUB_MEAN:.4e}   median {PUB_MED:.4e}"
              f"   -> {100*(b['mean']/PUB_MEAN-1):+.1f}% mean, "
              f"{100*(b['median']/PUB_MED-1):+.1f}% median\n", flush=True)
        res[mode] = dict(best=b, all=rows)

    pool.close(); pool.join()
    (OUT / "physical.json").write_text(json.dumps(
        {"published": {"mean": PUB_MEAN, "median": PUB_MED}, "steps": res},
        indent=1, default=float))
    print(f"wrote {OUT/'physical.json'}")


if __name__ == "__main__":
    main()
