"""Is the corrected pn_anc_spin_physical (11c) really 23% worse than published, or was
Thursday's refit under-converged?  Small grid, equal effort.  Diagnosis only.

Small grid 4 q x 4 chi1 (16 cases).  Same model code (fit_pn_anc_spin_physical.mismatch,
full-resolution), OLD beta_r (published form, = new / X1) vs NEW (corrected), identical
starts mapped between the two conventions, identical wall-clock budget per run.
Output: spin_dep/diag_spin_pna_small.json
"""
from __future__ import annotations
import json, multiprocessing as mp, sys, time, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")
import fit_pn_anchored_spin as S
import fit_pn_anc_spin_physical as PH

QS = [3.0, 4.2, 5.9487179487, 8.0]
CHIS = [-0.6, -0.2, 0.2, 0.6]
BUDGET_S = 20 * 60
OUT = HERE / "diag_spin_pna_small.json"
NEW_BR = S.beta_r_phys
#              b1   b2   a2   p00  p01   w0    r0   m0   m1  sSO  sSS
LO = np.array([-5, -30, -60, -10, -20, 0.30, 0.30, 0.0, -10, -20, -20], float)
HI = np.array([10,  30,  20,  10,  20, 6.00, 2.00, 3.0,  15,  20,  20], float)
CASES = {}


def set_br(which):
    br = NEW_BR if which == "new" else (lambda q, c: NEW_BR(q, c) / (q / (1.0 + q)))
    S.beta_r_phys = br; PH.S.beta_r_phys = br


def objective(th):
    return float(np.mean([PH.mismatch(np.clip(th, LO, HI), q, c, CASES[(q, c)], None)[0]
                          for (q, c) in CASES]))


def run(arg):
    which, tag, th0 = arg
    set_br(which)
    t0 = time.time(); best = {"f": np.inf, "th": np.clip(th0, LO, HI)}

    class Done(Exception):
        pass

    def f(th):
        if time.time() - t0 > BUDGET_S:
            raise Done
        v = objective(th)
        if v < best["f"]:
            best.update(f=v, th=np.clip(th, LO, HI))
        return v

    f0 = f(best["th"])
    try:
        minimize(f, best["th"], method="Powell", bounds=list(zip(LO, HI)),
                 options={"maxiter": 400, "xtol": 1e-4, "ftol": 1e-6})
        st = "converged"
    except Done:
        st = "budget"
    e = [PH.mismatch(best["th"], q, c, CASES[(q, c)], None)[0] for (q, c) in CASES]
    return dict(beta_r=which, start=tag, status=st, f0=f0, mean=float(np.mean(e)),
                median=float(np.median(e)), max=float(np.max(e)), theta=best["th"].tolist())


def main():
    for q in QS:
        for c in CHIS:
            CASES[(q, c)] = S.build_case(q, c)
    pub = np.array(json.loads((HERE / "pn_anc_spin_physical_results/coeffs_physical.json")
                              .read_text())["theta"], float)
    rows = json.loads((HERE / "redo_properly_results/physical_partial.json").read_text())["rows"]
    corr = np.array(min(rows, key=lambda r: r["mean"])["theta"], float)
    c = float(np.mean([q / (1 + q) for q in S.TRAIN_Q]))       # new = old * X1
    starts = {"published": pub, "thu_corrected": corr}
    jobs = []
    for tag, th in starts.items():
        t_old = th.copy(); t_new = th.copy()
        if tag == "published":
            t_new[6] = th[6] / c          # r0 mapped so beta_r*r0 matches at mean X1
        else:
            t_old[6] = th[6] * c
        jobs += [("old", tag, t_old), ("new", tag, t_new)]
    print(f"{len(CASES)} cases, {len(jobs)} runs x {BUDGET_S/60:.0f} min", flush=True)
    res = []
    with mp.get_context("fork").Pool(len(jobs)) as pool:
        for r in pool.imap_unordered(run, jobs):
            res.append(r); OUT.write_text(json.dumps(res, indent=1))
            print(f"{r['beta_r']:>3} {r['start']:<14} {r['status']:<9} start {r['f0']:.4e} -> "
                  f"mean {r['mean']:.4e} median {r['median']:.4e} max {r['max']:.4e}", flush=True)
    for w in ("old", "new"):
        b = min((r for r in res if r["beta_r"] == w), key=lambda r: r["mean"])
        print(f"best {w}: mean {b['mean']:.4e} median {b['median']:.4e} (start {b['start']})")


if __name__ == "__main__":
    main()
