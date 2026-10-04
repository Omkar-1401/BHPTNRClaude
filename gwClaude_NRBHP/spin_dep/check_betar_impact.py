"""How much does the beta_r_phys (1+q)/q correction actually move the SHIPPED models?

Scores pn_anc_spin_physical's shipped 11 coefficients on the 96-point grid twice --
once with beta_r_phys as shipped, once with the (1+q)/q removed -- with NO refit.  That
isolates the correction from any re-optimisation.  A third pass refits `r0` ALONE
(1 parameter) to show how much of the change r0 can absorb.
"""
import json, multiprocessing as mp, os, sys, warnings
import numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, "."); sys.path.insert(0, "..")
warnings.filterwarnings("ignore")
import fit_pn_anchored_spin as S
import fit_pn_anc_spin_physical as PH

_orig = S.beta_r_phys
def _fixed(q, chi1):                      # no (1+q)/q
    return _orig(q, chi1) * (q / (1.0 + q))

_CASES = {}
def _score(arg):
    th, q, c, fix = arg
    S.beta_r_phys = _fixed if fix else _orig
    PH.S.beta_r_phys = S.beta_r_phys
    e, _ = PH.mismatch(np.asarray(th, float), q, c, _CASES[(q, c)], None)
    return e

if __name__ == "__main__":
    th = PH.shipped11()[0]
    QS, CS = S.TRAIN_Q, S.CHIS
    for q in QS:
        for c in CS:
            _CASES[(q, c)] = S.build_case(q, c)
    keys = [(q, c) for q in QS for c in CS]
    pool = mp.get_context("fork").Pool(min(24, (os.cpu_count() or 4) - 2))

    def grid(thv, fix):
        return np.array(pool.map(_score, [(thv, q, c, fix) for q, c in keys]))

    a = grid(th, False); b = grid(th, True)
    print(f"{'variant':>34} {'mean':>11} {'median':>11} {'max':>11}")
    print(f"{'shipped beta_r (with (1+q)/q)':>34} {a.mean():>11.4e} "
          f"{np.median(a):>11.4e} {a.max():>11.4e}")
    print(f"{'corrected beta_r, NO refit':>34} {b.mean():>11.4e} "
          f"{np.median(b):>11.4e} {b.max():>11.4e}")

    i_r0 = PH.NAMES.index("r0")
    def f(r0):
        t2 = th.copy(); t2[i_r0] = r0
        return grid(t2, True).mean()
    r = minimize_scalar(f, bounds=(0.2, 3.0), method="bounded",
                        options={"xatol": 1e-3, "maxiter": 40})
    t2 = th.copy(); t2[i_r0] = r.x; c_ = grid(t2, True)
    print(f"{'corrected, r0 refit only':>34} {c_.mean():>11.4e} "
          f"{np.median(c_):>11.4e} {c_.max():>11.4e}   r0 {th[i_r0]:.4f} -> {r.x:.4f}")
    pool.close(); pool.join()
    json.dump({"shipped": a.tolist(), "corrected": b.tolist(),
               "corrected_r0refit": c_.tolist(), "r0_old": float(th[i_r0]),
               "r0_new": float(r.x), "keys": [f"{q:.4f}_{c:+.1f}" for q, c in keys]},
              open("junction_floor_results/betar_impact.json", "w"), indent=1)
