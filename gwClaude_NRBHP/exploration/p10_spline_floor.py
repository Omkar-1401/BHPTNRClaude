"""EXPLORATION pass 10 -- free-form (spline) floor: is chi1 = +0.8 limited by the FORM or by the alpha-beta FRAMEWORK?

Per case, start from the p5b M2 floor solution and add free multiplicative corrections on the ppBHPT clock
(s = t - t_peak,pp, m1):
    ln beta  += Sb(s),   ln alpha += Sa(s),   Sb, Sa = natural cubic splines through knot values (0 outside the knots)
knots s = -2000, -800, -400, -200, -120, -80, -50, -30, -18, -9, 0, 8, 18, 35, 60   (15 each -> 30 parameters)
Powell on the fit-stride error, full-resolution reported.  Cases: chi1 = +0.8 at q = 3, 5, 8; controls chi1 = 0 and
-0.6 at q = 5.  If the spline floor stays near 1e-3 at +0.8, the alpha-beta framework (not the form) is the limit there.
Writes exploration/p10/p10_spline.json.
"""
import json, sys
from pathlib import Path
import multiprocessing as mp
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p5b_floor as F

OUT = HERE / "p10"; OUT.mkdir(exist_ok=True)
KN = np.array([-2000, -800, -400, -200, -120, -80, -50, -30, -18, -9, 0, 8, 18, 35, 60], float)
CASES = [(3.0, 0.8), (5.0, 0.8), (8.0, 0.8), (5.0, 0.0), (5.0, -0.6)]
FLOOR = {(r["q"], r["chi1"]): np.array(r["M2"]["p"]) for r in json.loads((HERE / "p5b" / "p5b_floor.json").read_text())}


def spl(vals, s):
    y = CubicSpline(KN, vals, bc_type="natural")(np.clip(s, KN[0], KN[-1]))
    return y


def run(k):
    C = F.Case(k); p0 = FLOOR[k]
    base_ab = C.ab

    def err(z, key="fit", wide=False):
        zb, za = z[:len(KN)], z[len(KN):]
        def ab(p, form, key_):
            a, b = base_ab(p, form, key_)
            if a is None:
                return a, b
            t = C.case[("fit_" if key_ == "fit" else "") + "t_bhpt"]
            s = t - C.tpk
            return a * np.exp(spl(za, s)), b * np.exp(spl(zb, s))
        C.ab = ab
        e = C.err(p0, "M2", key, wide=wide)
        C.ab = base_ab
        return e

    z0 = np.zeros(2 * len(KN))
    e0 = err(z0, "full", wide=True)
    C.t0 = None; err(z0, "fit", wide=True)
    best = (np.inf, z0)
    for it in range(3):
        r = minimize(lambda z: err(z), best[1], method="Powell", options={"xtol": 1e-5, "ftol": 1e-8, "maxiter": 30})
        if r.fun < best[0]:
            best = (r.fun, r.x)
    e1 = err(best[1], "full", wide=True)
    print(f"q {k[0]:.1f} chi1 {k[1]:+.1f}: M2 floor {e0:.3e} -> + free splines {e1:.3e}", flush=True)
    return dict(q=k[0], chi1=k[1], E_M2_floor=e0, E_spline=e1, z=list(map(float, best[1])))


if __name__ == "__main__":
    with mp.get_context("fork").Pool(len(CASES)) as pool:
        rows = pool.map(run, CASES)
    (OUT / "p10_spline.json").write_text(json.dumps(rows, indent=1))
