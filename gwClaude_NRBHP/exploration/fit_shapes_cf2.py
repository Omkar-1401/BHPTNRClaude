"""Closed-form BETA shapes that preserve the integrated clock (pass 47b; see EXPLORATION.md pass 47).

phi_B, psi_B act on beta, so the cumulative integral of their error over the inspiral moves the time map.  Form:
    f(s) = A0 exp(min(s,0)/tau) [A0 for s >= 0] + sum_{k=1..NB} A_k sech^2((s - mu_k)/w_k)      (sech^2 = d tanh / ds)
fitted on s in [-2000, 60] m1 to the pointwise error AND its cumulative integral (weight 0.02 per m1); best of 48 seeds
(parallel), NB = 4, 5, 6.  The alpha shapes keep the pointwise fits of shapes_closedform.json.
    python fit_shapes_cf2.py      -> shapes_closedform_v2.json
"""
import json
import numpy as np
from multiprocessing import get_context
from scipy.optimize import least_squares
from scipy.integrate import cumulative_trapezoid as ctz
import bhpt_spin_calib_p40 as G

S = G.shapes(); s = np.linspace(-2000, 60, 4121)


def f2(p, s, nb):
    s = np.asarray(s, float); y = p[0] * np.where(s < 0, np.exp(np.minimum(s, 0) / abs(p[1])), 1.0)
    for k in range(nb):
        A, mu, w = p[2 + 3 * k: 5 + 3 * k]; y = y + A / np.cosh(np.clip((s - mu) / abs(w), -300, 300)) ** 2
    return y


def one(args):
    key, nb, seed = args; y = G._ev(S[key], s); rng = np.random.default_rng(seed)
    p0 = [0.03, 600.0] + sum([[rng.normal(0, 0.3), rng.uniform(-200, 40), rng.uniform(4, 120)] for _ in range(nb)], [])
    res = lambda p: np.r_[f2(p, s, nb) - y, 0.02 * ctz(f2(p, s, nb) - y, s, initial=0)[::5]]
    r = least_squares(res, p0, max_nfev=8000)
    e = f2(r.x, s, nb) - y; I = ctz(e, s, initial=0)
    return key, nb, float(np.max(np.abs(I)) + 20 * np.max(np.abs(e))), r.x.tolist(), float(np.max(np.abs(e))), float(np.max(np.abs(I)))


def main():
    tasks = [(k, nb, sd) for k in ("phiB", "psiB") for nb in (4, 5, 6) for sd in range(48)]
    with get_context("fork").Pool(96) as pool:
        res = pool.map(one, tasks, chunksize=1)
    out = json.loads(open("shapes_closedform.json").read())
    for key in ("phiB", "psiB"):
        for nb in (4, 5, 6):
            b = min([r for r in res if r[0] == key and r[1] == nb], key=lambda r: r[2])
            print(f"{key}: tail + {nb} sech^2 ({2 + 3 * nb} constants): pointwise max {b[4]:.4f}, cumulative max {b[5]:.3f} m1", flush=True)
            if nb == 6:
                out[key] = dict(form="sech2", nb=nb, params=b[3], max=b[4], max_cum_int=b[5], clip=out[key]["clip"])
    out["note_beta"] = "phiB, psiB: tail + 6 sech^2 bumps fitted to the value AND its cumulative integral (fit_shapes_cf2.py)"
    json.dump(out, open("shapes_closedform_v2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
