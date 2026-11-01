"""Closed-form versions of the four fixed shape functions (user, 2026-10-21: closed forms wherever possible).

Each spline shape (phi_B, phi_A from pass 12; psi_B, psi_A from pass 17) is fitted, offline, by
    f(s) = A0 exp(min(s, 0) / tau)  [held at A0 for s >= 0]  +  sum_{k=1..4} A_k sech((s - mu_k) / w_k)
on s in [-2000, 60] m1 (log-spaced before the peak, weight 1 for s > -300 and 0.3 earlier); 14 constants per shape, best of
12 seeded starts.  Outside [-2000, 60] the model clips s exactly as the splines do.  -> shapes_closedform.json
    python fit_shapes_closedform.py
"""
import json
import numpy as np
from scipy.optimize import least_squares
import bhpt_spin_calib_p40 as G

NB = 4


def shape_cf(p, s):
    s = np.asarray(s, float)
    y = p[0] * np.where(s < 0, np.exp(np.minimum(s, 0) / abs(p[1])), 1.0)
    for k in range(NB):
        A, mu, w = p[2 + 3 * k: 5 + 3 * k]; y = y + A / np.cosh((s - mu) / abs(w))
    return y


def main():
    S = G.shapes()
    s = np.unique(np.r_[-np.logspace(np.log10(2000), 0, 300), np.linspace(0, 60, 121)])
    wt = np.where(s > -300, 1.0, 0.3); out = {}
    for key in ("phiB", "phiA", "psiB", "psiA"):
        y = G._ev(S[key], s); tries = []
        for seed in range(12):
            rng = np.random.default_rng(seed); p0 = [0.05, 500.0]
            for k in range(NB):
                p0 += [rng.normal(0, 0.3), rng.uniform(-120, 40), rng.uniform(5, 60)]
            r = least_squares(lambda p: wt * (shape_cf(p, s) - y), p0, max_nfev=4000)
            tries.append((float(np.sqrt(np.mean((shape_cf(r.x, s) - y) ** 2))), r.x))
        rms, p = min(tries, key=lambda z: z[0])
        out[key] = dict(params=list(map(float, p)), rms=rms, max=float(np.max(np.abs(shape_cf(p, s) - y))),
                        clip=list(map(float, S[key][1])))
        print(f"{key}: rms {rms:.4f} max {out[key]['max']:.4f}", flush=True)
    out["form"] = "A0 exp(min(s,0)/tau) [A0 for s >= 0] + sum_k A_k sech((s - mu_k)/w_k), params = [A0, tau, (A_k, mu_k, w_k) x 4]; s clipped to `clip`"
    json.dump(out, open("shapes_closedform.json", "w"), indent=1)


if __name__ == "__main__":
    main()
