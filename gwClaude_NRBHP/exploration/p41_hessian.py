"""EXPLORATION pass 41 -- parameter uncertainties of the pass-40 leader from the Hessian of the training objective
(user / collaborators, 2026-10-18: "check the sigma of each parameter to see which coefficients can be removed").  No fit.

Objective f(theta) = mean over T96 of the fit-stride mathcalE (the one the coefficients were fitted on), at the pass-40
point (pass 39 L4 coefficients + closed-form R2, O2).  Hessian H of f in the 18 free coefficients by finite differences:
  * per-coefficient step h_i chosen so that a one-sided step raises f by ~TARGET (1%) -- the objective has numerical
    noise from the per-case t0 optimisation, so steps must clear it; the whole Hessian is repeated with h/2 as a check.
  * t1a and t1b sit on a constraint (s1 = -150 m1 exactly at chi1 = -0.6; s1 < -150 makes that case invalid), so they
    are stepped only in the feasible direction (t1a up, t1b down) with one-sided second differences; every other pair
    uses central differences.
Noise model (needed only for the ABSOLUTE scale): treat N f / s^2 as a chi^2 with chi^2_min = N - p, i.e.
    cov = 2 f0 H^-1 / (N - p),  N = 96, p = 18.
Scale-free outputs: t_i = theta_i / sigma_i, the correlation matrix, the eigenvalues of H (degenerate directions), and the
predicted drop-and-refit cost of each coefficient, Delta f_i / f0 = t_i^2 / (N - p)  (to compare with pass 34's measured
single-drop costs).  Usage: python p41_hessian.py [target].  Writes exploration/p41/p41_hessian.json.
"""
from __future__ import annotations
import json, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
import p40_r2closed as C                        # noqa: E402  (imports pass 39 L4; C.use("O2n") installs the O2 R2)
L = C.L; P6 = C.P6
OUT = HERE / "p41"; OUT.mkdir(exist_ok=True)
R = json.loads((HERE / "p39" / "p39_L4.json").read_text())
TH0 = np.array(R["theta"]); FREE = list(R["free"])
NAMES = json.loads((HERE / "p32" / "p32_G2.json").read_text())["names"]
FN = [NAMES[i] for i in FREE]
N, P = len(P6.T96), len(FREE)
ONE_SIDED = {"t1a": +1.0, "t1b": -1.0}          # feasible direction for the two coefficients on the s1 bound
POOL = None


def f(z):
    th = TH0.copy(); th[FREE] = z
    return float(np.mean(POOL.map(P6._fit, [(list(th), k) for k in P6.T96])))


def steps(z0, f0):
    """h_i such that f(z0 + d_i h_i e_i) - f0 ~ TARGET f0 (geometric bracket, then one secant step)."""
    h = np.zeros(P)
    for i, n in enumerate(FN):
        d = ONE_SIDED.get(n, 1.0); hi = max(abs(z0[i]) * 1e-3, 1e-4)
        for _ in range(30):
            z = z0.copy(); z[i] += d * hi; r = f(z) / f0 - 1
            if r >= TARGET or r < 0 and abs(r) > 0.5:
                break
            hi *= 2.0
        if 0 < r:                                # quadratic scaling to hit TARGET
            hi *= np.sqrt(TARGET / r)
        h[i] = hi
        print(f"   step {n:8s} h = {hi:.4g}  (rise {100 * r:.2f}% at the bracket)", flush=True)
    return h


def hessian(z0, f0, h):
    H = np.zeros((P, P)); fc = {}

    def F(idx):
        key = tuple(sorted(idx.items()))
        if key not in fc:
            z = z0.copy()
            for i, m in idx.items():
                z[i] += m * h[i]
            fc[key] = f(z)
        return fc[key]
    one = [ONE_SIDED.get(n) for n in FN]
    for i in range(P):
        if one[i] is None:
            H[i, i] = (F({i: 1}) - 2 * f0 + F({i: -1})) / h[i] ** 2
        else:
            d = one[i]; H[i, i] = (F({i: 2 * d}) - 2 * F({i: d}) + f0) / h[i] ** 2
    g = np.array([(F({i: 1}) - F({i: -1})) / (2 * h[i]) if one[i] is None else
                  one[i] * (-3 * f0 + 4 * F({i: one[i]}) - F({i: 2 * one[i]})) / (2 * h[i]) for i in range(P)])
    for i in range(P):
        for j in range(i + 1, P):
            di = one[i] or 0; dj = one[j] or 0
            if not di and not dj:
                v = (F({i: 1, j: 1}) - F({i: 1, j: -1}) - F({i: -1, j: 1}) + F({i: -1, j: -1})) / (4 * h[i] * h[j])
            else:                                # forward differences in the constrained coordinate(s)
                si = di or 1; sj = dj or 1
                v = (F({i: si, j: sj}) - F({i: si}) - F({j: sj}) + f0) / (si * sj * h[i] * h[j])
            H[i, j] = H[j, i] = v
        print(f"   row {i + 1}/{P} done ({len(fc)} evaluations)", flush=True)
    return H, g


def analyse(H, z0, f0):
    w, V = np.linalg.eigh(H)
    Hi = np.linalg.inv(H)
    cov = 2 * f0 * Hi / (N - P)
    sig = np.sqrt(np.clip(np.diag(cov), 0, None))
    corr = cov / np.outer(sig, sig)
    t = z0 / sig
    return dict(eig=w.tolist(), eigvec=V.tolist(), sigma=sig.tolist(), t=t.tolist(), corr=corr.tolist(),
                pred_drop_cost=(t ** 2 / (N - P)).tolist(), diag_H_positive=bool(np.all(np.diag(H) > 0)),
                pos_def=bool(w.min() > 0))


def main():
    global POOL
    P6.get = L.get; P6.ab = L.ab; C.use("O2n")                    # BEFORE the fork
    POOL = mp.get_context("fork").Pool(96)
    t0 = time.time(); z0 = TH0[FREE].copy(); f0 = f(z0)
    print(f"pass 40 fit-stride T96 mean f0 = {f0:.6e}  ({time.time() - t0:.1f} s per evaluation)", flush=True)
    noise = np.std([f(z0) for _ in range(3)])
    print(f"   repeat-evaluation spread {noise:.2e} (relative {noise / f0:.1e})", flush=True)
    h = steps(z0, f0)
    out = dict(names=FN, theta=z0.tolist(), f0=f0, N=N, p=P, target=TARGET, steps=h.tolist(), one_sided=ONE_SIDED)
    for tag, hh in (("h", h), ("h_half", h / 2)):
        H, g = hessian(z0, f0, hh)
        dz = -np.linalg.solve(H, g)              # Newton step to the true optimum (pass 40 = L4 coefficients, not refit)
        out[tag] = dict(H=H.tolist(), grad=g.tolist(), newton_step=dz.tolist(),
                        newton_gain=float(0.5 * g @ dz / f0), **analyse(H, z0, f0))
        (OUT / "p41_hessian.json").write_text(json.dumps(out, indent=1))
        a = out[tag]
        print(f"{tag}: H positive definite {a['pos_def']}; eigenvalues {np.min(a['eig']):.3e} .. {np.max(a['eig']):.3e}; "
              f"distance to optimum: Newton gain {100 * a['newton_gain']:.2f}% of f0", flush=True)
        for n, th, s, tt, c in sorted(zip(FN, z0, a["sigma"], a["t"], a["pred_drop_cost"]), key=lambda r: abs(r[3])):
            print(f"   {n:8s} theta {th:+.5g}  sigma {s:.3g}  theta/sigma {tt:+7.2f}  predicted drop cost {100 * c:7.1f}%", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
