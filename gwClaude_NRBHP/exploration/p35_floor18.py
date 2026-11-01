"""EXPLORATION pass 35 -- pass-27 per-case spline floor + PCA on the 18c model (pass 33 A), to read the residual
structure at the spin extremes (chi1 = -0.6 and +0.8 columns fail U).  Writes exploration/p35/p35_floor.json.
"""
import json, sys
from pathlib import Path
import multiprocessing as mp
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, minimize_scalar
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0], "G2"]
import p31_spinq as M25                         # noqa: E402  (P6.ab = gamma-wrapped p25 model)
P6 = M25.P6
import fit_pn_anchored_spin as S                # noqa: E402
import fit_scaling_PN_opt_creative as creative  # noqa: E402

OUT = HERE / "p35"; OUT.mkdir(exist_ok=True)
KN = np.array([-2000, -800, -400, -200, -120, -80, -50, -30, -18, -9, 0, 8, 18, 35, 60], float)
TH = np.array(json.loads((HERE / "p33" / "p33_A.json").read_text())["theta"])


def spl(vals, s):
    return CubicSpline(KN, vals, bc_type="natural")(np.clip(s, KN[0], KN[-1]))


def run(k):
    c = P6.get(k); case = c["case"]
    base = {key: M25.ab(TH, k, key) for key in ("fit", "full")}
    st = {"t0": None}

    def err(z, key="fit", wide=False):
        pre = "fit_" if key == "fit" else ""
        t, h, tn, hn = case[pre + "t_bhpt"], case[pre + "h_bhpt"], case[pre + "t_nr"], case[pre + "h_nr"]
        a0, b0 = base[key]; s = t - c["tpk"]
        a = a0 * np.exp(spl(z[len(KN):], s)); b = b0 * np.exp(spl(z[:len(KN)], s))
        bc = creative.cumulative_trapezoid(b, t); ts = bc - np.interp(S.T_ANCHOR, t, bc)
        if np.min(np.diff(ts)) <= 0:
            return 1.0
        f = lambda x0: S._err_at_t0(x0, a, ts, t, h, tn, hn)
        if st["t0"] is None or wide:
            g = np.linspace(-300, 200, 126); c0 = g[int(np.argmin([f(x) for x in g]))]; w = 6
        else:
            c0, w = st["t0"], 15
        r = minimize_scalar(f, bounds=(c0 - w, c0 + w), method="bounded", options={"xatol": 1e-3})
        st["t0"] = float(r.x)
        return float(r.fun)

    z0 = np.zeros(2 * len(KN))
    e0 = err(z0, "full", wide=True)
    st["t0"] = None; err(z0, "fit", wide=True)
    best = (np.inf, z0)
    for _ in range(3):
        r = minimize(err, best[1], method="Powell", options={"xtol": 1e-5, "ftol": 1e-8, "maxiter": 30})
        if r.fun < best[0]:
            best = (r.fun, r.x)
    e1 = err(best[1], "full", wide=True)
    return dict(q=k[0], chi1=k[1], E_base=e0, E_spline=e1, z=list(map(float, best[1])))


if __name__ == "__main__":
    with mp.get_context("fork").Pool(96) as pool:
        rows = pool.map(run, P6.T96)
    n = len(KN)
    Z = np.array([r["z"] for r in rows]); chi = np.array([r["chi1"] for r in rows])
    out = dict(base="p25", knots=list(KN), rows=rows, pca={})
    E0 = np.array([r["E_base"] for r in rows]); E1 = np.array([r["E_spline"] for r in rows])
    print(f"T96: base mean {E0.mean():.3e} median {np.median(E0):.3e} -> + per-case splines mean {E1.mean():.3e} "
          f"median {np.median(E1):.3e} max {E1.max():.3e}", flush=True)
    for name, M, kn in (("dlnB", Z[:, :n - 2], KN[:n - 2]), ("dlnA", Z[:, n:-3], KN[:n - 3])):
        u, sv, vt = np.linalg.svd(M, full_matrices=False)
        frac = sv ** 2 / np.sum(sv ** 2)
        out["pca"][name] = dict(knots=list(kn), frac=list(frac[:4]), comps=[list(vt[j]) for j in range(3)],
                                amps=[list(M @ vt[j]) for j in range(3)])
        print(f"{name}: variance fractions {np.round(frac[:4], 3)}", flush=True)
        for j in range(2):
            a = M @ vt[j]
            print(f"   comp {j}: " + " ".join(f"{v:+.2f}" for v in vt[j]), flush=True)
            print("      amp by chi1: " + " ".join(f"{ch:+.1f}:{np.mean(a[chi == ch]):+.3f}±{np.std(a[chi == ch]):.3f}"
                                          for ch in S.CHIS), flush=True)
    (OUT / "p35_floor.json").write_text(json.dumps(out))
