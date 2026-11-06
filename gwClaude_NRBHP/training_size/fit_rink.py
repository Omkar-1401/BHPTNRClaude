"""Training-size refit of the Rink et al. 2024 (arXiv:2407.18319, Sec. IV) calibration FORM on our grids.

Their procedure (main.tex Eq. inner_product_alpha_beta, Eqs. alpha_fit / beta_fit):
  1. per training point: constant alpha, beta minimising the L2 error between alpha h_S(beta t) and NR (t0, phi0 free);
  2. fit  alpha(q, chi1) - 1 = (sum_{n=1..4} A_n q^-n)(1 + B_1 chi1 + B_2 chi1^2),  same form for beta  (6 + 6 = 12),
     by least squares on the per-point values.
Differences from the paper, kept so all models in this study share one scorer: their window is [-10000, 100] M and
their 90 points span q = 3..10; ours is the repo scorer (NR window [-5000.1, +100] M, full-window E, full alignment,
spin_dep_split/fit_pn_anchored_spin_split.score) and the study grids in q = 3..8.
Step 1 starts from the published model's alpha, beta at that point (from the surrogate h5; no NR data of ours in it);
NM on the fit-stride E, warm t0.  Step 2 starts from the published coefficients.  Starts 'pertN' perturb step 2's start.
Gate: published coefficients through this scorer reproduce results/rink_scores.json T96 (calibrated-output scores).

    python fit_rink.py --grid 24 [--start t96]
"""
from __future__ import annotations
import argparse, json, sys, time
from multiprocessing import get_context

import numpy as np
from scipy.optimize import minimize, least_squares

import grids as G                              # FIRST: cache routing + sealed guard on S / FS
import common as C
SPLIT = G.REPO / "spin_dep_split"
sys.path.insert(0, str(SPLIT))
import extrap_lowq as XL                       # noqa: E402
M = XL.M
assert M.__file__.startswith(str(SPLIT)), M.__file__
M.D.get_case = lambda q, chi, where="grid": G.FS.build_case(q, chi)
H5 = "/work2/11702/omkarsridhar1401/stampede3/BHPTNRSurrogate/data/BHPTNRSur2dq1e3.h5"


def published():
    import h5py
    with h5py.File(H5, "r") as f:
        g = f["nr_calib_params"]["(2,2)"]
        return np.array(g["alpha"], float), np.array(g["beta"], float)


def val(co, q, chi):
    q, chi = np.asarray(q, float), np.asarray(chi, float)
    P = sum(co[n] * q ** -(n + 1) for n in range(4)); S = co[4] * chi + co[5] * chi ** 2
    return 1 + P * (1 + S)


def E_const(al, be, k, key="full", seed=-50.0, wide=True):
    c = M.CASES[k]; pre = "fit_" if key == "fit" else ""
    a = np.full_like(c[pre + "t_bhpt"], al); b = np.full_like(c[pre + "t_bhpt"], be)
    (_, _, ef), t0 = M.score(a, b, c, key, "full", seed, wide)
    return ef, t0


def full_E(al, be, k):
    """The stored protocol (score_rink._check): fit-stride wide t0, then full resolution."""
    _, t0 = E_const(al, be, k, "fit", -50.0, True)
    return E_const(al, be, k, "full", t0, True)[0]


_CO = {}


def _score(arg):
    k, a, b = arg                              # coefficients travel with the task (workers hold a fork-time _CO)
    return k, full_E(float(val(a, *k)), float(val(b, *k)), k)


def _point(k):
    """Step 1 at one point: (alpha, beta) minimising fit-stride E (warm t0), polished; full-resolution E reported."""
    A, B = _CO["pubA"], _CO["pubB"]
    x0 = np.array([val(A, *k), val(B, *k)], float)
    st = {"seed": E_const(*x0, k, "fit", -50.0, True)[1]}

    def f(x):
        e, t0 = E_const(x[0], x[1], k, "fit", st["seed"], False)
        if e < 1.0:
            st["seed"] = t0
        return e
    x, fbest = x0, np.inf
    for rep in range(3):                                       # NM restarts, a wide t0 re-scan between them
        r = minimize(f, x, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-12})
        st["seed"] = E_const(r.x[0], r.x[1], k, "fit", st["seed"], True)[1]
        done = r.fun > fbest * (1 - 1e-3)
        if r.fun < fbest:
            x, fbest = r.x, float(r.fun)
        if done:
            break
    return k, float(x[0]), float(x[1]), full_E(x[0], x[1], k)


def score(pool, a, b, keys):
    a, b = list(map(float, a)), list(map(float, b))
    return dict(pool.map(_score, [(k, a, b) for k in keys], chunksize=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, required=True, choices=list(G.GRIDS))
    ap.add_argument("--start", default="t96")
    ap.add_argument("--gate-only", action="store_true")
    a = ap.parse_args()
    train = G.GRIDS[a.grid]
    for k in sorted(set(train) | set(G.T96) | set(G.V300)):
        M.CASES[k] = M.prepare(*k)
    pubA, pubB = published(); _CO.update(pubA=pubA, pubB=pubB)
    ref = {G.key(r["q"], r["chi1"]): r["full_E_full"]
           for r in json.loads((SPLIT / "results" / "rink_scores.json").read_text())["T96"]}
    t = time.time()
    with get_context("fork").Pool(128) as pool:
        E = score(pool, pubA, pubB, G.T96)
        r = np.array([E[k] / ref[k] for k in G.T96])
        print(f"gate: published Rink via repo map / calibrated-output scores on T96 in [{r.min():.6f}, {r.max():.6f}]", flush=True)
        if np.any(np.abs(r - 1) > 1e-3):
            sys.exit("gate failed: the constant alpha, beta map does not reproduce the published model's scores")
        if a.gate_only:
            return
        pts = pool.map(_point, train, chunksize=1)
    print(f"step 1: {len(pts)} per-point fits ({time.time() - t:.0f}s); per-point E median "
          f"{np.median([p[3] for p in pts]):.3e}, max {max(p[3] for p in pts):.2e}", flush=True)
    q = np.array([p[0][0] for p in pts]); ch = np.array([p[0][1] for p in pts])
    al = np.array([p[1] for p in pts]); be = np.array([p[2] for p in pts])
    coef = {}
    for name, y, pub in (("alpha", al, pubA), ("beta", be, pubB)):
        x0 = C.start_point(pub, a.start, lambda c_: bool(np.all(np.isfinite(val(c_, q, ch)))))
        r_ = least_squares(lambda c_: val(c_, q, ch) - y, x0, method="lm", xtol=1e-14, ftol=1e-14)
        coef[name] = r_.x
        print(f"step 2 {name}: rms residual {np.sqrt(np.mean(r_.fun ** 2)):.3e} (max {np.abs(r_.fun).max():.2e}); "
              f"coefficients {np.round(r_.x, 5).tolist()}  (published {np.round(pub, 5).tolist()})", flush=True)
    with get_context("fork").Pool(128) as pool:
        out = {g: score(pool, coef["alpha"], coef["beta"], ks) for g, ks in (("train", train), ("T96", G.T96), ("V300", G.V300))}
        pub_v = score(pool, pubA, pubB, G.V300)
    for g, v in out.items():
        print(f"  {g:5s} {C.fmt(C.stats(v))}", flush=True)
    print(f"  published Rink on V300: {C.fmt(C.stats(pub_v))}", flush=True)
    C.write("rink", a.grid, a.start, dict(
        form="1 + (sum_{n=1..4} A_n q^-n)(1 + B_1 chi1 + B_2 chi1^2), alpha and beta (Rink et al. 2024 Eqs. alpha_fit, beta_fit)",
        coef_alpha=list(map(float, coef["alpha"])), coef_beta=list(map(float, coef["beta"])),
        published_alpha=list(map(float, pubA)), published_beta=list(map(float, pubB)),
        per_point={C.kstr(p[0]): dict(alpha=p[1], beta=p[2], E=p[3]) for p in pts},
        fit=dict(minutes=(time.time() - t) / 60),
        stats={g: C.stats(v) for g, v in out.items()} | {"V300_published": C.stats(pub_v),
                                                       "train_per_point_floor": C.stats([p[3] for p in pts])},
        per_case={g: {C.kstr(k): float(e) for k, e in v.items()} for g, v in out.items()}))


if __name__ == "__main__":
    main()
