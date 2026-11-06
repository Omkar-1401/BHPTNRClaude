"""Rink et al. form with a higher spin order (user 2026-10-21: does a naive extra power help?).

  alpha(q, chi1) - 1 = (sum_{n=1..4} A_n q^-n)(1 + B_1 chi1 + ... + B_K chi1^K),  same form for beta.
K = 2 is the published form (fit_rink.py); this script runs K = 2 (sanity: must reproduce results/rink__G96__t96.json)
and the requested K = 3, 4.  Everything else is fit_rink.py unchanged: same gate, same step 1 (per-point constant
alpha, beta), step 2 least squares from the K = 2 refit padded with B_k = 0.  Training grid G96 only; scored on T96 and
V300.  Output: results_rink_chi/rink_chi{K}__G96.json (kept out of results/ so summarize.py is unaffected).

    python fit_rink_chi.py
"""
from __future__ import annotations
import json, sys, time
from multiprocessing import get_context

import numpy as np
from scipy.optimize import least_squares

import fit_rink as R                           # routing, guard, M.CASES machinery, scorer
G, C = R.G, R.C
OUT = G.HERE / "results_rink_chi"; OUT.mkdir(exist_ok=True)
KS = (2, 3, 4)


def val(co, q, chi):
    """Rink form with len(co) - 4 spin powers (co[:4] = A_1..A_4, co[4:] = B_1..B_K)."""
    q, chi = np.asarray(q, float), np.asarray(chi, float)
    P = sum(co[n] * q ** -(n + 1) for n in range(4)); S = sum(co[4 + m] * chi ** (m + 1) for m in range(len(co) - 4))
    return 1 + P * (1 + S)


def main():
    train = G.GRIDS[96]
    for k in sorted(set(train) | set(G.V300)):
        R.M.CASES[k] = R.M.prepare(*k)
    pubA, pubB = R.published(); R._CO.update(pubA=pubA, pubB=pubB)
    assert np.allclose(val(pubA, 4.0, 0.3), R.val(pubA, 4.0, 0.3)) and np.allclose(val(pubB, 6.0, -0.5), R.val(pubB, 6.0, -0.5))
    R.val = val                                # before any fork: workers' _score / _point use the general form
    ref = {G.key(r["q"], r["chi1"]): r["full_E_full"]
           for r in json.loads((R.SPLIT / "results" / "rink_scores.json").read_text())["T96"]}
    stored = json.loads((G.HERE / "results" / "rink__G96__t96.json").read_text())
    t = time.time()
    with get_context("fork").Pool(128) as pool:
        E = R.score(pool, pubA, pubB, G.T96)
        r = np.array([E[k] / ref[k] for k in G.T96])
        print(f"gate: published Rink / stored scores on T96 in [{r.min():.6f}, {r.max():.6f}]", flush=True)
        if np.any(np.abs(r - 1) > 1e-3):
            sys.exit("gate failed")
        pts = pool.map(R._point, train, chunksize=1)
    q = np.array([p[0][0] for p in pts]); ch = np.array([p[0][1] for p in pts])
    al = np.array([p[1] for p in pts]); be = np.array([p[2] for p in pts])
    prev = {C.kstr(p[0]): p for p in pts}
    d_ab = max(max(abs(prev[s][1] - v["alpha"]), abs(prev[s][2] - v["beta"])) for s, v in stored["per_point"].items())
    print(f"step 1: {len(pts)} per-point fits ({time.time() - t:.0f}s); max |d alpha|, |d beta| vs stored run {d_ab:.1e}", flush=True)
    start = {"alpha": np.array(stored["coef_alpha"]), "beta": np.array(stored["coef_beta"])}
    for K in KS:
        coef, resid = {}, {}
        for name, y in (("alpha", al), ("beta", be)):
            x0 = np.r_[start[name], np.zeros(K - 2)]
            r_ = least_squares(lambda c_: val(c_, q, ch) - y, x0, method="lm", xtol=1e-14, ftol=1e-14)
            coef[name], resid[name] = r_.x, float(np.sqrt(np.mean(r_.fun ** 2)))
            print(f"K={K} step 2 {name}: rms residual {resid[name]:.3e}; coefficients {np.round(r_.x, 5).tolist()}", flush=True)
        if K == 2:
            dc = max(np.abs(coef[n] - start[n]).max() for n in coef)
            print(f"K=2 sanity: max |coef - stored refit| {dc:.1e}", flush=True)
        with get_context("fork").Pool(128) as pool:
            out = {g: R.score(pool, coef["alpha"], coef["beta"], ks) for g, ks in (("T96", G.T96), ("V300", G.V300))}
        for g, v in out.items():
            print(f"  K={K} {g:5s} {C.fmt(C.stats(v))}", flush=True)
        (OUT / f"rink_chi{K}__G96.json").write_text(json.dumps(dict(
            model=f"rink_chi{K}", grid=96, form=f"1 + (sum_n=1..4 A_n q^-n)(1 + sum_k=1..{K} B_k chi1^k), alpha and beta",
            coef_alpha=list(map(float, coef["alpha"])), coef_beta=list(map(float, coef["beta"])), step2_rms=resid,
            stats={g: C.stats(v) for g, v in out.items()} | {"train_per_point_floor": C.stats([p[3] for p in pts])},
            per_case={g: {C.kstr(k): float(e) for k, e in v.items()} for g, v in out.items()},
            minutes=(time.time() - t) / 60), indent=1))
    print(f"done ({(time.time() - t) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
