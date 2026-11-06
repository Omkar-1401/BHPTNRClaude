"""Paradigm D: PySR symbolic regression of the PCA coefficients (direct or alpha-beta targets) over (nu, chi1).

Same PCA as bench_direct.Surrogate (fitted on the training pool).  One PySR search per coefficient, inputs nu and chi1
(raw, so the expressions read physically), operators + - * /, square, cube, sqrt, exp.  The expression per coefficient is
chosen on a fixed 20 % validation split of the training pool (lowest validation MSE along PySR's Pareto front, ties to the
simpler one); V300 waveform E scored once with the chosen expressions.

    python pysr_bench.py --target direct --K 6 --niter 300 --procs 120     -> results/pysr_<target>_K<K>.json
"""
from __future__ import annotations
import argparse, json, time
from multiprocessing import get_context

import numpy as np
import sympy

import common as C
import bench_direct as B

RES = C.HERE / "results"
SEED = 20261031


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="direct", choices=["direct", "ab"]); ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--niter", type=int, default=300); ap.add_argument("--procs", type=int, default=120)
    ap.add_argument("--maxsize", type=int, default=30); ap.add_argument("--nproc", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=150.0)   # seconds per coefficient search (budget rule)
    a = ap.parse_args()
    B.TARGET = a.target
    keys, A, P = B.load_train()
    sur = B.Surrogate("poly", "nu", a.K, params={"polynomialfeatures__degree": 1, "ridge__alpha": 1.0})
    sur.fit(keys, A, P, inner=False)                                # only for its PCA (mA, VA, ...), regressor unused
    Y = np.c_[(A - sur.mA) @ sur.VA.T, (P - sur.mP) @ sur.VP.T] / sur.ys
    q = keys[:, 0]; X = np.c_[q / (1 + q) ** 2, keys[:, 1]]
    r = np.random.default_rng(SEED); perm = r.permutation(len(keys)); va, tr = perm[: len(keys) // 5], perm[len(keys) // 5:]

    from pysr import PySRRegressor                                   # after numpy; Julia starts here
    exprs, out = [], dict(target=a.target, K=a.K, outputs=[])
    t0 = time.time()
    for j in range(Y.shape[1]):
        m = PySRRegressor(niterations=a.niter, binary_operators=["+", "-", "*", "/"],
                          unary_operators=["square", "cube", "sqrt", "exp"], maxsize=a.maxsize, populations=3 * a.procs // 4,
                          procs=a.procs, parallelism="multiprocessing", model_selection="accuracy",
                          variable_names=["nu", "chi"], random_state=SEED + j, verbosity=0, progress=False,
                          temp_equation_file=True, timeout_in_seconds=a.timeout)
        m.fit(X[tr], Y[tr, j])
        eqs = m.equations_; vm = []
        for i in range(len(eqs)):
            pv = m.predict(X[va], index=i); vm.append(float(np.mean((pv - Y[va, j]) ** 2)))
        vm = np.array(vm); i_best = int(np.argmin(vm + 1e-12 * eqs["complexity"].values))
        e = eqs.iloc[i_best]
        exprs.append((m, i_best))
        out["outputs"].append(dict(j=j, field=("A" if j < a.K else "P") + str(j % a.K), expr=str(e["sympy_format"]),
                                   complexity=int(e["complexity"]), val_mse=float(vm[i_best]),
                                   val_var=float(np.var(Y[va, j]))))
        print(f"coef {j:2d}: complexity {int(e['complexity']):2d}  val MSE / var {vm[i_best] / np.var(Y[va, j]):.3e}  "
              f"{e['sympy_format']}  ({time.time() - t0:.0f}s)", flush=True)
    out["search_s"] = time.time() - t0

    V = np.array(C.V300); qv = V[:, 0]; XV = np.c_[qv / (1 + qv) ** 2, V[:, 1]]
    YV = np.column_stack([m.predict(XV, index=i) for m, i in exprs]) * sur.ys
    AV = sur.mA + YV[:, :a.K] @ sur.VA; PV = sur.mP + YV[:, a.K:] @ sur.VP
    with get_context("fork").Pool(a.nproc) as pool:
        eV = np.array(pool.map(B._score, [(tuple(V[i]), AV[i], PV[i]) for i in range(len(V))], chunksize=2))
    out["V300"] = C.stats(eV); out["V300_cases"] = eV.tolist()
    out["total_complexity"] = int(sum(o["complexity"] for o in out["outputs"]))
    print(f"PySR {a.target} K={a.K}: V300 {C.fmt(C.stats(eV))}  total complexity {out['total_complexity']}", flush=True)
    (RES / f"pysr_{a.target}_K{a.K}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
