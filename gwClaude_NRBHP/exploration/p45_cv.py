"""EXPLORATION pass 45 -- 4-fold cross-validation of the pass-40 leader on T96 (selection-free generalisation; coefficient
spread across folds as an empirical sigma, to compare with pass 41's Hessian).

Folds: T96 point (iq, ic) -> fold (iq + ic) % 4 (24 points each; every q twice, every chi1 three times; the rule of
spin_dep_split/cv_split.py).  Per fold: the 18 coefficients refit on the other 72 (Powell on the fit-stride mean, started
from the full-T96 pass-40 coefficients -- a start, not data), then all 24 held-out points scored at full resolution.
    python p45_cv.py FOLD [FOLD ...] [--budget MIN]    -> exploration/p45/p45_fold<K>.json
"""
from __future__ import annotations
import argparse, json, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ap = argparse.ArgumentParser(); ap.add_argument("folds", nargs="+", type=int); ap.add_argument("--budget", type=float, default=40)
ap.add_argument("--model", default="p40", choices=["p40", "k1"])     # k1 = the lean variant (pass 42 K1, OT onset, 16c)
ARGS = ap.parse_args()
if ARGS.model == "k1":
    sys.argv = [sys.argv[0], "K1"]
    import p42_otonset as K42                   # noqa: E402
    C = K42.C
    R = json.loads((HERE / "p42" / "p42_K1.json").read_text())
else:
    sys.argv = [sys.argv[0], "L4"]
    import p40_r2closed as C                    # noqa: E402
    R = json.loads((HERE / "p39" / "p39_L4.json").read_text())
P6, A = C.P6, C.A
OUT = HERE / "p45"; OUT.mkdir(exist_ok=True)
TAG = "" if ARGS.model == "p40" else "_k1"
TH0 = np.array(R["theta"]); FREE = list(R["free"])
T96 = list(P6.T96); NC = 8
FOLD = {k: (i // NC + i % NC) % 4 for i, k in enumerate(T96)}


def _full(a):
    return P6._full(a)


def main():
    P6.get = C.L.get; P6.ab = C.L.ab; C.use("O2n")
    if ARGS.model == "k1":
        K42.M.S1FUN = K42.onset
    for f in ARGS.folds:
        t0 = time.time()
        train = [k for k in T96 if FOLD[k] != f]; test = [k for k in T96 if FOLD[k] == f]
        P6.T96 = train                           # A.refit's objective loops over P6.T96
        pool = mp.get_context("fork").Pool(96)
        th, ef = A.refit(pool, TH0, FREE, ARGS.budget, f"fold{f}")
        P6.T96 = T96
        e_te = np.array(pool.map(_full, [(list(th), k) for k in test]))
        e_tr = np.array(pool.map(_full, [(list(th), k) for k in train]))
        pool.close()
        out = dict(fold=f, theta=list(map(float, th)), free=FREE, fit_mean=ef,
                   test={f"{q:.6f}_{c:+.4f}": float(v) for (q, c), v in zip(test, e_te)},
                   train={f"{q:.6f}_{c:+.4f}": float(v) for (q, c), v in zip(train, e_tr)})
        (OUT / f"p45{TAG}_fold{f}.json").write_text(json.dumps(out, indent=1))
        print(f"{ARGS.model} fold {f}: train {np.median(e_tr):.3e} / {e_tr.mean():.3e} / {e_tr.max():.2e} | held-out {np.median(e_te):.3e} / "
              f"{e_te.mean():.3e} / {e_te.max():.2e}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
