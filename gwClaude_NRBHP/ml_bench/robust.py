"""Robustness of the paradigm-A learning curve: train on EXACTLY the grids the hand-built models were refit on in
training_size/ (G24, G48, T96 = G96, G192; same keys) and on 5 random pool draws per size; score V300.
Configs (features, K, method hyperparameter grid) = the CV-best of bench_direct; inner hyperparameters re-chosen on each
training set.  For the hand-built comparison read training_size/results/<model>__G<n>__*.json (not rerun here).

    python robust.py --methods rbf,gpr,poly --nproc 128      -> results/robust_direct.json
"""
from __future__ import annotations
import argparse, json
from multiprocessing import get_context

import numpy as np
import common as C
import bench_direct as B

RES = C.HERE / "results"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--methods", default="rbf,gpr,poly")
    ap.add_argument("--nproc", type=int, default=128); ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    keys, A, P = B.load_train()
    idx = {tuple(k): i for i, k in enumerate(map(tuple, keys))}
    V = np.array(C.V300); best = B.best_configs(); out = {}
    path = RES / "robust_direct.json"
    if path.exists():
        out = json.loads(path.read_text())
    with get_context("fork").Pool(a.nproc) as pool:
        for m in a.methods.split(","):
            _, kind, K = best[m].split("|")
            for n, grid in C.G.GRIDS.items():
                sets = {"grid": [idx[tuple(k)] for k in grid if tuple(k) in idx]}
                for s in range(a.seeds):
                    sets[f"rand{s}"] = list(np.random.default_rng(1000 + s).permutation(len(keys))[:n])
                for lab, ii in sets.items():
                    tag = f"{m}|{n}|{lab}"
                    if tag in out:
                        continue
                    ii = np.array(ii)
                    sur = B.Surrogate(m, kind, min(int(K), len(ii) - 1)).fit(keys[ii], A[ii], P[ii])
                    e = B.score_keys(pool, sur, V)
                    out[tag] = dict(C.stats(e), n_train=int(len(ii)), params=str(sur.params))
                    print(f"{tag:18s} (n_train {len(ii):3d}) V300 {C.fmt(C.stats(e))}  {sur.params}", flush=True)
                    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
