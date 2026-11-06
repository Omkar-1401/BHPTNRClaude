"""Training-size refit of the exploration leader, pass 39 L4 (exploration/p39_numlimit.py; 18 free coefficients,
nu_c = 0.04), with that pass's own fit: A.refit (Powell on the mean fit-stride E over P6.T96, budget in minutes).
The training grid is swapped in by setting P6.T96 (read by the objective at call time) after the gate.

CAVEAT (recorded in every result): the model's fixed basis shapes (p12 spline shapes, pass-17 PCA) and every earlier
prune / term choice were derived on T96.  Only the 18 coefficients are refit here; the form itself carries T96
information, so for this model the learning curve measures coefficient sensitivity, not a from-scratch rebuild.
Gate: the shipped L4 theta reproduces p39/p39_L4.json's T96_cases on all 96 points (ratio within 2e-3, the pass's own
tolerance).

    python fit_p39.py --grid 24 --start t96 --budget 60
"""
from __future__ import annotations
import argparse, json, sys, time
import multiprocessing as mp

import numpy as np

import grids as G                              # FIRST: cache routing + sealed guard on S / FS
import common as C
EXP = G.REPO / "exploration"
sys.path.insert(0, str(EXP))
_argv = sys.argv
sys.argv = [_argv[0], "L4"]                    # p39_numlimit reads its variant from argv at import
import p39_numlimit as L                       # noqa: E402
sys.argv = _argv
P6, A = L.P6, L.A
assert L.NUC == 0.04
SRC = EXP / "p39" / "p39_L4.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, required=True, choices=list(G.GRIDS))
    ap.add_argument("--start", default="t96")
    ap.add_argument("--budget", type=float, default=60.0)
    ap.add_argument("--nproc", type=int, default=128)
    ap.add_argument("--gate-only", action="store_true")
    a = ap.parse_args()
    d = json.loads(SRC.read_text())
    th_ship = np.array(d["theta"], float); free = list(d["free"])
    P6.get = L.get; P6.ab = L.ab; L._NC["v"] = L.NUC          # BEFORE the fork (as p39_numlimit.main)
    train = G.GRIDS[a.grid]
    t = time.time()
    pool = mp.get_context("fork").Pool(a.nproc)
    full = lambda th, ks: dict(zip(ks, pool.map(P6._full, [(list(th), k) for k in ks])))
    E = full(th_ship, G.T96)
    r = np.array([E[k] / d["T96_cases"][f"{k[0]:.6f}_{k[1]:+.4f}"] for k in G.T96])
    print(f"gate: shipped p39 L4 recomputed / stored on T96 in [{r.min():.6f}, {r.max():.6f}] ({time.time() - t:.0f}s)", flush=True)
    if np.any(np.abs(r - 1) > 2e-3):
        sys.exit("gate failed: this scorer does not reproduce the shipped model")
    if a.gate_only:
        pool.close(); return

    P6.T96 = list(train)                                      # the objective of A.refit reads it at call time
    fmask = np.zeros(th_ship.size, bool); fmask[free] = True
    valid = lambda th: float(np.mean(pool.map(P6._fit, [(list(th), k) for k in train]))) < 1.0
    th0 = C.start_point(th_ship, a.start, valid, fmask)
    th, ef = A.refit(pool, th0, free, a.budget, f"p39L4 G{a.grid} {a.start}")
    print(f"fit: mean training (fit stride) {ef:.4e} ({(time.time() - t) / 60:.1f} min)", flush=True)
    out = {g: full(th, ks) for g, ks in (("train", train), ("T96", G.T96), ("V300", G.V300))}
    for g, v in out.items():
        print(f"  {g:5s} {C.fmt(C.stats(v))}", flush=True)
    ship_v = full(th_ship, G.V300)
    print(f"  shipped T96 coefficients on V300: {C.fmt(C.stats(ship_v))}", flush=True)
    pool.close()
    C.write("p39L4", a.grid, a.start, dict(
        caveat="basis shapes and term choices derived on T96; only the 18 coefficients are refit",
        free=free, theta=list(map(float, th)), theta_start=list(map(float, th0)), theta_shipped=list(map(float, th_ship)),
        source=str(SRC), fit=dict(best_mean_fit=ef, minutes=(time.time() - t) / 60, budget_min=a.budget),
        stats={g: C.stats(v) for g, v in out.items()} | {"V300_shipped": C.stats(ship_v)},
        per_case={g: {C.kstr(k): float(e) for k, e in v.items()} for g, v in out.items()}))


if __name__ == "__main__":
    main()
