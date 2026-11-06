"""Training-size refits of the two current exploration candidates (user 2026-10-22), as fit_p39.py:
  p40   pass 40 = pass 39 L4 (18c) with the closed-form R2 O2 on its natural range (p40_r2closed.use("O2n")).
        shipped theta = p39/p39_L4.json (pass 40 froze L4's coefficients); gate: p40/p40.json["O2n"]["T96_cases"].
  lean  pass 42 K1 = pass 40 with the Ori–Thorne blend onset at T_start = -1 (onset coefficients t1a, t1b removed; 16c),
        exploration/p42_otonset.py (widened onset guard + M.S1FUN hook).  shipped theta / gate: p42/p42_K1.json.
  lean12, lean9  pass 49b / 49c: lean pruned to 12 / 9 coefficients (user 2026-10-28); same chain as lean, dropped
        coefficients held at 0, only the survivors refit.  shipped theta, free, gate: p49b/p49.json steps n = 12 / 9.
        Their surviving-coefficient SET was also chosen on T96 (stronger caveat than the basis alone).
Fit: A.refit (Powell on the mean fit-stride E over P6.T96, budget in minutes), training grid swapped in via P6.T96.
p42_otonset is imported only for 'lean': its import widens the onset guard of p25's ab for every model in the process.

CAVEAT (recorded in every result): basis shapes and term choices were derived on T96; only the coefficients are refit.

    python fit_p40.py p40 --grid 24 --start t96 --budget 75
"""
from __future__ import annotations
import argparse, json, sys, time
import multiprocessing as mp

import numpy as np

import grids as G                              # FIRST: cache routing + sealed guard on S / FS
import common as C
EXP = G.REPO / "exploration"
sys.path.insert(0, str(EXP))
MODEL = sys.argv[1] if len(sys.argv) > 1 else ""
assert MODEL in ("p40", "lean", "lean12", "lean9"), "usage: fit_p40.py p40|lean|lean12|lean9 --grid N [--start S] [--budget MIN]"
_argv = sys.argv
if MODEL.startswith("lean"):
    sys.argv = [_argv[0], "K1"]                # p42_otonset reads its variant from argv at import
    import p42_otonset as K                    # noqa: E402  (imports p40_r2closed -> p39_numlimit L4)
    R2C = K.C
else:
    K = None
    import p40_r2closed as R2C                 # noqa: E402  (sets argv to L4 for p39_numlimit)
sys.argv = _argv
L = R2C.L; P6, A = L.P6, L.A
assert L.NUC == 0.04
if MODEL in ("lean12", "lean9"):
    SRC = EXP / "p49b" / "p49.json"; N_KEEP = int(MODEL[4:])
    SHIP = [s for s in json.loads(SRC.read_text())["steps"] if s["n"] == N_KEEP][0]; REF = SHIP["T96_cases"]
    assert len(SHIP["free"]) == N_KEEP
elif MODEL == "lean":
    SRC = EXP / "p42" / "p42_K1.json"; SHIP = json.loads(SRC.read_text()); REF = SHIP["T96_cases"]
    assert SHIP["variant"] == "K1" and SHIP["n_coef"] == 16
else:
    SRC = EXP / "p39" / "p39_L4.json"; SHIP = json.loads(SRC.read_text())
    REF = json.loads((EXP / "p40" / "p40.json").read_text())["O2n"]["T96_cases"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["p40", "lean", "lean12", "lean9"])
    ap.add_argument("--grid", type=int, required=True, choices=list(G.GRIDS))
    ap.add_argument("--start", default="t96")
    ap.add_argument("--budget", type=float, default=60.0)
    ap.add_argument("--nproc", type=int, default=128)
    ap.add_argument("--gate-only", action="store_true")
    a = ap.parse_args()
    th_ship = np.array(SHIP["theta"], float); free = list(SHIP["free"])
    P6.get = L.get; P6.ab = L.ab; L._NC["v"] = L.NUC; R2C.use("O2n")     # BEFORE the fork (pass 40 base)
    if K is not None:
        K.M.S1FUN = K.onset                                              # OT onset hook (as p42_otonset.main)
    train = G.GRIDS[a.grid]
    t = time.time()
    pool = mp.get_context("fork").Pool(a.nproc)
    full = lambda th, ks: dict(zip(ks, pool.map(P6._full, [(list(th), k) for k in ks])))
    E = full(th_ship, G.T96)
    r = np.array([E[k] / REF[f"{k[0]:.6f}_{k[1]:+.4f}"] for k in G.T96])
    print(f"gate: shipped {a.model} recomputed / stored on T96 in [{r.min():.6f}, {r.max():.6f}] ({time.time() - t:.0f}s)", flush=True)
    if np.any(np.abs(r - 1) > 2e-3):
        sys.exit("gate failed: this scorer does not reproduce the shipped model")
    if a.gate_only:
        pool.close(); return

    P6.T96 = list(train)                                      # the objective of A.refit reads it at call time
    fmask = np.zeros(th_ship.size, bool); fmask[free] = True
    valid = lambda th: float(np.mean(pool.map(P6._fit, [(list(th), k) for k in train]))) < 1.0
    th0 = C.start_point(th_ship, a.start, valid, fmask)
    th, ef = A.refit(pool, th0, free, a.budget, f"{a.model} G{a.grid} {a.start}")
    print(f"fit: mean training (fit stride) {ef:.4e} ({(time.time() - t) / 60:.1f} min)", flush=True)
    out = {g: full(th, ks) for g, ks in (("train", train), ("T96", G.T96), ("V300", G.V300))}
    for g, v in out.items():
        print(f"  {g:5s} {C.fmt(C.stats(v))}", flush=True)
    ship_v = full(th_ship, G.V300)
    print(f"  shipped T96 coefficients on V300: {C.fmt(C.stats(ship_v))}", flush=True)
    pool.close()
    C.write(a.model, a.grid, a.start, dict(
        caveat=f"basis shapes and term choices derived on T96; only the {len(free)} coefficients are refit",
        free=free, theta=list(map(float, th)), theta_start=list(map(float, th0)), theta_shipped=list(map(float, th_ship)),
        source=str(SRC), r2="O2n", fit=dict(best_mean_fit=ef, minutes=(time.time() - t) / 60, budget_min=a.budget),
        stats={g: C.stats(v) for g, v in out.items()} | {"V300_shipped": C.stats(ship_v)},
        per_case={g: {C.kstr(k): float(e) for k, e in v.items()} for g, v in out.items()}))


if __name__ == "__main__":
    main()
