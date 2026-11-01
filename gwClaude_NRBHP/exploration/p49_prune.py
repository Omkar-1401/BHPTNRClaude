"""EXPLORATION pass 49 -- lean models with 15, 14, 13, 12 fitted coefficients, for the Pareto front (user, 2026-10-26).

Base: pass 48 lean (16c; closed-form inputs; Ori–Thorne onset, T_start = -1).  Greedy backward elimination:
    at each size n, try dropping each of the NCAND cheapest remaining coefficients (ranking: pass 41 Hessian predicted
    drop cost, cheapest first), each with a short refit (SHORT min); drop the cheapest, refit longer (LONG min); score.
A dropped coefficient is set to 0 (every candidate is an additive term or a correction amplitude, 0 = term off).
Every size is written to p49/p49.json as soon as it is done; a rerun resumes from the last finished size.
    python p49_prune.py [target_n=12] [NCAND=5] [SHORT=2.5] [LONG=10] [base=p48lean]
base p40lean (pass 49b): the same pruning on pass 40 lean, i.e. with the qnm package, NRSur3dq8Remnant and the spline
shapes called directly instead of their closed forms; writes p49b/p49.json.
"""
from __future__ import annotations
import json, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ARG = sys.argv[1:]
TARGET = int(ARG[0]) if len(ARG) > 0 else 12; NCAND = int(ARG[1]) if len(ARG) > 1 else 5
SHORT = float(ARG[2]) if len(ARG) > 2 else 2.5; LONG = float(ARG[3]) if len(ARG) > 3 else 10.0
BASE = ARG[4] if len(ARG) > 4 else "p48lean"
import modelsetup                               # noqa: E402
_M = modelsetup.load(BASE)                      # p48lean: closed-form inputs + OT onset; p40lean: module calls + splines + OT onset
C = sys.modules["p40_r2closed"]                 # both chains share its refit / score machinery
P6 = _M["P6"]; A = C.A; PR = C.PR
OUT = HERE / {"p48lean": "p49", "p40lean": "p49b"}[BASE]; OUT.mkdir(exist_ok=True); RES = OUT / "p49.json"
NAMES = json.loads((HERE / "p32" / "p32_G2.json").read_text())["names"]
# pass-41 Hessian ranking (pass 40, cheapest first); t1a, t1b are not in the lean model
RANK = ["pb2", "eb1", "ea0", "pa0", "pb1", "pb0", "pa2", "flux_b", "flux_s2", "r3", "r1", "flux_b2", "l0", "flux_s3",
        "flux_P0", "flux_P1"]
T_START = time.time()


def fitmean(th):
    pool = mp.get_context("fork").Pool(96)
    v = float(np.mean(pool.map(P6._fit, [(list(th), k) for k in P6.T96]))); pool.close(); return v


def main():
    st = json.loads(RES.read_text()) if RES.exists() else None
    if st is None:
        R = _M["R"]; th = np.array(R["theta"], float); free = list(R["free"])
        e0 = fitmean(th)
        st = dict(base=_M["label"], steps=[dict(n=len(free), dropped=None, theta=list(map(float, th)), free=free, fit_mean=e0)])
        print(f"base 16c: fit-mean {e0:.4e}", flush=True)
    while len(st["steps"][-1]["free"]) > TARGET:
        last = st["steps"][-1]; th0 = np.array(last["theta"]); free = list(last["free"]); e_ref = last["fit_mean"]
        names_free = [NAMES[i] for i in free]
        cands = [n for n in RANK if n in names_free][:NCAND]
        trials = {}
        for n in cands:
            i = NAMES.index(n); f0 = th0.copy(); f0[i] = 0.0
            pool = mp.get_context("fork").Pool(96)
            _, e = A.refit(pool, f0, [j for j in free if j != i], SHORT, f"drop_{n}"); pool.close()
            trials[n] = dict(fit_mean=e, cost=e / e_ref - 1)
            print(f"   n={len(free)}: drop {n:8s} cost {100 * (e / e_ref - 1):+6.2f} %  ({(time.time() - T_START) / 60:.0f} min)", flush=True)
        best = min(trials, key=lambda n: trials[n]["fit_mean"])
        i = NAMES.index(best); f0 = th0.copy(); f0[i] = 0.0; free2 = [j for j in free if j != i]
        pool = mp.get_context("fork").Pool(96)
        th, e = A.refit(pool, f0, free2, LONG, f"n{len(free2)}")
        sc = PR.score(pool, th); pool.close()
        st["steps"].append(dict(n=len(free2), dropped=best, trials=trials, theta=list(map(float, th)), free=free2, fit_mean=e, **sc))
        RES.write_text(json.dumps(st, indent=1))
        t = sc["T96"]; h = sc["H270"]; lq = [v for k, v in sc["lowq"].items() if k.startswith("2.5")]
        print(f"n={len(free2)} (dropped {best}): T96 {t[1]:.4e} / {t[0]:.4e} / {t[2]:.3e} | H270 median {h[1]:.4e} | "
              f"q2.5 max {max(lq):.3e}  ({(time.time() - T_START) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
