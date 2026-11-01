"""EXPLORATION pass 30 -- per-q-row refit of the pass-25 model (diagnostic for the rise past q = 5).

For each of the 12 T96 q rows separately, refit the pass-25 (20c) model on that row's 8 spins only.  Coefficients that
only set the nu-dependence are degenerate inside a row and stay at their global values: flux_b2, flux_P1, p_sh.
The other 17 are free.  Output: per-row error (is the row floor flat in q?) and per-row coefficients (which ones drift
with q, and how -- the nu-form the global model is missing).
The 12 rows run concurrently, 8 workers each (one Process per row).
Usage: python p30_perrow.py [budget_min].  Writes exploration/p30/p30_perrow.json.
"""
from __future__ import annotations
import json, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
BUD = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
sys.argv = [sys.argv[0], "35"]
import p25_nuexp as M                           # noqa: E402
P6 = M.P6; S = M.S
OUT = HERE / "p30"; OUT.mkdir(exist_ok=True)
R = json.loads((HERE / "p25" / "p25_nuexp.json").read_text())["round2"]
TH0 = np.array(R["theta"])
FIXED = {"flux_b2", "flux_P1", "p_sh"}
FREE = [j for j in R["free"] if M.NAMES[j] not in FIXED]


class Stop(Exception):
    pass


def row_fit(q, outq):
    keys = [(q, c) for c in S.CHIS]
    pool = mp.get_context("fork").Pool(8)
    st = dict(n=0, best=np.inf, bx=TH0[FREE].copy(), t=time.time())

    def expand(x):
        f = TH0.copy(); f[FREE] = x; return f

    def obj(x):
        v = float(np.mean(pool.map(P6._fit, [(list(expand(x)), k) for k in keys])))
        st["n"] += 1
        if v < st["best"]:
            st["best"], st["bx"] = v, np.array(x, float)
        if time.time() - st["t"] > BUD * 60:
            raise Stop
        return v
    e_glob = np.array(pool.map(P6._full, [(list(TH0), k) for k in keys]))
    try:
        minimize(obj, TH0[FREE], method="Powell", options={"xtol": 1e-4, "ftol": 1e-7, "maxiter": 60})
    except Stop:
        pass
    th = expand(st["bx"])
    e_row = np.array(pool.map(P6._full, [(list(th), k) for k in keys]))
    pool.close()
    outq.put(dict(q=q, evals=st["n"], theta=list(map(float, th)), E_global=list(map(float, e_glob)),
                  E_row=list(map(float, e_row))))


def main():
    outq = mp.get_context("fork").Queue()
    procs = [mp.get_context("fork").Process(target=row_fit, args=(q, outq)) for q in S.TRAIN_Q]
    for p in procs:
        p.start()
    rows = [outq.get() for _ in procs]
    for p in procs:
        p.join()
    rows.sort(key=lambda r: r["q"])
    (OUT / "p30_perrow.json").write_text(json.dumps(dict(names=M.NAMES, free=FREE, fixed=sorted(FIXED), rows=rows), indent=1))
    print("q      evals  global median   per-row median   per-row max", flush=True)
    for r in rows:
        print(f"{r['q']:5.2f} {r['evals']:6d}   {np.median(r['E_global']):.3e}      {np.median(r['E_row']):.3e}     {max(r['E_row']):.3e}", flush=True)
    print("per-row coefficients (global first):", flush=True)
    for j in FREE:
        print(f"  {M.NAMES[j]:8s} {TH0[j]:+9.4f} | " + " ".join(f"{r['theta'][j]:+8.3f}" for r in rows), flush=True)


if __name__ == "__main__":
    main()
