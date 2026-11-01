"""EXPLORATION pass 24 -- does the fit objective cause the U failure?  Refit the 20c model (pass 23 Z: V7 with the S3 1PA
alpha amplitude) with a power-mean objective  (mean E^p)^(1/p)  over T96 instead of the mean (p = 1), so the edges
(chi1 = +0.8, -0.6; q = 3, 8) weigh more.  No new coefficients.
Usage: python p24_objective.py P [budget]   (e.g. 2 or 3).  Writes exploration/p24/p24_p<P>.json.
"""
from __future__ import annotations
import json, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PW = float(sys.argv[1]); BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
sys.argv = [sys.argv[0], "Z"]
import p23_s3 as Z                              # noqa: E402  (P6.ab = the 20c model)
P6 = Z.P6; PR = Z.PR
OUT = HERE / "p24"; OUT.mkdir(exist_ok=True)
B = json.loads((HERE / "p23" / "p23_Z.json").read_text())
TH0 = np.array(B["theta"]); FREE = list(B["free"])


class Stop(Exception):
    pass


def refit(pool):
    st = dict(n=0, best=np.inf, bx=TH0[FREE].copy(), t=time.time())

    def expand(x):
        f = TH0.copy(); f[FREE] = x; return f

    def obj(x):
        e = np.array(pool.map(P6._fit, [(list(expand(x)), k) for k in P6.T96]))
        v = float(np.mean(e ** PW) ** (1 / PW))
        st["n"] += 1
        if v < st["best"]:
            st["best"], st["bx"] = v, np.array(x, float)
        if time.time() - st["t"] > BUDGET * 60:
            raise Stop
        return v
    try:
        minimize(obj, TH0[FREE], method="Powell", options={"xtol": 1e-4, "ftol": 1e-6, "maxiter": 40})
    except Stop:
        pass
    print(f"  p = {PW}: {st['n']} evals, objective {st['best']:.4e}", flush=True)
    return expand(st["bx"])


def main():
    pool = mp.get_context("fork").Pool(96)
    th = refit(pool)
    sc = PR.score(pool, th)
    out = dict(power=PW, n_coef=len(FREE), free=FREE, theta=list(map(float, th)), **sc)
    (OUT / f"p24_p{PW:g}.json").write_text(json.dumps(out, indent=1))
    t, h = sc["T96"], sc["H270"]
    print(f"p = {PW:g}, {len(FREE)}c: T96 {t[0]:.4e} / {t[1]:.4e} / {t[2]:.4e} | H270 {h[0]:.4e} / {h[1]:.4e} / {h[2]:.4e}", flush=True)
    print("   column medians: " + " ".join(f"{k} {v:.2e}" for k, v in sc["col_median"].items()), flush=True)
    print("   row medians: " + " ".join(f"{v:.2e}" for v in sc["row_median"].values()), flush=True)
    for qq in ("2.5", "2.0"):
        v = [x for k, x in sc["lowq"].items() if k.startswith(qq)]
        print(f"   low q {qq}: max {max(v):.3e} median {np.median(v):.3e}", flush=True)


if __name__ == "__main__":
    main()
