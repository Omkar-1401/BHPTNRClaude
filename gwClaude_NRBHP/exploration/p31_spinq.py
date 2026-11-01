"""EXPLORATION pass 31 -- q-dependence of the spin terms (from the pass-30 per-row refit).

Pass 30: refitted per q row, the pass-25 equations reach 1.70 / 0.97 / 1.19 e-4 (row medians at q = 3 / 5 / 8) against
3.03 / 1.18 / 2.63 e-4 globally, and the coefficients that drift with q are mostly the spin terms (s3 1.96 -> 3.44,
r1 0.055 -> 0.132, eb1 0.067 -> 0.151, pb1 -0.23 -> -0.37, r3 -0.09 -> -0.15; roughly ~ (nu5/nu)^1 extra).
Here each listed coefficient is multiplied by F = (nu/nu5)^(-gamma) (= 1 at q = 5; gamma = 0 is pass 25 exactly):
    G1: one gamma on {s2, s3, r1, r3, pb1, eb1}                                          (21c)
    G2: gamma_I on the inspiral spin terms {s2, s3}; gamma_M on {r1, r3, pb0, pb1, pb2, eb1} (22c)
Usage: python p31_spinq.py G1|G2 [budget_min].  Writes exploration/p31/p31_<var>.json.
"""
from __future__ import annotations
import json, sys
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
VAR = sys.argv[1]; BUD = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
sys.argv = [sys.argv[0], "35"]
import p25_nuexp as M                           # noqa: E402
P6 = M.P6; A = M.A; PR = M.PR
OUT = HERE / "p31"; OUT.mkdir(exist_ok=True)
R = json.loads((HERE / "p25" / "p25_nuexp.json").read_text())["round2"]
TH25 = np.array(R["theta"]); FREE25 = list(R["free"]); N = len(TH25)
IX = {n: M.NAMES.index(n) for n in M.NAMES}
GROUPS = {"G1": [["flux_s2", "flux_s3", "r1", "r3", "pb1", "eb1"]],
          "G2": [["flux_s2", "flux_s3"], ["r1", "r3", "pb0", "pb1", "pb2", "eb1"]]}[VAR]
GIDX = [[IX[n] for n in g] for g in GROUPS]
base_ab = M.ab


def ab(th, k, key):
    th = np.array(th, float); q = k[0]; nu = q / (1 + q) ** 2
    t = th[:N].copy()
    for g, idx in enumerate(GIDX):
        t[idx] *= (nu / M.NU5) ** (-th[N + g])
    return base_ab(t, k, key)


P6.ab = ab


def main():
    pool = mp.get_context("fork").Pool(96)
    x0 = np.r_[TH25, np.zeros(len(GIDX))]
    free = FREE25 + list(range(N, len(x0)))
    chk = P6.T96[::8]
    e = np.array(pool.map(P6._full, [(list(x0), k) for k in chk]))
    r = e / np.array([R["T96_cases"][f"{q:.6f}_{c:+.4f}"] for q, c in chk])
    print(f"{VAR}: reproduce pass 25 ratio {r.min():.4f}..{r.max():.4f}; {len(free)}c", flush=True)
    assert np.allclose(r, 1, atol=2e-3), r
    th, ef = A.refit(pool, x0, free, BUD, VAR)
    sc = PR.score(pool, th)
    out = dict(variant=VAR, groups=GROUPS, n_coef=len(free), free=free, theta=list(map(float, th)), fit_mean=ef, **sc)
    (OUT / f"p31_{VAR}.json").write_text(json.dumps(out, indent=1))
    t, h = sc["T96"], sc["H270"]
    print(f"{VAR} {len(free)}c: gamma = " + ", ".join(f"{v:+.3f}" for v in th[N:]), flush=True)
    print(f"   T96 {t[0]:.4e} / {t[1]:.4e} / {t[2]:.4e} | H270 {h[0]:.4e} / {h[1]:.4e} / {h[2]:.4e}", flush=True)
    print("   rows " + " ".join(f"{v:.2e}" for v in sc["row_median"].values()), flush=True)
    print("   cols " + " ".join(f"{k} {v:.2e}" for k, v in sc["col_median"].items()), flush=True)
    for qq in ("2.5", "2.0"):
        v = [x for kk, x in sc["lowq"].items() if kk.startswith(qq)]
        print(f"   low q {qq}: max {max(v):.3e} median {np.median(v):.3e}", flush=True)


if __name__ == "__main__":
    main()
