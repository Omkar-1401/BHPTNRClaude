"""EXPLORATION pass 33 -- below 19 coefficients, and the remaining rise, on the pass-32 model.

Pass 32 (G2 pruned, 19c): spin terms x (nu5/nu) exactly (both gammas -> 1, p_sh -> 1 cost < 1%),
T96 2.02e-4 / 1.28e-4 / 9.31e-4.  Remaining rise: q = 8 row 1.91e-4 vs 0.98e-4 at q = 5, mostly at chi1 <= 0 and the
(8, +0.8) corner.  Variants (each a refit of the rest, then full scoring):
    A: drop r0 (single-drop cost 3.9%)                                   18c
    B: ISCO-crossing onset s1 = s_isco + t1a (t1b dropped; pass 26 I1)    18c
    C: free r2 (MR beta offset r0 + r2 nu; pass 30: r0 drifts +0.012 -> -0.018 from q = 5 to 8)   20c
Usage: python p33_prune19.py A|B|C [budget_min].  Writes exploration/p33/p33_<var>.json.
"""
from __future__ import annotations
import json, sys
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
VAR = sys.argv[1]; BUD = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
sys.argv = [sys.argv[0], "G2"]
import p31_spinq as G                           # noqa: E402  (P6.ab = gamma-wrapped p25 model)
M = G.M; P6 = G.P6; A = G.A; PR = G.PR
OUT = HERE / "p33"; OUT.mkdir(exist_ok=True)
R = json.loads((HERE / "p32" / "p32_G2.json").read_text()); NAMES = R["names"]; R = R["round2"]
TH0 = np.array(R["theta"]); FREE0 = list(R["free"])
I = {n: NAMES.index(n) for n in NAMES}
_SI = {}


def x_isco(chi):
    z1 = 1 + (1 - chi ** 2) ** (1 / 3) * ((1 + chi) ** (1 / 3) + (1 - chi) ** (1 / 3))
    z2 = np.sqrt(3 * chi ** 2 + z1 ** 2)
    r = 3 + z2 - np.sign(chi) * np.sqrt((3 - z1) * (3 + z1 + 2 * z2))
    return (r ** 1.5 + chi) ** (-2 / 3)


def s_isco(k):
    if k not in _SI:
        c = P6.get(k); t, x = c["case"]["t_bhpt"], c["case"]["x"]
        pre = t < c["tpk"]; i = int(np.argmax((x >= x_isco(k[1])) & pre))
        _SI[k] = float(t[i] - c["tpk"])
    return _SI[k]


def s1_I1(k, d):
    return s_isco(k) + d["t1a"]


def main():
    pool = mp.get_context("fork").Pool(96)
    chk = P6.T96[::8]
    e = np.array(pool.map(P6._full, [(list(TH0), k) for k in chk]))
    r = e / np.array([R["T96_cases"][f"{q:.6f}_{c:+.4f}"] for q, c in chk])
    print(f"{VAR}: reproduce pass 32 ratio {r.min():.4f}..{r.max():.4f}", flush=True)
    assert np.allclose(r, 1, atol=2e-3), r
    f0 = TH0.copy(); free = list(FREE0)
    if VAR == "A":
        f0[I["r0"]] = 0.0; free.remove(I["r0"])
    elif VAR == "C":
        free.append(I["r2"])
    elif VAR == "B":
        pool.close()
        with mp.get_context("fork").Pool(96) as p0:
            si = dict(zip(P6.T96, p0.map(s_isco, P6.T96)))
        _SI.update(si)
        f0[I["t1a"]] = float(np.median([TH0[I["t1a"]] + TH0[I["t1b"]] * k[1] - si[k] for k in P6.T96]))
        f0[I["t1b"]] = 0.0; free.remove(I["t1b"])
        M.S1FUN = s1_I1                     # BEFORE the fork
        pool = mp.get_context("fork").Pool(96)
    th, ef = A.refit(pool, f0, free, BUD, VAR)
    sc = PR.score(pool, th)
    out = dict(variant=VAR, n_coef=len(free), free=free, theta=list(map(float, th)), fit_mean=ef, **sc)
    (OUT / f"p33_{VAR}.json").write_text(json.dumps(out, indent=1))
    t, h = sc["T96"], sc["H270"]
    print(f"{VAR} {len(free)}c: T96 {t[0]:.4e} / {t[1]:.4e} / {t[2]:.4e} | H270 {h[0]:.4e} / {h[1]:.4e} / {h[2]:.4e}", flush=True)
    if VAR == "C":
        print(f"   r0 {th[I['r0']]:+.4f}  r2 {th[I['r2']]:+.4f}", flush=True)
    print("   rows " + " ".join(f"{v:.2e}" for v in sc["row_median"].values()), flush=True)
    print("   cols " + " ".join(f"{k} {v:.2e}" for k, v in sc["col_median"].items()), flush=True)
    for qq in ("2.5", "2.0"):
        v = [x for kk, x in sc["lowq"].items() if kk.startswith(qq)]
        print(f"   low q {qq}: max {max(v):.3e} median {np.median(v):.3e}", flush=True)


if __name__ == "__main__":
    main()
