"""EXPLORATION pass 28 -- spin-dependent nu exponent of the shape terms, on the pass-25 model (20c).

Pass 27 (per-case floor on the pass-25 model): the second residual component of both fields has a q-trend whose
steepness depends on spin (steep at chi1 = -0.6 and +0.8, mild in the middle), and chi x (1, nu) explains 76-87% of it.
Cheapest structured version: let the shape weight's nu exponent depend on chi1 (and on the field):
    X2:  ws_beta = ws_alpha = (nu/nu5)^(p_sh + g2 chi1^2)                                   (+1c, 21c)
    X3:  ws_beta = (nu/nu5)^(p_sh + g2 chi1^2),  ws_alpha = (nu/nu5)^(pA + gA2 chi1^2)       (+3c, 23c)
Start = pass-25 round 2 with g2 = gA2 = 0, pA = p_sh (reproduces it; checked).  Pruning comes after, if it pays.
Usage: python p28_spinexp.py X2|X3 [budget_min].  Writes exploration/p28/p28_<var>.json.
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
OUT = HERE / "p28"; OUT.mkdir(exist_ok=True)
R = json.loads((HERE / "p25" / "p25_nuexp.json").read_text())["round2"]
TH25 = np.array(R["theta"]); FREE25 = list(R["free"]); N = len(TH25)
IPS = M.NAMES.index("p_sh")


def ws_X2(th, nu, chi1):
    w = (nu / M.NU5) ** (th[IPS] + th[N] * chi1 ** 2)
    return w, w


def ws_X3(th, nu, chi1):
    return (nu / M.NU5) ** (th[IPS] + th[N] * chi1 ** 2), (nu / M.NU5) ** (th[N + 1] + th[N + 2] * chi1 ** 2)


def main():
    fn, extra, names = {"X2": (ws_X2, [0.0], ["g2"]),
                        "X3": (ws_X3, [0.0, TH25[IPS], 0.0], ["g2", "pA", "gA2"])}[VAR]
    M.WSFUN = fn                                # before the fork
    pool = mp.get_context("fork").Pool(96)
    x0 = np.r_[TH25, extra]
    free = FREE25 + list(range(N, len(x0)))
    chk = P6.T96[::8]
    e = np.array(pool.map(P6._full, [(list(x0), k) for k in chk]))
    r = e / np.array([R["T96_cases"][f"{q:.6f}_{c:+.4f}"] for q, c in chk])
    print(f"{VAR}: reproduce pass 25 ratio {r.min():.4f}..{r.max():.4f}; {len(free)}c", flush=True)
    assert np.allclose(r, 1, atol=2e-3), r
    th, ef = A.refit(pool, x0, free, BUD, VAR)
    sc = PR.score(pool, th)
    out = dict(variant=VAR, extra_names=names, n_coef=len(free), free=free, theta=list(map(float, th)), fit_mean=ef, **sc)
    (OUT / f"p28_{VAR}.json").write_text(json.dumps(out, indent=1))
    t, h = sc["T96"], sc["H270"]
    print(f"{VAR} {len(free)}c: " + ", ".join(f"{n} = {v:+.3f}" for n, v in zip(names, th[N:])) + f", p_sh = {th[IPS]:+.3f}", flush=True)
    print(f"   T96 {t[0]:.4e} / {t[1]:.4e} / {t[2]:.4e} | H270 {h[0]:.4e} / {h[1]:.4e} / {h[2]:.4e}", flush=True)
    print("   rows " + " ".join(f"{v:.2e}" for v in sc["row_median"].values()), flush=True)
    print("   cols " + " ".join(f"{k} {v:.2e}" for k, v in sc["col_median"].items()), flush=True)
    for qq in ("2.5", "2.0"):
        v = [x for kk, x in sc["lowq"].items() if kk.startswith(qq)]
        print(f"   low q {qq}: max {max(v):.3e} median {np.median(v):.3e}", flush=True)


if __name__ == "__main__":
    main()
