"""EXPLORATION pass 38 -- chi-odd merger shapes from the pass-35 residual PCA, on the pass-36 D1 model (18c, measured RD decay).

Pass 37 (knot ablation): at chi1 = 0 / +0.8 the per-case spline gain is mostly beta (1.42 -> 0.25, 4.56 -> 1.13 e-4).
Pass 35 PCA: dlnB comp 0 (knots -30..0 m1) has an amplitude 64% explained by 0.055 chi1; dlnA comp 1 (knots -50..-18)
66% explained by 0.072 chi1.  New terms (shapes = those PCA vectors, zero-padded to s = 60, q-flat like the other spin shapes):
    N1:  ln beta  += c_B chi1 phiB35(s)                                       (19c)
    N2:  N1 and ln alpha += c_A chi1 phiA35(s)                                 (20c)
Start c = 0 (= pass 36 D1, checked).  Usage: python p38_newshape.py N1|N2 [budget].  Writes exploration/p38/p38_<var>.json.
"""
from __future__ import annotations
import json, sys
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
VAR = sys.argv[1]; BUD = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
sys.argv = [sys.argv[0], "D1"]
import p36_rdslope as D                         # noqa: E402  (measured-decay get; P6.ab = p31 wrapper)
G = D.G; M = G.M; P6 = G.P6; A = G.A; PR = G.PR
OUT = HERE / "p38"; OUT.mkdir(exist_ok=True)
R = json.loads((HERE / "p36" / "p36_D1.json").read_text())
TH36 = np.array(R["theta"]); FREE36 = list(R["free"]); N36 = len(TH36)
PCA = json.loads((HERE / "p35" / "p35_floor.json").read_text())["pca"]


def _shape(field, j):
    kn = list(PCA[field]["knots"]); v = list(PCA[field]["comps"][j])
    for x in (35.0, 60.0):
        if x > kn[-1]:
            kn.append(x); v.append(0.0)
    sp = CubicSpline(kn, v, bc_type="natural"); lo, hi = kn[0], kn[-1]
    return lambda s: sp(np.clip(s, lo, hi))


PHB, PHA = _shape("dlnB", 0), _shape("dlnA", 1)
NEW = {"N1": 1, "N2": 2}[VAR]
base_ab = G.ab


def ab(th, k, key):
    th = np.asarray(th, float)
    a, b = base_ab(th[:N36], k, key)
    if a is None:
        return a, b
    c = P6.get(k); s = c["case"][("fit_" if key == "fit" else "") + "t_bhpt"] - c["tpk"]; chi = k[1]
    b = b * np.exp(th[N36] * chi * PHB(s))
    if NEW == 2:
        a = a * np.exp(th[N36 + 1] * chi * PHA(s))
    return a, b


def main():
    P6.get = D.get; P6.ab = ab                  # BEFORE the fork
    pool = mp.get_context("fork").Pool(96)
    x0 = np.r_[TH36, np.zeros(NEW)]; free = FREE36 + list(range(N36, N36 + NEW))
    chk = P6.T96[::8]
    e = np.array(pool.map(P6._full, [(list(x0), k) for k in chk]))
    r = e / np.array([R["T96_cases"][f"{q:.6f}_{c:+.4f}"] for q, c in chk])
    print(f"{VAR}: reproduce pass 36 D1 ratio {r.min():.4f}..{r.max():.4f}; {len(free)}c", flush=True)
    assert np.allclose(r, 1, atol=2e-3), r
    th, ef = A.refit(pool, x0, free, BUD, VAR)
    sc = PR.score(pool, th)
    out = dict(variant=VAR, n_coef=len(free), free=free, theta=list(map(float, th)), fit_mean=ef, **sc)
    (OUT / f"p38_{VAR}.json").write_text(json.dumps(out, indent=1))
    t, h = sc["T96"], sc["H270"]
    print(f"{VAR} {len(free)}c: new c = " + ", ".join(f"{v:+.4f}" for v in th[N36:]), flush=True)
    print(f"   T96 {t[0]:.4e} / {t[1]:.4e} / {t[2]:.4e} | H270 {h[0]:.4e} / {h[1]:.4e} / {h[2]:.4e}", flush=True)
    print("   rows " + " ".join(f"{v:.2e}" for v in sc["row_median"].values()), flush=True)
    print("   cols " + " ".join(f"{k} {v:.2e}" for k, v in sc["col_median"].items()), flush=True)
    for qq in ("2.5", "2.0"):
        v = [x for kk, x in sc["lowq"].items() if kk.startswith(qq)]
        print(f"   low q {qq}: max {max(v):.3e} median {np.median(v):.3e}", flush=True)


if __name__ == "__main__":
    main()
