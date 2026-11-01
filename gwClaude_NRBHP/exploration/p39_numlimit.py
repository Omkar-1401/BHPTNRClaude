"""EXPLORATION pass 39 -- restore the test-mass limit (gate P: alpha, beta -> 1 as nu -> 0) on the pass-36 D1 leader.

Found while writing scaling_p36.md (2026-10-15): since pass 32 several terms are q-flat and survive nu -> 0:
the nu5/nu spin terms (s2, s3, r1, r3, pb1, pb2, eb1) AND pb0 (it sits in the same group), and pass 36's measured-decay
shift of kappa_RD, which tends to gamma_meas - gamma_Kerr(chi1) != 0.  Applied to q = 8 ppBHPT waveforms, the limiting map
is off the identity by mathcalE 2.3e-4 (chi1 = 0) to 1.1e-2 (chi1 = +0.8).
Fix (0 new fitted coefficients): multiply every one of those terms by
    g(nu) = 1 - exp(-(nu/nu_c)^2)
which is 1 to < 0.2% in the training range (q <= 8, nu >= 0.0988) for nu_c <= 0.04 and goes as (nu/nu_c)^2 as nu -> 0, so
every surviving term vanishes (s3 (nu5/nu) chi1 E -> 0 because E ~ nu and g ~ nu^2).  nu_c is NOT constrained by
q in [3, 8]: where the turnover happens is an assumption, so two values are run to show what it costs in range.
    L4: nu_c = 0.04 (turnover near q ~ 23)      L6: nu_c = 0.06 (turnover near q ~ 15; g(q = 8) = 0.934)
The 18 coefficients are refit on T96 from the D1 solution.  nu_c = 0 must reproduce D1 exactly (checked first).
Usage: python p39_numlimit.py L4|L6 [budget_min].  Writes exploration/p39/p39_<var>.json.
"""
from __future__ import annotations
import json, sys
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
VAR = sys.argv[1]; BUD = float(sys.argv[2]) if len(sys.argv) > 2 else 38.0
NUC = {"L4": 0.04, "L6": 0.06, "L0": 0.0}[VAR]
sys.argv = [sys.argv[0], "D1"]
import p36_rdslope as D                         # noqa: E402  (G = p31 wrapper, measured-decay machinery)
G = D.G; M = G.M; P6 = G.P6; A = G.A; PR = G.PR; PC = D.PC
OUT = HERE / "p39"; OUT.mkdir(exist_ok=True)
R = json.loads((HERE / "p36" / "p36_D1.json").read_text())
TH0 = np.array(R["theta"]); FREE0 = list(R["free"]); N = G.N
NU5 = 5.0 / 36.0
_NC = {"v": NUC}


def gfac(nu):
    nc = _NC["v"]
    return 1.0 if nc <= 0 else 1.0 - np.exp(-(nu / nc) ** 2)


def ab(th, k, key):
    th = np.array(th, float); q = k[0]; nu = q / (1 + q) ** 2
    t = th[:N].copy()
    for gi, idx in enumerate(G.GIDX):
        t[idx] *= (nu / NU5) ** (-th[N + gi]) * gfac(nu)
    return G.base_ab(t, k, key)


def get(k):
    c = D._get0(k)
    if "g_meas" not in c:
        t, h = c["case"]["t_bhpt"], c["case"]["h_bhpt"]; s = t - c["tpk"]
        m = (s >= P6.S_RING) & (s <= D.SE)
        g = -float(np.polyfit(s[m], np.log(np.abs(h[m])), 1)[0])
        gpp = PC.w220_exact(k[1]) / (2 * PC.Q220_exact(k[1]))
        br, kap = c["anc"]; nu = k[0] / (1 + k[0]) ** 2
        c["g_meas"], c["g_pp"] = g, gpp
        c["anc"] = (br, kap + gfac(nu) * (g - gpp) / br)
    return c


def main():
    assert D.SE == 40.0
    P6.get = get; P6.ab = ab                    # BEFORE the fork
    chk = P6.T96[::8]
    _NC["v"] = 0.0                              # nu_c = 0: g = 1, must be D1 exactly
    pool = mp.get_context("fork").Pool(96)
    e = np.array(pool.map(P6._full, [(list(TH0), k) for k in chk]))
    pool.close()
    r = e / np.array([R["T96_cases"][f"{q:.6f}_{c:+.4f}"] for q, c in chk])
    print(f"{VAR}: g = 1 reproduces pass 36 D1, ratio {r.min():.4f}..{r.max():.4f}", flush=True)
    assert np.allclose(r, 1, atol=2e-3), r
    _NC["v"] = NUC
    qs = (3.0, 5.0, 8.0, 10.0, 15.0, 30.0, 100.0)
    print(f"   nu_c = {NUC}: g(q) " + ", ".join(f"q={q:g} {gfac(q / (1 + q) ** 2):.4f}" for q in qs), flush=True)
    pool = mp.get_context("fork").Pool(96)      # fresh fork: workers see nu_c and empty caches
    e0 = float(np.mean(pool.map(P6._fit, [(list(TH0), k) for k in P6.T96])))
    print(f"   D1 coefficients with g: fit-mean {e0:.4e} (D1 {R['fit_mean']:.4e})", flush=True)
    th, ef = A.refit(pool, TH0, FREE0, BUD, VAR)
    sc = PR.score(pool, th)
    out = dict(variant=VAR, nu_c=NUC, n_coef=len(FREE0), free=FREE0, theta=list(map(float, th)), fit_mean=ef,
               start_fit_mean=e0, g_at_q={f"{q:g}": float(gfac(q / (1 + q) ** 2)) for q in qs}, **sc)
    (OUT / f"p39_{VAR}.json").write_text(json.dumps(out, indent=1))
    t, h = sc["T96"], sc["H270"]
    print(f"{VAR} {len(FREE0)}c: T96 {t[0]:.4e} / {t[1]:.4e} / {t[2]:.4e} | H270 {h[0]:.4e} / {h[1]:.4e} / {h[2]:.4e}", flush=True)
    print("   rows " + " ".join(f"{v:.2e}" for v in sc["row_median"].values()), flush=True)
    print("   cols " + " ".join(f"{k} {v:.2e}" for k, v in sc["col_median"].items()), flush=True)
    for qq in ("2.5", "2.0"):
        v = [x for kk, x in sc["lowq"].items() if kk.startswith(qq)]
        print(f"   low q {qq}: max {max(v):.3e} median {np.median(v):.3e}", flush=True)


if __name__ == "__main__":
    main()
