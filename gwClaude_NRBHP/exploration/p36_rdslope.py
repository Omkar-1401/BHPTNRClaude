"""EXPLORATION pass 36 -- measured ppBHPT ringdown decay in kappa_RD (0 coefficients), on the 18c model (pass 33 A).

Pass 35 (per-case spline floor on 18c): the leading residual at the spin extremes is a q-FLAT ringdown slope in ln alpha,
d(dlnA)/ds over s = 18..35 m1 ~ -0.026 / -0.004 / +0.013 per m1 at chi1 = -0.6 / 0 / +0.8 (std over q <= 0.04 at s = 35).
q-flat in m1 = a ppBHPT-side property: the Kerr gamma_pp(chi1) = w220 / 2 Q220 in kappa_RD = gamma_pp / beta_r - gamma_NR is
not the decay rate the ppBHPT (2,2) actually has over the scored ringdown (overtones; mirror modes when retrograde).
Here gamma_pp is replaced by the measured slope of -ln|h_pp| over s in [S_RING, SE] m1 (per case, from the ppBHPT input):
    kappa_RD -> kappa_RD + (gamma_meas - gamma_pp) / beta_r
Variants: D1 SE = 40, D2 SE = 60.  The coefficients (18) are refit.  Falsified if the -0.6 / +0.8 columns do not drop.
Usage: python p36_rdslope.py D1|D2 [budget_min].  Writes exploration/p36/p36_<var>.json.
"""
from __future__ import annotations
import json, sys
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
VAR = sys.argv[1]; BUD = float(sys.argv[2]) if len(sys.argv) > 2 else 35.0
SE = {"D1": 40.0, "D2": 60.0}[VAR]
sys.argv = [sys.argv[0], "G2"]
import p31_spinq as G                           # noqa: E402
M = G.M; P6 = G.P6; A = G.A; PR = G.PR
import physics_common as PC                     # noqa: E402
OUT = HERE / "p36"; OUT.mkdir(exist_ok=True)
R = json.loads((HERE / "p33" / "p33_A.json").read_text())
TH0 = np.array(R["theta"]); FREE0 = list(R["free"])
_get0 = P6.get


def get(k):
    c = _get0(k)
    if "g_meas" not in c:
        t, h = c["case"]["t_bhpt"], c["case"]["h_bhpt"]; s = t - c["tpk"]
        m = (s >= P6.S_RING) & (s <= SE)
        g = -float(np.polyfit(s[m], np.log(np.abs(h[m])), 1)[0])
        gpp = PC.w220_exact(k[1]) / (2 * PC.Q220_exact(k[1]))
        br, kap = c["anc"]
        c["g_meas"], c["g_pp"] = g, gpp
        c["anc"] = (br, kap + (g - gpp) / br)
    return c


def _g(k):
    c = get(k); return c["g_meas"], c["g_pp"]


def main():
    assert P6.ab is G.ab
    pool0 = mp.get_context("fork").Pool(96)
    chk = P6.T96[::8]
    e = np.array(pool0.map(P6._full, [(list(TH0), k) for k in chk]))
    r = e / np.array([R["T96_cases"][f"{q:.6f}_{c:+.4f}"] for q, c in chk])
    print(f"{VAR}: reproduce pass 33 A ratio {r.min():.4f}..{r.max():.4f}", flush=True)
    assert np.allclose(r, 1, atol=2e-3), r
    pool0.close()
    P6.get = get                                # BEFORE the fork
    pool = mp.get_context("fork").Pool(96)
    gm = dict(zip(P6.T96, pool.map(_g, P6.T96)))
    for ch in sorted({k[1] for k in P6.T96}):
        v = np.array([gm[k][0] for k in P6.T96 if k[1] == ch])
        print(f"   chi1 {ch:+.1f}: gamma_pp {gm[[k for k in P6.T96 if k[1] == ch][0]][1]:.4f}, measured {v.mean():.4f} +- {v.std():.4f} per m1", flush=True)
    e0 = float(np.mean(pool.map(P6._fit, [(list(TH0), k) for k in P6.T96])))
    print(f"   start fit-mean {e0:.4e} (pass 33 A {R['fit_mean']:.4e})", flush=True)
    th, ef = A.refit(pool, TH0, FREE0, BUD, VAR)
    sc = PR.score(pool, th)
    out = dict(variant=VAR, SE=SE, n_coef=len(FREE0), free=FREE0, theta=list(map(float, th)), fit_mean=ef,
               g_meas={f"{q:.6f}_{c:+.4f}": v[0] for (q, c), v in gm.items()}, **sc)
    (OUT / f"p36_{VAR}.json").write_text(json.dumps(out, indent=1))
    t, h = sc["T96"], sc["H270"]
    print(f"{VAR} {len(FREE0)}c: T96 {t[0]:.4e} / {t[1]:.4e} / {t[2]:.4e} | H270 {h[0]:.4e} / {h[1]:.4e} / {h[2]:.4e}", flush=True)
    print("   rows " + " ".join(f"{v:.2e}" for v in sc["row_median"].values()), flush=True)
    print("   cols " + " ".join(f"{k} {v:.2e}" for k, v in sc["col_median"].items()), flush=True)
    for qq in ("2.5", "2.0"):
        v = [x for kk, x in sc["lowq"].items() if kk.startswith(qq)]
        print(f"   low q {qq}: max {max(v):.3e} median {np.median(v):.3e}", flush=True)


if __name__ == "__main__":
    main()
