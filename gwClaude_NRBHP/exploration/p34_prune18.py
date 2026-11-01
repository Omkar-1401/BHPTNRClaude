"""EXPLORATION pass 34 -- single-drop costs and joint prune of the pass-33 A model (18c).

Pass 31: G1 (21c, gamma 1.09) T96 2.11e-4 / 1.37e-4 / 9.96e-4, G2 (22c, gammas 1.15 / 1.28) 2.01e-4 / 1.27e-4 / 9.60e-4,
against pass 25 (20c) 2.35e-4 / 1.61e-4 / 1.18e-3.  Same recipe as pass 25: single-drop costs (drop = 0; p_sh and the
gammas drop to 1, i.e. integer exponents), then a joint drop of everything with cost < CUT, refit, full scoring.
Usage: python p32_prune31.py G1|G2.  Writes exploration/p32/p32_<var>.json (rewritten after each step).
"""
from __future__ import annotations
import json, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

T_START = time.time()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
VAR = "G2"
sys.argv = [sys.argv[0], VAR]
import p31_spinq as G                           # noqa: E402  (P6.ab = gamma-wrapped p25 model)
M = G.M; P6 = G.P6; A = G.A; PR = G.PR; N = G.N
OUT = HERE / "p34"; OUT.mkdir(exist_ok=True)
R = json.loads((HERE / "p33" / "p33_A.json").read_text())
TH0 = np.array(R["theta"]); FREE0 = list(R["free"])
NAMES = M.NAMES + [f"gamma{g}" for g in range(len(G.GIDX))]
ONE = {M.NAMES.index("p_sh")} | set(range(N, len(TH0)))
CUT = 0.03; STEP_MIN = 2.2


def save(out):
    (OUT / f"p34_{VAR}.json").write_text(json.dumps(out, indent=1))


def main():
    pool = mp.get_context("fork").Pool(96)
    chk = P6.T96[::8]
    e = np.array(pool.map(P6._full, [(list(TH0), k) for k in chk]))
    r = e / np.array([R["T96_cases"][f"{q:.6f}_{c:+.4f}"] for q, c in chk])
    print(f"{VAR}: reproduce pass 33 A ratio {r.min():.4f}..{r.max():.4f}; {len(FREE0)}c", flush=True)
    assert np.allclose(r, 1, atol=2e-3), r
    e1 = R["fit_mean"]
    out = dict(variant=VAR, names=NAMES, CUT=CUT, base_fit_mean=e1, round1={})
    for i in FREE0:
        f0 = TH0.copy(); f0[i] = 1.0 if i in ONE else 0.0
        _, e = A.refit(pool, f0, [j for j in FREE0 if j != i], STEP_MIN, NAMES[i])
        out["round1"][NAMES[i]] = dict(fit_mean=e, cost=e / e1 - 1)
        print(f"{NAMES[i]:10s} cost {100 * (e / e1 - 1):+7.2f} %", flush=True); save(out)
    zero = sorted(n for n, rr in out["round1"].items() if rr["cost"] < CUT)
    f0 = TH0.copy()
    for n in zero:
        i = NAMES.index(n); f0[i] = 1.0 if i in ONE else 0.0
    free = [j for j in FREE0 if NAMES[j] not in zero]
    left = 113 - (time.time() - T_START) / 60 - 8
    th2, e2 = A.refit(pool, f0, free, max(left, 6), "joint")
    out["round2"] = dict(zeroed=zero, n_coef=len(free), free=free, theta=list(map(float, th2)), fit_mean=e2, **PR.score(pool, th2))
    save(out)
    t, h = out["round2"]["T96"], out["round2"]["H270"]
    print(f"pruned {len(free)}c (zeroed {zero}): T96 {t[0]:.4e} / {t[1]:.4e} / {t[2]:.4e} | H270 {h[0]:.4e} / {h[1]:.4e} / {h[2]:.4e}", flush=True)
    print("   gamma " + ", ".join(f"{v:+.3f}" for v in th2[N:]), flush=True)
    print("   rows " + " ".join(f"{v:.2e}" for v in out["round2"]["row_median"].values()), flush=True)
    print("   cols " + " ".join(f"{k} {v:.2e}" for k, v in out["round2"]["col_median"].items()), flush=True)
    for qq in ("2.5", "2.0"):
        v = [x for kk, x in out["round2"]["lowq"].items() if kk.startswith(qq)]
        print(f"   low q {qq}: max {max(v):.3e} median {np.median(v):.3e}", flush=True)


if __name__ == "__main__":
    main()
