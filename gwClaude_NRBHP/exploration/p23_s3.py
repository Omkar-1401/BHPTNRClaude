"""EXPLORATION pass 23 (S3) -- 1PA amplitude in the inspiral alpha of the pass-22 pruned leader (21c).

The pruned leader's inspiral alpha is a constant, X1^(6/5)(1 + c0 nu) (flux coupling A0 dropped in pass 22).  Replace
(1 + c0 nu) by the second-order self-force (2,2) amplitude ratio, Schwarzschild, from WaSABI (wardell_data/R2_22_from_amplitudes.json;
R2 = 2 Re[h2eff/h0] is the |h|^2 ratio, so the amplitude factor is 1 + nu R2/2; R2/x -> 55/21 at small x):
    ln alpha += (1 - W) [ln(1 + k nu R2(x)/2) - ln(1 + c0 nu)],   c0 -> 0
x = x_scale(q) min(x_pp, XCLIP) (binary x), R2 held constant outside its table [0.04, 0.16].
  Z: k = 1 fixed (20c, zero-parameter alpha inspiral beyond X1^(6/5));   K: k free (21c; k ~ 1 checks the theory scale).
Usage: python p23_s3.py Z|K [budget].  Writes exploration/p23/p23_<var>.json.
"""
from __future__ import annotations
import json, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; REPO = HERE.parent
sys.path.insert(0, str(HERE))
VARIANT = sys.argv[1]; BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
sys.argv = [sys.argv[0], "V6", "p15", "1", "1"]
import p18_shapes2 as L                         # noqa: E402
import p22_prune as PR                          # noqa: E402  (names, scoring)
import p3_rd as P3                              # noqa: E402
import fit_pn_anchored_spin as S                # noqa: E402
P6 = L.P6; A = L.A
OUT = HERE / "p23"; OUT.mkdir(exist_ok=True)
R2D = json.loads((REPO / "wardell_data" / "R2_22_from_amplitudes.json").read_text())
_RX, _RY = np.array(R2D["x"]), np.array(R2D["R2_22"])
B = json.loads((HERE / "p22" / "p22_prune.json").read_text())["round2"]
TH0 = np.array(B["theta"]); FREE0 = list(B["free"]); NL = len(TH0)
IC0 = PR.NAMES.index("flux_c0")
_X = {}


def ab(th, k, key):
    a, b = L.ab(th[:NL], k, key)
    if a is None:
        return a, b
    q, chi1 = k; c = P6.get(k); case = c["case"]; pre = "fit_" if key == "fit" else ""
    if (k, key) not in _X:
        if len(_X) > 400:
            _X.clear()
        _X[(k, key)] = np.interp(S.PC.x_scale(q) * np.minimum(case[pre + "x"], S.XCLIP), _RX, _RY)
    r2 = _X[(k, key)]
    d = dict(zip(A.NAMES, th[:len(A.NAMES)]))
    s = case[pre + "t_bhpt"] - c["tpk"]
    s1 = d["t1a"] + d["t1b"] * chi1; s2 = d["t2a"] + d["t2b"] * chi1
    w = 1 - P3.smooth5((s - s1) / (s2 - s1))
    nu = q / (1 + q) ** 2; kk = th[NL] if len(th) > NL else 1.0
    return a * np.exp(w * (np.log1p(kk * nu * r2 / 2) - np.log1p(d["flux_c0"] * nu))), b


P6.ab = ab


def main():
    pool = mp.get_context("fork").Pool(96)
    x0 = TH0.copy(); x0[IC0] = 0.0
    free = [j for j in FREE0 if j != IC0]
    if VARIANT == "K":
        x0 = np.r_[x0, 1.0]; free.append(NL)
    print(f"S3 variant {VARIANT}: {len(free)}c", flush=True)
    th, e = A.refit(pool, x0, free, BUDGET, f"S3{VARIANT}")
    sc = PR.score(pool, th)
    out = dict(variant=VARIANT, n_coef=len(free), free=free, theta=list(map(float, th)), **sc)
    (OUT / f"p23_{VARIANT}.json").write_text(json.dumps(out, indent=1))
    t, h = sc["T96"], sc["H270"]
    print(f"{len(free)}c{'  k = %.3f' % th[NL] if VARIANT == 'K' else ''}: T96 {t[0]:.4e} / {t[1]:.4e} / {t[2]:.4e} | "
          f"H270 {h[0]:.4e} / {h[1]:.4e} / {h[2]:.4e}", flush=True)
    print("   column medians: " + " ".join(f"{k} {v:.2e}" for k, v in sc["col_median"].items()), flush=True)
    print("   row medians: " + " ".join(f"{v:.2e}" for v in sc["row_median"].values()), flush=True)
    for qq in ("2.5", "2.0"):
        v = [x for k, x in sc["lowq"].items() if k.startswith(qq)]
        print(f"   low q {qq}: max {max(v):.3e} median {np.median(v):.3e}", flush=True)


if __name__ == "__main__":
    main()
