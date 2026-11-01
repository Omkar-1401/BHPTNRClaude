"""EXPLORATION pass 7 -- M2 with a spin-dependent blend window (and optional chi1^2 terms).

Same model as p6_m2_global.py, theta extended:
  V1 (19):  t1 = t1a + t1b chi1,  t2 = t2a + t2b chi1                  (blend window, ppBHPT clock from its peak)
  V2 (21):  V1 + rho += r3 chi1^2,  ln alpha level += l3 chi1^2
theta = FLUX 9 | t1a t1b t2a t2b | r0 r1 r2 [r3] | l0 l1 l2 [l3]
Both start from the pass-6 T96 solution and are refitted on T96 (Powell), then scored on T96 and H270 at full
resolution.  Writes exploration/p7/p7_<variant>.json (keys q_chi1 with 6 / 4 decimals).
"""
from __future__ import annotations
import json, sys, time, warnings
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, minimize_scalar

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; REPO = HERE.parent
for p in (HERE, REPO / "spin_dep", REPO):
    sys.path.insert(0, str(p))
import p6_m2_global as P6                     # noqa: E402  (get, err machinery, anchors)
import fit_pn_anchored_spin as S              # noqa: E402
import fit_flux_anc_spin_with_orbit as FO     # noqa: E402
import p3_rd as P3                            # noqa: E402
import fit_scaling_PN_opt_creative as creative  # noqa: E402

OUT = HERE / "p7"; OUT.mkdir(exist_ok=True)
VAR = sys.argv[1] if len(sys.argv) > 1 else "V1"
FLUXN = [f"flux_{n}" for n in FO.NAMES]
NAMES = FLUXN + ["t1a", "t1b", "t2a", "t2b", "r0", "r1", "r2"] + (["r3"] if VAR == "V2" else []) + \
        ["l0", "l1", "l2"] + (["l3"] if VAR == "V2" else [])


def unpack(th):
    d = dict(zip(NAMES, th)); d.setdefault("r3", 0.0); d.setdefault("l3", 0.0); return d


def ab(th, k, key):
    q, chi1 = k; c = P6.get(k); case = c["case"]; pre = "fit_" if key == "fit" else ""
    t = case[pre + "t_bhpt"]
    aF, bF = FO.model_ab(np.asarray(th[:9]), q, chi1, case, key)
    d = unpack(th)
    s1 = d["t1a"] + d["t1b"] * chi1; s2 = d["t2a"] + d["t2b"] * chi1
    if not (-90 <= s1 <= 30 and s1 + 3 <= s2 <= 90) or np.any(aF <= 0) or np.any(bF <= 0):
        return None, None
    nu, X1 = q / (1 + q) ** 2, q / (1 + q)
    br, kap = c["anc"]
    s = t - c["tpk"]
    W = P3.smooth5((s - s1) / (s2 - s1))
    lbm = np.log(br) + d["r0"] + d["r1"] * chi1 + d["r2"] * nu + d["r3"] * chi1 ** 2
    lam = (1.2 * np.log(X1) + d["l0"] + d["l1"] * chi1 + d["l2"] * nu + d["l3"] * chi1 ** 2
           + kap * br * np.clip(s - P6.S_RING, 0, None))
    return np.exp((1 - W) * np.log(aF) + W * lam), np.exp((1 - W) * np.log(bF) + W * lbm)


P6.ab = ab          # route P6.err through this model


def key(q, c):
    return f"{q:.6f}_{c:+.4f}"


def main():
    pool = mp.get_context("fork").Pool(96)
    d6 = json.loads((HERE / "p6" / "p6_m2.json").read_text())["stage2_T96"]["theta"]
    flux9, (s1, s2, r0, r1, r2, l0, l1, l2) = d6[:9], d6[9:]
    x0 = flux9 + [s1, 0.0, s2, 0.0, r0, r1, r2] + ([0.0] if VAR == "V2" else []) + [l0, l1, l2] + ([0.0] if VAR == "V2" else [])
    P6.OUT = OUT
    th, note = P6.run_fit(pool, x0, P6.T96, 100, VAR)
    e96 = np.array(pool.map(P6._full, [(list(th), k) for k in P6.T96]))
    import plot_spin_paper_figs as PF
    H = PF.heldout_points()
    eh = np.array(pool.map(P6._full, [(list(th), k) for k in H]))
    out = dict(variant=VAR, names=NAMES, theta=list(map(float, th)), note=note,
               T96={key(*k): float(e) for k, e in zip(P6.T96, e96)}, H270={key(*k): float(e) for k, e in zip(H, eh)})
    (OUT / f"p7_{VAR}.json").write_text(json.dumps(out, indent=1))
    print(f"{VAR}: T96 mean {e96.mean():.4e} median {np.median(e96):.4e} max {e96.max():.4e} | "
          f"H270 mean {eh.mean():.4e} median {np.median(eh):.4e} max {eh.max():.4e}", flush=True)
    for ch in S.CHIS:
        v = [e for (q, c), e in zip(P6.T96, e96) if c == ch]
        print(f"   chi1 {ch:+.1f}: median {np.median(v):.3e} max {np.max(v):.3e}", flush=True)
    print("   q medians: " + " ".join(f"{np.median([e for (qq, c), e in zip(P6.T96, e96) if qq == q]):.2e}" for q in S.TRAIN_Q), flush=True)
    print("   theta: " + ", ".join(f"{n}={v:+.4f}" for n, v in zip(NAMES, th)), flush=True)


if __name__ == "__main__":
    main()
