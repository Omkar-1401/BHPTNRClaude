"""EXPLORATION pass 25 -- nu exponent of the corrections (D1) on V7-S3, then pruning below 20 coefficients.

V7-S3 (pass 23 Z, 20c) weights every merger-ringdown correction by wv = nu/nu5.  The user (2026-10-08) asked to remove the
rise past q = 5; the pass-17 per-case floor is flat in q (0.12-0.17e-4), and the leading residual amplitudes are q-flat
at fixed chi1, so the nu^1 weight is the suspect.  Here
    wv_mr = (nu/nu5)^p_mr   on the MR offsets (r0, r1, r3, l0),     wv_sh = (nu/nu5)^p_sh   on the four shape terms.
p_mr = p_sh = 1 is V7-S3 exactly (checked at start against p23_Z.json, ratio must be 1.000).
Step 1: refit V7-S3 + (p_mr, p_sh)  (22c).
Step 2: single-drop costs of every free coefficient from the step-1 solution (drop = 0; for p_* drop = back to 1).
Step 3: joint drop of all with cost < CUT; refit; full scoring (T96, H270, low q).
Usage: python p25_nuexp.py [budget1_min].  Writes exploration/p25/p25_nuexp.json (rewritten after each step).
"""
from __future__ import annotations
import json, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; REPO = HERE.parent
sys.path.insert(0, str(HERE))
B1 = float(sys.argv[1]) if len(sys.argv) > 1 else 35.0
sys.argv = [sys.argv[0], "Z"]
import p23_s3 as Z                              # noqa: E402  (loads V7-S3 machinery; P6.ab = Z.ab)
L = Z.L; A = Z.A; P6 = Z.P6; PR = Z.PR; P3 = Z.P3; S = Z.S
from p14_v6 import flux13                       # noqa: E402
OUT = HERE / "p25"; OUT.mkdir(exist_ok=True)
REF = json.loads((HERE / "p23" / "p23_Z.json").read_text())
TH0 = np.r_[np.array(REF["theta"]), 1.0, 1.0]; NV = len(REF["theta"])          # 40 + p_mr + p_sh
NAMES = PR.NAMES + ["p_mr", "p_sh"]
IPM, IPS = NV, NV + 1
FREE0 = list(REF["free"]) + [IPM, IPS]
CUT = 0.03; STEP_MIN = 2.2
T_START = time.time()
NU5 = 5.0 / 36.0
S1FUN = None                    # optional hook (pass 26): s1 = S1FUN(k, d) instead of t1a + t1b chi1
WSFUN = None                    # optional hook (pass 28): (ws_beta, ws_alpha) = WSFUN(th, nu, chi1) instead of ws for both


def ab(th, k, key):
    q, chi1 = k; c = P6.get(k); case = c["case"]; pre = "fit_" if key == "fit" else ""
    d = dict(zip(NAMES, th))
    aF, bF = flux13(np.asarray(th[:13]), q, chi1, case, key)
    s1 = d["t1a"] + d["t1b"] * chi1 if S1FUN is None else S1FUN(k, d); s2 = d["t2a"] + d["t2b"] * chi1
    if not (-150 <= s1 <= 30 and s1 + 3 <= s2 <= 150) or np.any(aF <= 0) or np.any(bF <= 0):
        return None, None
    nu, X1 = q / (1 + q) ** 2, q / (1 + q)
    br, kap = c["anc"]
    s = case[pre + "t_bhpt"] - c["tpk"]
    W = P3.smooth5((s - s1) / (s2 - s1))
    wm = (nu / NU5) ** d["p_mr"]; ws = (nu / NU5) ** d["p_sh"]
    lbm = np.log(br) + wm * (d["r0"] + d["r1"] * chi1 + d["r2"] * nu + d["r3"] * chi1 ** 2)
    lam = (1.2 * np.log(X1) + wm * (d["l0"] + d["l1"] * chi1 + d["l2"] * nu + d["l3"] * chi1 ** 2)
           + kap * br * np.clip(s - P6.S_RING, 0, None))
    # S3: inspiral alpha amplitude (1 + nu R2/2), flux_c0 = 0 in V7-S3
    if (k, key) not in Z._X:
        if len(Z._X) > 400:
            Z._X.clear()
        Z._X[(k, key)] = np.interp(S.PC.x_scale(q) * np.minimum(case[pre + "x"], S.XCLIP), Z._RX, Z._RY)
    laI = np.log(aF) + np.log1p(nu * Z._X[(k, key)] / 2) - np.log1p(d["flux_c0"] * nu)
    la = (1 - W) * laI + W * lam
    lb = (1 - W) * np.log(bF) + W * lbm
    Aa = d["pa0"] + d["pa1"] * chi1 + d["pa2"] * chi1 ** 2 + d["pa3"] * nu
    Bb = d["pb0"] + d["pb1"] * chi1 + d["pb2"] * chi1 ** 2
    wsb, wsa = (ws, ws) if WSFUN is None else WSFUN(th, nu, chi1)
    lb = lb + wsb * Aa * A._PB(np.clip(s, *A._KB))
    la = la + wsa * Bb * A._PA(np.clip(s, *A._KA))
    for j, (field, sp, (lo, hi), _) in enumerate(L.SH):
        e0, e1, e2 = th[34 + 3 * j: 37 + 3 * j]
        y = (wsb if field == "dlnB" else wsa) * (e0 + e1 * chi1 + e2 * chi1 ** 2) * sp(np.clip(s, lo, hi))
        if field == "dlnB":
            lb = lb + y
        else:
            la = la + y
    return np.exp(la), np.exp(lb)


P6.ab = ab


def save(out):
    (OUT / "p25_nuexp.json").write_text(json.dumps(out, indent=1))


def main():
    assert abs(REF["theta"][NAMES.index("flux_c0")]) == 0 and REF["theta"][NAMES.index("tie")] == 0
    pool = mp.get_context("fork").Pool(96)
    # reproduce V7-S3 (p_mr = p_sh = 1) on 12 cases at full resolution
    chk = P6.T96[::8]
    e = np.array(pool.map(P6._full, [(list(TH0), k) for k in chk]))
    r = e / np.array([REF["T96_cases"][f"{q:.6f}_{c:+.4f}"] for q, c in chk])
    print(f"reproduce V7-S3: ratio min {r.min():.4f} max {r.max():.4f}", flush=True)
    assert np.allclose(r, 1, atol=2e-3), r
    out = dict(names=NAMES, CUT=CUT, reproduce=[float(r.min()), float(r.max())])
    th1, e1 = A.refit(pool, TH0, FREE0, B1, "nu-exp")
    out["step1"] = dict(theta=list(map(float, th1)), free=FREE0, fit_mean=e1, p_mr=float(th1[IPM]), p_sh=float(th1[IPS]))
    print(f"step 1 (22c): fit-mean {e1:.4e}, p_mr {th1[IPM]:.3f}, p_sh {th1[IPS]:.3f}", flush=True)
    out["step1"].update(PR.score(pool, th1)); save(out)
    t = out["step1"]["T96"]
    print(f"   T96 {t[0]:.4e} / {t[1]:.4e} / {t[2]:.4e}  rows " + " ".join(f"{v:.2e}" for v in out["step1"]["row_median"].values()), flush=True)
    out["round1"] = {}
    for i in FREE0:
        f0 = th1.copy(); f0[i] = 1.0 if i in (IPM, IPS) else 0.0
        _, e = A.refit(pool, f0, [j for j in FREE0 if j != i], STEP_MIN, NAMES[i])
        out["round1"][NAMES[i]] = dict(fit_mean=e, cost=e / e1 - 1)
        print(f"{NAMES[i]:10s} cost {100 * (e / e1 - 1):+7.2f} %", flush=True); save(out)
    zero = sorted(n for n, rr in out["round1"].items() if rr["cost"] < CUT)
    f0 = th1.copy()
    for n in zero:
        f0[NAMES.index(n)] = 1.0 if n in ("p_mr", "p_sh") else 0.0
    free = [j for j in FREE0 if NAMES[j] not in zero]
    left = 113 - (time.time() - T_START) / 60 - 8
    th2, e2 = A.refit(pool, f0, free, max(left, 6), "joint")
    out["round2"] = dict(zeroed=zero, n_coef=len(free), free=free, theta=list(map(float, th2)), fit_mean=e2, **PR.score(pool, th2))
    save(out)
    t, h = out["round2"]["T96"], out["round2"]["H270"]
    print(f"pruned {len(free)}c (zeroed {zero}): T96 {t[0]:.4e} / {t[1]:.4e} / {t[2]:.4e} | H270 {h[0]:.4e} / {h[1]:.4e} / {h[2]:.4e}", flush=True)
    print("   rows " + " ".join(f"{v:.2e}" for v in out["round2"]["row_median"].values()), flush=True)
    print("   cols " + " ".join(f"{k} {v:.2e}" for k, v in out["round2"]["col_median"].items()), flush=True)
    for qq in ("2.5", "2.0"):
        v = [x for kk, x in out["round2"]["lowq"].items() if kk.startswith(qq)]
        print(f"   low q {qq}: max {max(v):.3e} median {np.median(v):.3e}", flush=True)


if __name__ == "__main__":
    main()
