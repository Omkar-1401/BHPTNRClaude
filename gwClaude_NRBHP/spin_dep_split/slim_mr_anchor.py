"""Pipeline item 2 (2026-10-05): tie the slim split's merger-ringdown coefficients to physical q-dependence, to fix its
extrapolation below q = 3 (slim14 is best in range but worse than the 9-coefficient one-form model at q = 2, 2.5).

Inspiral copy and Ori–Thorne window unchanged (results/slim_confirm_aF_bPP_S.json).  MR-branch variants, each refit
in stage 2 only (mean E_MR on T96, 'insp' alignment, as the slim model):
  slim14 : as installed-variant: free A0', A1', b', s1', s2'                                   (14 coefficients)
  a      : A1' shared with the inspiral copy (only A0' free in the MR flux coupling)              (13)
  b      : MR beta prefactor anchored to the exact Kerr ringdown ratio:
           b_pp' = beta_r(q, chi1) (1 + b' nu)  instead of  X1^(6/5) (1 + b' nu);  P' E term unchanged   (14)
  c      : a + b                                                                                 (13)
Scored (full-window E, full alignment): T96, H270, q = 2 and 2.5 (8 spins; ppBHPT itself extrapolated there).
-> results/slim_mr_anchor.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from multiprocessing import get_context

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extrap_lowq as XL                 # patched get_case (shared caches read-only; low-q drives local)
SL, FF, M = XL.SL, XL.FF, XL.M
import score_h270 as H
FS = FF.FS
SL.GROUPS["aF0"] = [2]                   # A0' only
_slim_ab = FF.ab                         # slim_split's masked-MR ab (already installed by slim_split)


def ab(kind, ti, tm, q, chi, c, key):
    """Adds mode 'maskR': masked MR copy with the beta prefactor anchored to beta_r(q, chi1)."""
    if kind == "split" and isinstance(tm, tuple) and tm[0] == "maskR":
        pre = "fit_" if key == "fit" else ""
        S = c[pre + "S"]
        a_i, b_i = _slim_ab(kind, ti, np.asarray(ti), q, chi, dict(c, **{pre + "S": np.zeros_like(S)}), key)
        a_pp, a_f, b_pp, b_e = FS.params_at(q, chi, np.asarray(tm[1]))
        los = c[pre + "losses"]; X1 = q / (1 + q)
        R = M.S.beta_r_phys(q, chi) / X1 ** 1.2
        a_m = a_pp * (1 + a_f * los["flux_hat"])
        b_m = b_pp * R + b_e * b_pp * los["e_oft"]
        return (1 - S) * a_i + S * a_m, (1 - S) * b_i + S * b_m
    return _slim_ab(kind, ti, tm, q, chi, c, key)


FF.ab = ab


def tm_of(ti, groups, anchored, x):
    th = np.array(ti, float).copy()
    th[[i for g in groups for i in SL.GROUPS[g]]] = x
    return ("maskR" if anchored else "mask", th)


def stage2(pool, ti, groups, anchored, x0):
    FF.run(pool, "split", ti, tm_of(ti, groups, anchored, x0), wide=True)
    def obj(x):
        R = FF.run(pool, "split", ti, tm_of(ti, groups, anchored, x))
        return float(np.mean([R[k]["errs"][1] for k in FF.KEYS]))
    x = np.array(x0, float); best = f0 = obj(x)
    for rep in range(5):
        r = minimize(obj, x, method="Nelder-Mead",
                     options={"maxiter": 900 * len(x), "xatol": 1e-8, "fatol": 1e-12, "adaptive": True})
        FF.run(pool, "split", ti, tm_of(ti, groups, anchored, r.x), wide=True); fr = obj(r.x)
        done = fr > best * (1 - 1e-3)
        if fr < best:
            x, best = r.x, fr
        if done:
            break
    return x, f0, best


def stats(rows):
    v = np.array([r["full_E_full"] for r in rows]); return [float(np.median(v)), float(v.mean()), float(v.max())]


def main():
    T96 = [(q, c) for q in M.S.TRAIN_Q for c in M.S.CHIS]
    LOWQ = XL.CASES_LIST
    for k in T96 + H.H270 + LOWQ:
        M.CASES[k] = M.prepare(*k)
    R = json.loads((HERE / "results" / "slim_confirm_aF_bPP_S.json").read_text())
    ti = np.array(R["theta_insp"]); x_slim = np.array(R["x"])          # order: A0', A1', b', s1', s2'
    variants = {"slim14": (["aF", "bPP", "S"], False, None),
                "a":      (["aF0", "bPP", "S"], False, np.r_[x_slim[0], x_slim[2:]]),
                "b":      (["aF", "bPP", "S"], True, x_slim.copy()),
                "c":      (["aF0", "bPP", "S"], True, np.r_[x_slim[0], x_slim[2:]])}
    out = {}
    with get_context("fork").Pool(128) as pool:
        for name, (groups, anchored, x0) in variants.items():
            FF.KEYS = T96; FF.SEEDS.clear(); t = time.time()
            if x0 is None:
                x = x_slim; msg = "no refit (slim14 as fitted)"
            else:
                x, f0, f1 = stage2(pool, ti, groups, anchored, x0)
                msg = f"stage 2: mean E_MR {f0:.4e} -> {f1:.4e} ({time.time() - t:.0f}s)"
            tm = tm_of(ti, groups, anchored, x)
            res = {}
            for g, keys in (("T96", T96), ("H270", H.H270), ("q2.5", [k for k in LOWQ if k[0] == 2.5]),
                            ("q2", [k for k in LOWQ if k[0] == 2.0])):
                FF.KEYS = keys; FF.SEEDS.clear()
                res[g] = FF.final(pool, "split", ti, tm)
            out[name] = dict(groups=groups, anchored=anchored, n_coef=9 + len(x), x=list(map(float, x)),
                             **{g: stats(rows) for g, rows in res.items()}, rows=res)
            print(f"{name:7s} ({9 + len(x)}c) {msg}\n        " + "  ".join(
                f"{g}: {out[name][g][0]:.3e}/{out[name][g][1]:.3e}/{out[name][g][2]:.2e}" for g in ("T96", "H270", "q2.5", "q2")),
                flush=True)
    (HERE / "results" / "slim_mr_anchor.json").write_text(json.dumps(out, indent=1))
    print("reference (full-window E median): one-form flux_anc_spin_with_orbit T96 8.86e-4, H270 7.20e-4, q2.5 4.36e-3, q2 1.94e-2")
    print("wrote results/slim_mr_anchor.json")


if __name__ == "__main__":
    main()
