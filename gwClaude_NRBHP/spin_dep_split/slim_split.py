"""Slim merger-ringdown branch for the flux split (2026-10-04; staged in $SCRATCH while /work MDT0 is full).

Fixed: the inspiral copy theta_i (9 flux coefficients + derived PN spin-orbit term, stage-1 fit from
spin_dep_split/results/flux_spin_split_{S18,T96}_T-1.json) and the user's Ori–Thorne window (T = -1 to the ppBHPT
peak, eta = 1/q).  Only the MR branch changes.  Stage 2 objective: mean E_MR ('insp' alignment), as before.

MR branch candidates:
  'copy' with a mask : the flux form with the coefficients in the FREE groups fitted, all others EQUAL to theta_i
                       groups: aPP = (c0, c1), aF = (A0, A1), bPP = (b), P = (P0, P1), S = (s1, s2)
  'anchor'           : beta_m = r0 beta_r(q, chi1) (exact Kerr QNM ratio), alpha_m = m X1^(6/5)          (2 coefficients)
Greedy forward selection on S18: round 0 = no MR coefficients (MR copy = theta_i, without the SO factor);
each round adds the group that most lowers the MEAN full-window error ('full' alignment, the published comparison);
stop when the best addition gains < 5% per added coefficient (EXPLORATION.md rule K, compounded).
Then --confirm: the chosen branch refit on T96 (stage 2), scored on T96 and H270.
  python slim_split.py --select          python slim_split.py --confirm "aF,P"
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from multiprocessing import get_context

import numpy as np
from scipy.optimize import minimize

OUT = Path(__file__).resolve().parent
SPLIT = Path("/work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/spin_dep_split")
sys.path.insert(0, str(SPLIT))
import fit_flux_spin_split as FF
import score_h270 as H
M = FF.M
GROUPS = {"aPP": [0, 1], "aF": [2, 3], "bPP": [4], "P": [5, 6], "S": [7, 8]}
_orig_ab = FF.ab


def ab(kind, ti, tm, q, chi, c, key):
    """kind 'split' with tm = ('mask', theta_m) or ('anchor', [m, r0])."""
    if kind != "split" or isinstance(tm, np.ndarray):
        return _orig_ab(kind, ti, tm, q, chi, c, key)
    mode, th = tm
    if mode == "mask":
        return _orig_ab(kind, ti, np.asarray(th), q, chi, c, key)
    pre = "fit_" if key == "fit" else ""
    S = c[pre + "S"]
    a_ins, b_ins = _orig_ab(kind, ti, ti, q, chi, dict(c, **{pre + "S": np.zeros_like(S)}), key)
    m, r0 = th
    X1 = q / (1 + q)
    return (1 - S) * a_ins + S * m * X1 ** 1.2, (1 - S) * b_ins + S * r0 * M.S.beta_r_phys(q, chi)


FF.ab = ab


def build_tm(ti, groups, x):
    th = np.array(ti, float).copy()
    idx = [i for g in groups for i in GROUPS[g]]
    th[idx] = x
    return ("mask", th)


def stage2(pool, ti, spec, x0):
    """Fit the free MR coefficients on mean E_MR ('insp' alignment); spec = ('groups', [...]) or ('anchor',)."""
    def tm_of(x):
        return ("anchor", list(x)) if spec[0] == "anchor" else build_tm(ti, spec[1], x)
    if len(x0) == 0:
        return np.array([]), tm_of(np.array([]))
    FF.run(pool, "split", ti, tm_of(x0), wide=True)
    def obj(x):
        R = FF.run(pool, "split", ti, tm_of(x))
        return float(np.mean([R[k]["errs"][1] for k in FF.KEYS]))
    x = np.array(x0, float); best = obj(x)
    for rep in range(5):
        r = minimize(obj, x, method="Nelder-Mead",
                     options={"maxiter": 900 * len(x), "xatol": 1e-8, "fatol": 1e-12, "adaptive": True})
        FF.run(pool, "split", ti, tm_of(r.x), wide=True); fr = obj(r.x)
        done = fr > best * (1 - 1e-3)
        if fr < best:
            x, best = r.x, fr
        if done:
            break
    return x, tm_of(x)


def score_full_mean(pool, ti, tm):
    rows = FF.final(pool, "split", ti, tm)
    return float(np.mean([r["full_E_full"] for r in rows])), rows


def x0_for(ti, spec):
    if spec[0] == "anchor":
        return np.array([1.0, 1.0])
    return np.array(ti)[[i for g in spec[1] for i in GROUPS[g]]]


def select(pool, ti):
    log = []
    chosen = []
    _, tm0 = stage2(pool, ti, ("groups", []), np.array([]))
    base, _ = score_full_mean(pool, ti, tm0)
    print(f"round 0 (no MR coefficients): mean E_full {base:.4e}", flush=True)
    log.append(dict(round=0, chosen=[], ncoef=0, mean_E_full=base))
    ncoef = 0
    while True:
        cands = [g for g in GROUPS if g not in chosen]
        trials = []
        for g in cands:
            spec = ("groups", chosen + [g]); t = time.time()
            x, tm = stage2(pool, ti, spec, x0_for(ti, spec)); e, _ = score_full_mean(pool, ti, tm)
            trials.append((e, g, list(map(float, x)))); print(f"  + {g:4s}: mean E_full {e:.4e} ({time.time() - t:.0f}s)", flush=True)
        if not chosen:
            t = time.time(); x, tm = stage2(pool, ti, ("anchor",), x0_for(ti, ("anchor",)))
            e, _ = score_full_mean(pool, ti, tm)
            print(f"  anchor (m, r0 on beta_r): mean E_full {e:.4e}  m={x[0]:.4f} r0={x[1]:.4f} ({time.time() - t:.0f}s)", flush=True)
            log.append(dict(round=1, candidate="anchor", mean_E_full=e, x=list(map(float, x))))
        e, g, x = min(trials)
        add = len(GROUPS[g]); need = base * 0.95 ** add
        log.append(dict(round=len(chosen) + 1, trials=[dict(group=tg, mean_E_full=te) for te, tg, _ in trials],
                        best=g, mean_E_full=e, need=need))
        print(f"round {len(chosen) + 1}: best +{g} -> {e:.4e} (need <= {need:.4e} for >= 5%/coef)", flush=True)
        if e > need:
            print("  stop: best addition does not earn its coefficients", flush=True)
            break
        chosen.append(g); base = e; ncoef += add
        if len(chosen) == len(GROUPS):
            break
    return chosen, ncoef, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true"); ap.add_argument("--confirm", default=None)
    a = ap.parse_args()
    if a.select:
        FF.KEYS = [(q, c) for q in (3.0, 5.0, 8.0) for c in (-0.6, -0.2, 0.0, 0.4, 0.6, 0.8)]
        for k in FF.KEYS:
            M.CASES[k] = M.prepare(*k)
        ti = np.array(json.loads((SPLIT / "results" / "flux_spin_split_S18_T-1.json").read_text())["theta_insp"])
        with get_context("fork").Pool(len(FF.KEYS)) as pool:
            chosen, n, log = select(pool, ti)
        print(f"SELECTED MR groups: {chosen} ({n} MR coefficients; total {9 + n})")
        (OUT / "results" / "slim_select_S18.json").write_text(json.dumps(dict(chosen=chosen, n_mr=n, log=log), indent=1))
    if a.confirm is not None:
        groups = [g for g in a.confirm.split(",") if g]
        T96 = [(q, c) for q in M.S.TRAIN_Q for c in M.S.CHIS]
        for k in T96 + H.H270:
            M.CASES[k] = M.prepare(*k)
        ti = np.array(json.loads((SPLIT / "results" / "flux_spin_split_T96_T-1.json").read_text())["theta_insp"])
        spec = ("anchor",) if groups == ["anchor"] else ("groups", groups)
        res = {}
        with get_context("fork").Pool(128) as pool:
            FF.KEYS = T96; FF.SEEDS.clear(); t = time.time()
            x, tm = stage2(pool, ti, spec, x0_for(ti, spec))
            print(f"T96 stage 2 for {groups}: {time.time() - t:.0f}s", flush=True)
            for grid, keys in (("T96", T96), ("H270", H.H270)):
                FF.KEYS = keys; FF.SEEDS.clear()
                res[grid] = FF.final(pool, "split", ti, tm)
                print(grid + ":"); FF.summary(res[grid], f"slim split {groups} ({9 + len(x)}c)")
        (OUT / "results" / f"slim_confirm_{'_'.join(groups) or 'none'}.json").write_text(json.dumps(dict(
            groups=groups, n_total=9 + len(x), x=list(map(float, x)), theta_insp=list(map(float, ti)),
            T96=res["T96"], H270=res["H270"]), indent=1))


if __name__ == "__main__":
    main()
