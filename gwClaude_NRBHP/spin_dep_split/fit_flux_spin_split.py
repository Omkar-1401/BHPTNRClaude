"""fluxanchored_spin_split (overnight step 5 VARIANT, 2026-10-04; not adopted, not installed).

Inspiral copy (9 coefficients, theta_i, fit on E_insp):
    alpha_i = a_pp (1 + a_F F),  beta_i = b_pp (1 + b_e E) (1 + so(q) chi1 x^(3/2))
    (a_pp, a_F, b_pp, b_e) = fit_fluxanchored_spin.params_at(q, chi1, theta_i);  so(q) = derived PN spin-orbit
    coefficient (3/2)[113/12 X1^(6/5) - 19/6 X1^2 - 25/4 X1] (OVERNIGHT_LOG step 1 + correction), no fitted multiplier.
MR copy (9 coefficients, theta_m, fit on E_MR): the same flux form WITHOUT the SO term, own coefficients.
    alpha = (1 - S) alpha_i + S alpha_m,   beta = (1 - S) beta_i + S beta_m
S = Ori–Thorne window blend (ot_join.blend; eta = 1/q, quintic, end at the ppBHPT peak), two variants:
    'T-1'  : window from T = -1 (the user's decision)        'isco' : window from T = T* (the ISCO) -- variant only
theta_m starts EQUAL to theta_i, so the blend initially changes nothing (stage 1 cannot be contaminated).
Two-stage: stage 1 theta_i on mean E_insp ('insp' alignment), stage 2 theta_m on mean E_MR (inspiral frozen),
repeat once if stage 2 moved mean E_insp by > 2%.  Scored both alignments, with E_full, and against the installed
fluxanchored_spin and pn_anchored_spin on the same S18 cases and alignments.
  python fit_flux_spin_split.py --window T-1|isco
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from multiprocessing import get_context

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_pn_anchored_spin_split as M
import ot_join as OT
import diag_so_term as SO

FS = M.F.FS
KEYS = [(q, c) for q in (3.0, 5.0, 8.0) for c in (-0.6, -0.2, 0.0, 0.4, 0.6, 0.8)]
SEEDS = {}


def set_window(window):
    for k in KEYS:
        c = M.CASES[k]
        if window == "isco":
            q, chi = k; t, h = c["t_bhpt"], c["h_bhpt"]
            T, _ = OT.ot_time(t, c["lm"]["t_isco_pp"], q, chi)
            Tp = T[int(np.argmax(np.abs(h)))]
            S = OT.smoothstep5((T - OT.T_STAR) / (Tp - OT.T_STAR))
            c["S"] = S; c["fit_S"] = np.interp(c["fit_t_bhpt"], t, S)
        # 'T-1': M.prepare already set S from ot_join.blend


def ab(kind, ti, tm, q, chi, c, key):
    pre = "fit_" if key == "fit" else ""
    if kind != "split":
        return M.F.ab(kind, M.F.installed(kind, q, chi), q, chi, c, key)
    los = c[pre + "losses"]; xc = SO.xnote(q, c, pre)
    def form(th, so):
        a_pp, a_f, b_pp, b_e = FS.params_at(q, chi, th)
        al = a_pp * (1 + a_f * los["flux_hat"]); be = b_pp * (1 + b_e * los["e_oft"])
        return al, (be * (1 + SO.so_coef(q, corrected=True) * chi * xc ** 1.5) if so else be)
    ai, bi = form(ti, True); am, bm = form(tm, False); S = c[pre + "S"]
    return (1 - S) * ai + S * am, (1 - S) * bi + S * bm


def _task(a):
    # the seed travels with the task: workers hold a fork-time copy of SEEDS (bug in run 3487212/3)
    kind, k, ti, tm, align, key, wide, seed = a
    al, be = ab(kind, ti, tm, *k, M.CASES[k], key)
    errs, t0 = M.score(al, be, M.CASES[k], key, align, seed, wide)
    return k, errs, t0, float(np.min(be)), float(np.min(al))


def run(pool, kind, ti, tm, align="insp", key="fit", wide=False):
    out = pool.map(_task, [(kind, k, ti, tm, align, key, wide, SEEDS.get((kind, align, k), -50.0)) for k in KEYS])
    for k, e, t0, _, _ in out:
        if e[0] < 1.0:
            SEEDS[(kind, align, k)] = t0
    return {k: dict(errs=e, beta_min=bm, alpha_min=am) for k, e, t0, bm, am in out}


def stage(pool, which, ti, tm, log):
    i = 0 if which == "insp" else 1
    run(pool, "split", ti, tm, wide=True)
    def obj(x):
        a, b = (x, tm) if which == "insp" else (ti, x)
        R = run(pool, "split", a, b)
        return float(np.mean([R[k]["errs"][i] for k in KEYS]))
    x = np.array(ti if which == "insp" else tm, float); t = time.time(); f0 = best = obj(x)
    for rep in range(5):
        r = minimize(obj, x, method="Nelder-Mead",
                     options={"maxiter": 800 * len(x), "xatol": 1e-8, "fatol": 1e-12, "adaptive": True})
        a, b = (r.x, tm) if which == "insp" else (ti, r.x)
        run(pool, "split", a, b, wide=True); fr = obj(r.x)
        done = fr > best * (1 - 1e-3)
        if fr < best:
            x, best = r.x, fr
        if done:
            break
    log.append(dict(stage=which, start=f0, end=best, seconds=round(time.time() - t)))
    print(f"  stage {which}: mean {'E_insp' if i == 0 else 'E_MR'} {f0:.4e} -> {best:.4e} ({time.time() - t:.0f}s)", flush=True)
    return (x, tm) if which == "insp" else (ti, x)


def final(pool, kind, ti, tm):
    rows = {k: dict(q=k[0], chi1=k[1]) for k in KEYS}
    for align in ("insp", "full"):
        run(pool, kind, ti, tm, align, "fit", wide=True)
        R = run(pool, kind, ti, tm, align, "full", wide=True)
        for k in KEYS:
            for j, n in enumerate(("E_insp", "E_MR", "E_full")):
                rows[k][f"{align}_{n}"] = R[k]["errs"][j]
            rows[k]["beta_min"] = min(rows[k].get("beta_min", 9), R[k]["beta_min"])
            rows[k]["alpha_min"] = min(rows[k].get("alpha_min", 9), R[k]["alpha_min"])
    return list(rows.values())


def summary(rows, label):
    def s(key):
        v = np.array([r[key] for r in rows]); return f"{np.median(v):.2e}/{np.mean(v):.2e}/{v.max():.1e}"
    print(f"  {label:30s} median/mean/max | insp-aligned: E_insp {s('insp_E_insp')}  E_MR {s('insp_E_MR')}  "
          f"E_full {s('insp_E_full')} | full-aligned: E_insp {s('full_E_insp')}  E_MR {s('full_E_MR')}  "
          f"E_full {s('full_E_full')}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--window", default="T-1", choices=["T-1", "isco"])
    ap.add_argument("--eta", default="1/q", choices=["1/q", "nu"])
    ap.add_argument("--grid", default="S18", choices=["S18", "T96"])
    ap.add_argument("--start", default=None, help="results json to start from (theta_insp, theta_mr)")
    a = ap.parse_args()
    global KEYS
    if a.grid == "T96":
        KEYS = [(q, c) for q in M.S.TRAIN_Q for c in M.S.CHIS]
    OT.ETA = a.eta                                  # must be set before M.prepare builds the blend
    for k in KEYS:
        M.CASES[k] = M.prepare(*k)
    set_window(a.window)
    ti = np.array(json.loads((HERE / "results" / "diag_so_term_corrected.json").read_text())["variants"]["fl_SOD"]["theta_best"])
    tm = ti.copy()
    if a.start:
        st = json.loads(Path(a.start).read_text()); ti = np.array(st["theta_insp"]); tm = np.array(st["theta_mr"])
    with get_context("fork").Pool(len(KEYS)) as pool:
        base = {kind: final(pool, kind, None, None) for kind in ("flux9", "pn13")}
        summary(base["flux9"], "installed fluxanchored_spin")
        summary(base["pn13"], "installed pn_anchored_spin")
        start = final(pool, "split", ti, tm); summary(start, "start (theta_m = theta_i)")
        log = []
        ti, tm = stage(pool, "insp", ti, tm, log); e1 = log[-1]["end"]
        ti, tm = stage(pool, "mr", ti, tm, log)
        R = run(pool, "split", ti, tm, wide=True); e2 = float(np.mean([R[k]["errs"][0] for k in KEYS]))
        print(f"  mean E_insp after stage 2: {e2:.4e} ({100 * abs(e2 / e1 - 1):.1f}% from stage 1)", flush=True)
        if abs(e2 / e1 - 1) > 0.02:
            ti, tm = stage(pool, "insp", ti, tm, log); ti, tm = stage(pool, "mr", ti, tm, log)
        rows = final(pool, "split", ti, tm)
        summary(rows, f"flux split ({a.window}, eta={a.eta})")
    badP = [(r["q"], r["chi1"]) for r in rows if r["beta_min"] <= 0 or r["alpha_min"] <= 0]
    print("gate P (beta > 0, alpha > 0, full resolution):", "ok" if not badP else f"FAIL {badP}")
    tag = a.window + ("" if a.eta == "1/q" else "_etanu")
    out = HERE / "results" / f"flux_spin_split_{a.grid}_{tag}.json"
    out.write_text(json.dumps(dict(model="fluxanchored_spin_split (variant)", window=a.window, names=list(FS.NAMES),
                                   theta_insp=list(map(float, ti)), theta_mr=list(map(float, tm)), stages=log,
                                   cases=rows, baselines=base, start=start, gateP_failures=badP), indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
