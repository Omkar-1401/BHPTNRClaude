"""pn_anchored_spin_split: pn_anchored_spin's inspiral and merger forms, joined over the Ori–Thorne window
and fitted in two stages to the Kerr-ISCO-split error.  Design decisions: README.md (user, 2026-10-03).

Model (ppBHPT time t, m1 units; X1 = q/(1+q), nu = q/(1+q)^2, base = X1^(6/5), xc = note's x, clipped):
    S(t)     = ot_join.blend: 0 -> 1 over OT time T in [-1, T(ppBHPT peak)], quintic smoothstep
    beta     = (1-S) base (1 + nu (b1c xc + b2c xc^2))           + S r0 beta_r(q, chi1)
    alpha    = (1-S) base (1 + (55/42) nu xc + nu a2 xc^2)       + S base (m0 + m1 nu)
    b1c = b1 + s11 chi1 + s12 chi1^2,   b2c = b2 + s21 chi1 + s22 chi1^2
Same inspiral and MR forms as fit_pn_anchored_spin.model_ab (anchors X1^(6/5), 55/42, exact-Kerr beta_r,
x clock, XCLIP); only S changes: the fitted energy-loss switch (p00, p01, w0) is replaced by the derived
OT window.  10 coefficients: inspiral b1 b2 a2 s11 s12 s21 s22, MR m0 m1 r0.

Error (shared t0, phi0 over the whole waveform; NR cut at t_isco = the baseline's):
    E_insp on NR [-5000.1, t_isco),  E_MR on [t_isco, +100],  each over its own NR norm (with the 1/2)
  alignment 'insp' : t0, phi0 minimise E_insp; the MR inherits them      (used for FITTING)
  alignment 'full' : t0, phi0 minimise the full-window error (as every installed model)
Every case is scored both ways, with E_full too.

Fit (two-stage, MEAN over the grid like the installed model):
  stage 1  inspiral coefficients on mean E_insp, MR fixed (installed m0, m1, r0 to start)
  stage 2  inspiral frozen, MR coefficients on mean E_MR (alignment re-optimised on E_insp each call)
  repeat 1 -> 2 once if stage 2 moved mean E_insp by more than 2%.

Gates before fitting (stop on failure): installed pn_anchored_spin rescored here with 'full' alignment
reproduces its stored E (and the baseline's E_insp, E_MR); the NR cut equals the baseline's t_isco.

  python fit_pn_anchored_spin_split.py --grid S18 [--nproc 18]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from multiprocessing import get_context

import numpy as np
from scipy.optimize import minimize, minimize_scalar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "spin_dep"))
import ot_join as OT
import plot_spin_diagnostics as D          # read-only cache access, peak(), the installed models' code
F, S = D.F, D.S
PC = S.PC

MODEL = "pn_anchored_spin_split"
FORM = ("pn_anchored_spin inspiral/MR forms; S = OT window [T=-1, ppBHPT peak], eta=1/q, quintic; "
        "2-stage fit on Kerr-ISCO-split error, 'insp' alignment")
NAMES_IN = ["b1", "b2", "a2", "s11", "s12", "s21", "s22"]
NAMES_MR = ["m0", "m1", "r0"]
GRIDS = {"S18": ([3.0, 5.0, 8.0], [-0.6, -0.2, 0.0, 0.4, 0.6, 0.8]),
         "T96": (S.TRAIN_Q, S.CHIS)}
BASELINE = HERE / "results" / "isco_split_baseline.json"
CASES = {}


def installed_start():
    d = json.loads(S.COEFFS.read_text())
    b1, b2, a2, p00, p01, w0, r0, m0, m1 = d["theta9"]
    return np.array([b1, b2, a2, *d["theta_spin"]]), np.array([m0, m1, r0])


def projected_start(keys):
    """Starting point: the installed pn_anchored_spin projected onto this form by linear least squares.
    The installed switch is never off (its MR weight is 5% at -5000 m1, 15-19% at the peak), so its own
    inspiral coefficients do not describe its actual alpha, beta; used with S = 0 they put beta ~1% high.
    Inspiral: fit (b1, b2, s11, s12, s21, s22) and a2 to the installed beta, alpha on ppBHPT t in
    [-6500, window start]; MR: r0 and (m0, m1) to the installed beta, alpha on [peak, peak + 60]."""
    Ab, yb, Aa, ya, Ar, yr, Am, ym = [], [], [], [], [], [], [], []
    for k in keys:
        q, chi = k; c = CASES[k]; t = c["t_bhpt"]
        X1 = q / (1 + q); nu = q / (1 + q) ** 2; base = X1 ** 1.2
        al, be = F.ab("pn13", F.installed("pn13", q, chi), q, chi, c, "full")
        xc = PC.x_scale(q) * np.minimum(c["x"], S.XCLIP)
        m = (t >= -6500.0) & (t <= c["lm"]["t_window_start"]); m &= np.arange(len(t)) % 20 == 0
        x1, x2 = xc[m], xc[m] ** 2
        Ab.append(nu * np.c_[x1, x2, chi * x1, chi ** 2 * x1, chi * x2, chi ** 2 * x2]); yb.append(be[m] / base - 1)
        Aa.append(nu * x2[:, None]); ya.append(al[m] / base - 1 - S.PN_SLOPE * nu * x1)
        ipk = int(np.argmax(np.abs(c["h_bhpt"]))); mm = (t >= t[ipk]) & (t <= t[ipk] + 60.0)
        Ar.append((S.beta_r_phys(q, chi) * np.ones(mm.sum()))[:, None]); yr.append(be[mm])
        Am.append(np.c_[np.ones(mm.sum()), nu * np.ones(mm.sum())]); ym.append(al[mm] / base)
    sol = lambda A, y: np.linalg.lstsq(np.vstack(A), np.concatenate(y), rcond=None)[0]
    b1, b2, s11, s12, s21, s22 = sol(Ab, yb); (a2,) = sol(Aa, ya); (r0,) = sol(Ar, yr); m0, m1 = sol(Am, ym)
    return np.array([b1, b2, a2, s11, s12, s21, s22]), np.array([m0, m1, r0])


def prepare(q, chi):
    c = D.get_case(q, chi, "grid")
    t, h = c["t_bhpt"], c["h_bhpt"]
    Sb, lm = OT.blend(t, h, q, chi)
    tpk_pp, _ = D.peak(t, h); tpk_nr, _ = D.peak(c["t_nr"], c["h_nr"])
    X1 = q / (1 + q)
    lm.update(t_cut_nr=float(tpk_nr + X1 ** 1.2 * (lm["t_isco_pp"] - tpk_pp)),
              in_join_window=bool(-9.0 <= lm["t_isco_pp"] <= -5.0))
    c["S"] = Sb; c["fit_S"] = np.interp(c["fit_t_bhpt"], t, Sb); c["lm"] = lm
    return c


def model_ab(p_in, p_mr, q, chi, c, key):
    pre = "fit_" if key == "fit" else ""
    b1, b2, a2, s11, s12, s21, s22 = p_in
    m0, m1, r0 = p_mr
    X1 = q / (1 + q); nu = q / (1 + q) ** 2; base = X1 ** 1.2
    b1c = b1 + s11 * chi + s12 * chi ** 2
    b2c = b2 + s21 * chi + s22 * chi ** 2
    xc = PC.x_scale(q) * np.minimum(c[pre + "x"], S.XCLIP)
    Sb = c[pre + "S"]
    beta = (1 - Sb) * base * (1 + nu * (b1c * xc + b2c * xc ** 2)) + Sb * r0 * S.beta_r_phys(q, chi)
    alpha = (1 - Sb) * base * (1 + S.PN_SLOPE * nu * xc + nu * a2 * xc ** 2) + Sb * base * (m0 + m1 * nu)
    return alpha, beta


def seg_err(t0, alpha, ts, t, h, tn, hn, tcut, align):
    """(E_insp, E_MR, E_full) with phi0 optimal for the 'insp' or the 'full' segment."""
    tau = t0 + ts
    m = (tn >= max(tn[0], tau[0])) & (tn <= min(tn[-1], tau[-1]))
    if m.sum() / len(tn) < S.H.MIN_COVERAGE:
        return 50.0, 50.0, 50.0
    tc, r = tn[m], hn[m]
    g = np.interp(tc, tau, alpha) * (np.interp(tc, tau, h.real) + 1j * np.interp(tc, tau, h.imag))
    ins = tc < tcut
    if ins.sum() == 0 or ins.all():
        return 50.0, 50.0, 50.0
    sel = ins if align == "insp" else np.ones_like(ins)
    z = np.vdot(g[sel], r[sel])
    g = g * (z / abs(z))
    res = np.abs(r - g) ** 2; nr = np.abs(r) ** 2
    return (float(res[ins].sum() / (2 * nr[ins].sum())), float(res[~ins].sum() / (2 * nr[~ins].sum())),
            float(res.sum() / (2 * nr.sum())))


def score(alpha, beta, c, key, align, seed, wide=False):
    """Align (t0 by 1-D search, phi0 analytic) on the chosen segment; return the three errors and t0."""
    pre = "fit_" if key == "fit" else ""
    t, h, tn, hn = c[pre + "t_bhpt"], c[pre + "h_bhpt"], c[pre + "t_nr"], c[pre + "h_nr"]
    if np.any(beta <= 0) or not np.all(np.isfinite(np.r_[alpha, beta])):
        return (50.0, 50.0, 50.0), seed
    ts = F.tau_shape(beta, t)
    if np.min(np.diff(ts)) <= 0:
        return (50.0, 50.0, 50.0), seed
    k = 0 if align == "insp" else 2
    f = lambda v: seg_err(v, alpha, ts, t, h, tn, hn, c["lm"]["t_cut_nr"], align)[k]
    if wide:
        g = np.linspace(seed - 200, seed + 200, 161)
        cen = float(g[int(np.argmin([f(v) for v in g]))]); half = 3.0
    else:
        cen, half = seed, 10.0
    r = minimize_scalar(f, bounds=(cen - half, cen + half), method="bounded",
                        options={"xatol": 1e-4, "maxiter": 80})
    return seg_err(r.x, alpha, ts, t, h, tn, hn, c["lm"]["t_cut_nr"], align), float(r.x)


def _task(a):
    kind, k, p_in, p_mr, seed, align, key, wide = a
    q, chi = k; c = CASES[k]
    if kind == "split":
        al, be = model_ab(p_in, p_mr, q, chi, c, key)
    else:                                                   # installed pn_anchored_spin
        p = F.installed("pn13", q, chi)
        al, be = F.ab("pn13", p, q, chi, c, key)
    errs, t0 = score(al, be, c, key, align, seed, wide)
    return k, errs, t0, float(np.min(be)), float(np.min(al))


class Grid:
    def __init__(self, pool, keys):
        self.pool, self.keys = pool, keys
        self.seeds = {k: -50.0 for k in keys}

    def run(self, kind, p_in, p_mr, align, key, wide=False, seeds=None):
        sd = seeds if seeds is not None else self.seeds
        out = self.pool.map(_task, [(kind, k, p_in, p_mr, sd[k], align, key, wide) for k in self.keys])
        return {k: dict(errs=e, t0=t0, beta_min=bm, alpha_min=am) for k, e, t0, bm, am in out}


def gates(G):
    base = {(r["q"], r["chi1"]): r for r in json.loads(BASELINE.read_text())["results"] if r["model"] == "pn13"}
    stored = json.loads(S.COEFFS.read_text())["mismatches"]
    R = G.run("installed", None, None, "full", "fit", wide=True)
    R = G.run("installed", None, None, "full", "full", wide=True, seeds={k: R[k]["t0"] for k in G.keys})
    bad = 0
    for k in G.keys:
        q, chi = k; b = base[(q, chi)]
        key = next(s for s in stored if abs(float(s.split("_")[0]) - q) < 1e-3 and abs(float(s.split("_")[1]) - chi) < 1e-6)
        ei, em, ef = R[k]["errs"]
        checks = [ef / stored[key], ei / b["E_insp"], em / b["E_MR"]]
        cut_ok = abs(CASES[k]["lm"]["t_cut_nr"] - b["t_isco"]) < 1e-6
        ok = all(abs(x - 1) < 1e-3 for x in checks) and cut_ok
        bad += not ok
        if not ok:
            print(f"GATE FAIL q={q} chi={chi}: ratios {checks}, cut {CASES[k]['lm']['t_cut_nr']:.4f} vs {b['t_isco']:.4f}")
    print(f"GATE installed pn_anchored_spin, 'full' alignment, {len(G.keys)} cases: "
          f"{'PASSED' if not bad else f'FAILED ({bad})'}", flush=True)
    return bad == 0, R


def mean_of(R, i):
    return float(np.mean([R[k]["errs"][i] for k in R]))


def stage(G, which, p_in, p_mr, log):
    """which = 'insp' (fit p_in on mean E_insp) or 'mr' (fit p_mr on mean E_MR)."""
    i = 0 if which == "insp" else 1
    x0 = np.array(p_in if which == "insp" else p_mr, float)
    nev = [0]

    def obj(x):
        pi, pm = (x, p_mr) if which == "insp" else (p_in, x)
        R = G.run("split", pi, pm, "insp", "fit")
        for k in R:
            if R[k]["errs"][0] < 1.0:
                G.seeds[k] = R[k]["t0"]
        nev[0] += 1
        return mean_of(R, i)

    G.run("split", p_in, p_mr, "insp", "fit", wide=True)          # fresh wide t0 seeds
    t = time.time(); f0 = obj(x0); x = x0
    for rep in range(3):
        r = minimize(obj, x, method="Nelder-Mead",
                     options={"maxiter": 600 * len(x), "xatol": 1e-7, "fatol": 1e-11, "adaptive": True})
        x = r.x
        if rep and abs(prev - r.fun) < 1e-3 * r.fun:
            break
        prev = r.fun
    log.append(dict(stage=which, start=f0, end=float(r.fun), evals=nev[0], seconds=round(time.time() - t)))
    print(f"  stage {which}: mean {'E_insp' if i == 0 else 'E_MR'} {f0:.4e} -> {r.fun:.4e} "
          f"({nev[0]} evals, {time.time() - t:.0f}s)", flush=True)
    return (x, p_mr) if which == "insp" else (p_in, x)


def final_scores(G, kind, p_in, p_mr):
    out = {}
    for align in ("insp", "full"):
        R = G.run(kind, p_in, p_mr, align, "fit", wide=True)
        R = G.run(kind, p_in, p_mr, align, "full", wide=True, seeds={k: R[k]["t0"] for k in G.keys})
        out[align] = R
    rows = []
    for k in G.keys:
        q, chi = k
        row = dict(q=q, chi1=chi, **{f"{a}_{n}": out[a][k]["errs"][j] for a in ("insp", "full")
                                     for j, n in enumerate(("E_insp", "E_MR", "E_full"))},
                   t0_insp=out["insp"][k]["t0"], t0_full=out["full"][k]["t0"],
                   beta_min=out["full"][k]["beta_min"], alpha_min=out["full"][k]["alpha_min"])
        if kind == "split":
            row.update(CASES[k]["lm"])
        rows.append(row)
    return rows


def summary(rows, label):
    def s(key):
        v = np.array([r[key] for r in rows]); return f"{np.mean(v):.3e}/{np.median(v):.3e}/{v.max():.2e}"
    print(f"  {label:34s} mean/median/max | insp-aligned: E_insp {s('insp_E_insp')}  E_MR {s('insp_E_MR')}"
          f"  E_full {s('insp_E_full')} | full-aligned: E_insp {s('full_E_insp')}  E_MR {s('full_E_MR')}"
          f"  E_full {s('full_E_full')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="S18", choices=list(GRIDS)); ap.add_argument("--nproc", type=int, default=0)
    a = ap.parse_args()
    qs, cs = GRIDS[a.grid]
    keys = [(q, c) for q in qs for c in cs]
    for k in keys:
        CASES[k] = prepare(*k)
    with get_context("fork").Pool(a.nproc or len(keys)) as pool:
        G = Grid(pool, keys)
        ok, _ = gates(G)
        if not ok:
            sys.exit("gates failed -- not fitting")
        p_in, p_mr = projected_start(keys)
        print("start (installed projected onto this form):", dict(zip(NAMES_IN + NAMES_MR, np.round(np.r_[p_in, p_mr], 4))))
        base_rows = final_scores(G, "installed", None, None)
        summary(base_rows, "baseline (installed pn_anchored_spin)")
        log = []
        p_in, p_mr = stage(G, "insp", p_in, p_mr, log)
        e1 = log[-1]["end"]
        p_in, p_mr = stage(G, "mr", p_in, p_mr, log)
        R = G.run("split", p_in, p_mr, "insp", "fit", wide=True)
        moved = abs(mean_of(R, 0) / e1 - 1)
        print(f"  mean E_insp after stage 2: {mean_of(R, 0):.4e} ({100 * moved:.1f}% from stage 1)", flush=True)
        if moved > 0.02:
            p_in, p_mr = stage(G, "insp", p_in, p_mr, log)
            p_in, p_mr = stage(G, "mr", p_in, p_mr, log)
        rows = final_scores(G, "split", p_in, p_mr)
        summary(rows, MODEL)
    badP = [(r["q"], r["chi1"]) for r in rows if r["beta_min"] <= 0 or r["alpha_min"] <= 0]
    print("gate P (beta > 0, alpha > 0 at full resolution):", "ok" if not badP else f"FAIL {badP}")
    out = HERE / "results" / f"{MODEL}_{a.grid}.json"
    out.write_text(json.dumps(dict(
        model=MODEL, model_form=FORM, grid=a.grid, names_insp=NAMES_IN, names_mr=NAMES_MR,
        theta_insp=list(map(float, p_in)), theta_mr=list(map(float, p_mr)), stages=log,
        cases=rows, baseline_installed=base_rows, gateP_failures=badP), indent=1))
    print("theta_insp", dict(zip(NAMES_IN, np.round(p_in, 5))), "\ntheta_mr", dict(zip(NAMES_MR, np.round(p_mr, 5))))
    print("wrote", out)


if __name__ == "__main__":
    main()
