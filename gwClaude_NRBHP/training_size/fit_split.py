"""Training-size refits of the split models (spin_dep_split/, read-only).  Each model replays the pipeline that produced
its SHIPPED coefficients (fresh-G96 reproduction, user 2026-10-18):

  splitA   slim__win-2.0 (14c).  shipped: results/explore/slim__win-2.0.json (insp=None: theta_i = TI_T96 as stored)
           stage 1 = fit_flux_spin_split.py's T96 pipeline (results/flux_spin_split_T96_T-1.json), window T = -1:
                     FF.stage insp (theta_i on mean E_insp, MR copy theta_m fixed), FF.stage mr (the 9-coefficient MR copy
                     theta_m on mean E_MR), repeated once if mean E_insp moved > 2% (as its main()).  theta_m is then dropped.
           stage 2 = slim MR branch (5) on mean E_MR, window T = -2 (explore_split.stage2).
  splitB   tline at its chosen point (13c + window).  shipped: results/cv/tline__foldall.json
           stage 1 = explore_split.stage1 (b pinned to 0, MR copy = inspiral copy) at the DEFAULT window T = -1 -- cv_split
                     ran it before set_tline;  stage 2 = slim MR (5) on window start min(-3 + 0 chi1, T_peak - 0.55).
           (T0 = -3, T1 = 0 were chosen on T96; held at that value here: the form, not re-tuned per grid.)
The window enters stage 1 although theta_m = theta_i there: the MR copy has no SO term, so the blend still matters.
(Before 2026-10-19 this harness ran stage 1 at the model's own window and used explore_split.stage1 for A; G96 then
missed shipped A / B by 2-4% -- check_split_stage.py, logs/chksplit.3521130.log.)
Starts: t96 = the shipped stage-1 inputs (A: TI_T96 + its theta_m; B: TI_T96) and the T96 slim x; pertN = all perturbed.
Gate: shipped coefficients reproduce their stored T96 full_E_full per case (ratio within 1e-3).
Score: full-window E with full alignment (FF.final -> full_E_full), the published comparison.

    python fit_split.py splitA --grid 24 --start t96
"""
from __future__ import annotations
import argparse, json, sys, time
from multiprocessing import get_context

import numpy as np

import grids as G                              # FIRST: cache routing + sealed guard on S / FS
import common as C
SPLIT = G.REPO / "spin_dep_split"
sys.path.insert(0, str(SPLIT))                 # its fit_pn_anchored_spin_split.py shadows spin_dep's (intended)
import cv_split as CV                          # noqa: E402  (imports explore_split -> extrap_lowq, slim_split)
X, FF, M = CV.X, CV.FF, CV.M
assert M.__file__.startswith(str(SPLIT)), M.__file__
assert X.FS is G.FS and M.D.S is G.S, "split modules must share the routed S / FS"


def get_case(q, chi, where="grid"):           # replaces extrap_lowq.get_case (which exits on a missing cache)
    return G.FS.build_case(q, chi)


M.D.get_case = get_case

ORIGIN_A = json.loads((SPLIT / "results" / "flux_spin_split_T96_T-1.json").read_text())
MODELS = {
    "splitA": dict(src=SPLIT / "results" / "explore" / "slim__win-2.0.json", pin=[], window=lambda: X.set_window(-2.0),
                   stored=lambda d: d["cases"]["T96"], stage1="origin"),
    "splitB": dict(src=SPLIT / "results" / "cv" / "tline__foldall.json", pin=["b"], window=lambda: CV.set_tline(-3.0, 0.0),
                   stored=lambda d: d["train_rows"], stage1="pin"),
}
MR = "slim"


def stage2(pool, ti, x0):
    """explore_split.stage2 with an explicit start."""
    def obj(x, wide=False):
        R = FF.run(pool, "split", ti, ("form", MR, list(x)), wide=wide)
        return float(np.mean([R[k]["errs"][1] for k in FF.KEYS]))
    obj(x0, wide=True)
    return X.nm(pool, obj, x0)


def stage1_origin(pool, ti, tm, log):
    """fit_flux_spin_split.main's stage sequence (insp, mr, and once more if E_insp drifted > 2%)."""
    ti, tm = FF.stage(pool, "insp", ti, tm, log); e1 = log[-1]["end"]
    ti, tm = FF.stage(pool, "mr", ti, tm, log)
    R = FF.run(pool, "split", ti, tm, wide=True); e2 = float(np.mean([R[k]["errs"][0] for k in FF.KEYS]))
    log.append(dict(stage="E_insp after mr", value=e2))
    if abs(e2 / e1 - 1) > 0.02:
        ti, tm = FF.stage(pool, "insp", ti, tm, log); ti, tm = FF.stage(pool, "mr", ti, tm, log)
    R = FF.run(pool, "split", ti, tm, wide=True)
    return ti, tm, float(np.mean([R[k]["errs"][0] for k in FF.KEYS]))


def rows_E(rows):
    return {G.key(r["q"], r["chi1"]): r["full_E_full"] for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(MODELS))
    ap.add_argument("--grid", type=int, required=True, choices=list(G.GRIDS))
    ap.add_argument("--start", default="t96")
    ap.add_argument("--gate-only", action="store_true")
    a = ap.parse_args()
    mdl = MODELS[a.model]; d = json.loads(mdl["src"].read_text())
    assert d["spec"]["mr"] == MR
    ti_ship, x_ship = np.array(d["theta_insp"]), np.array(d["x_mr"])
    train = G.GRIDS[a.grid]
    for k in sorted(set(train) | set(G.T96) | set(G.V300)):
        M.CASES[k] = M.prepare(*k)
    t = time.time()
    mdl["window"]()                            # gate at the model's window (every window change before a fork)
    with get_context("fork").Pool(128) as pool:
        E = rows_E(CV.score(pool, ti_ship, MR, x_ship, G.T96)); ref = rows_E(mdl["stored"](d))
        r = np.array([E[k] / ref[k] for k in G.T96])
        print(f"gate: shipped {a.model} recomputed / stored on T96 in [{r.min():.6f}, {r.max():.6f}]", flush=True)
        if np.any(np.abs(r - 1) > 1e-3):
            sys.exit("gate failed: this scorer does not reproduce the shipped model")
    if a.gate_only:
        return

    # ---- stage 1 at the default window T = -1 (both shipped stage-1 fits ran there)
    X.set_window(-1.0)
    free_i = np.array([n not in mdl["pin"] for n in X.NAMES])
    log = []
    with get_context("fork").Pool(128) as pool:
        FF.KEYS = train; FF.SEEDS.clear()

        def valid(ti_, tm_):
            FF.SEEDS.clear(); R = FF.run(pool, "split", ti_, tm_, wide=True); FF.SEEDS.clear()
            return float(np.mean([R[k]["errs"][0] for k in FF.KEYS])) < 1.0
        if mdl["stage1"] == "origin":
            ti_s, tm_s = np.array(ORIGIN_A["theta_insp"]), np.array(ORIGIN_A["theta_mr"])
            assert np.allclose(ti_s, ti_ship), "slim__win-2.0 must carry flux_spin_split_T96_T-1's theta_insp"
            ti0 = C.start_point(ti_s, a.start, lambda th: valid(th, tm_s))
            tm0 = tm_s.copy() if a.start == "t96" else \
                C.start_point(tm_s, f"pert{int(a.start[4:]) + 100}", lambda th: valid(ti0, th))   # own seed for theta_m
            FF.SEEDS.clear()
            ti, tm, e1 = stage1_origin(pool, ti0, tm0, log)
            start1 = dict(theta_insp=list(map(float, ti0)), theta_mr=list(map(float, tm0)))
            out1 = dict(theta_mr_stage1=list(map(float, tm)))
        else:
            ti0 = C.start_point(ti_ship, a.start, lambda th: valid(th, th), free_i)
            ti, e1 = X.stage1(pool, ti0, mdl["pin"])
            start1 = dict(theta_insp=list(map(float, ti0))); out1 = {}
    print(f"stage 1 (T = -1): mean E_insp {e1:.4e} ({time.time() - t:.0f}s)", flush=True)

    # ---- stage 2 + scores at the model's window
    mdl["window"]()
    with get_context("fork").Pool(128) as pool:
        FF.KEYS = train; FF.SEEDS.clear()
        x96 = X.MR_FORMS[MR][1](ti_ship)

        def valid_x(x):
            FF.SEEDS.clear(); R = FF.run(pool, "split", ti, ("form", MR, list(x)), wide=True); FF.SEEDS.clear()
            return float(np.mean([R[k]["errs"][1] for k in FF.KEYS])) < 1.0
        x0 = C.start_point(x96, a.start, valid_x)
        FF.KEYS = train; FF.SEEDS.clear()
        x, e2 = stage2(pool, ti, x0)
        print(f"stage 2: mean E_MR {e2:.4e} ({time.time() - t:.0f}s)", flush=True)
        out = {}
        for g, ks in (("train", train), ("T96", G.T96), ("V300", G.V300)):
            out[g] = rows_E(CV.score(pool, ti, MR, x, ks))
            print(f"  {g:5s} {C.fmt(C.stats(out[g]))}", flush=True)
        shipped_v = rows_E(CV.score(pool, ti_ship, MR, x_ship, G.V300))
        print(f"  shipped T96 coefficients on V300: {C.fmt(C.stats(shipped_v))}", flush=True)
    C.write(a.model, a.grid, a.start, dict(
        names_insp=X.NAMES, pin=mdl["pin"], mr_names=X.MR_FORMS[MR][0], source=str(mdl["src"]), stage1=mdl["stage1"],
        theta_insp=list(map(float, ti)), x_mr=list(map(float, x)), start_stage1=start1, x_mr_start=list(map(float, x0)),
        theta_insp_shipped=list(map(float, ti_ship)), x_mr_shipped=list(map(float, x_ship)), **out1,
        fit=dict(mean_E_insp=e1, mean_E_MR=e2, stage1_log=log, minutes=(time.time() - t) / 60),
        stats={g: C.stats(v) for g, v in out.items()} | {"V300_shipped": C.stats(shipped_v)},
        per_case={g: {C.kstr(k): float(e) for k, e in v.items()} for g, v in out.items()}))


if __name__ == "__main__":
    main()
