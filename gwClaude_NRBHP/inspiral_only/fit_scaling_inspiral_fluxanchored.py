"""
inspiral_fluxanchored -- the `gwr_energy_fluxanchored` per-q model with the NR
comparison window RESTRICTED TO t < -200 M (merger and ringdown simply not scored).

This module also carries the shared machinery for the sibling model
`inspiral_anchored` (see `fit_scaling_inspiral_anchored.py`), which differs ONLY in
alpha's drive coordinate.  Both are per-q fits of 4 free parameters; NEITHER has any
regression layer, master polynomial or q < 3 evaluation:

    alpha(t) = a_PP * (1 + a_C * D(t))     D = flux_hat  (fluxanchored)  <- F=Edot/max
                                           D = e_oft    (anchored)      <- E, cumulative
    beta (t) = b_PP * (1 + b_E * E(t))     E = gw_remnant Eoft, both models
    tau(t)   = t0_nr + integral_{T_ANCHOR}^{t} beta dt'      phi0 analytic

WHY.  RESUME_gwremnant.md, "RESOLVED: beta's falling drift is an ARTEFACT OF THE
FULL-WINDOW L2 OBJECTIVE": the full-window fit drives b_E negative above q ~ 3.9 (beta
FALLS with time), and five separate attempts to fix it by changing the PARAMETERISATION
all failed.  The cause is the objective, not the form.  The post-merger region carries
only ~2% of the radiated energy but 32-38% of the mismatch NUMERATOR, so it dominates
the vote on beta's sign; E(t) itself does 66-82% of its range in the last 2% of the
window, so b_E has almost no inspiral leverage.

`beta_drift_tests/inspiral_only_sign.py` established at THREE mass ratios (q = 3, 5, 8)
that truncating the scored window to t <= -200 M flips b_E positive at every one --
including q=8, where the full-window fit strongly prefers falling -- and that forcing the
opposite sign then costs x1.34 to x96.  This script extends that diagnostic to the FULL
64-point [3,8] grid cached in `gw_remnant_energy_results/per_q_cache_mult.json`, to
answer two things the 3-point version cannot:

  1. Does the inspiral want a RISING beta at EVERY q in [3,8], or only at those three?
  2. Does the full-window/inspiral-only sign DISAGREEMENT set in at the q ~ 3.9 crossing
     found independently in RESUME_gwremnant.md ("WHY the drift flips at q~4", crossings
     3.799-4.003 across six unrelated models)?  Note this is close to a tautology if the
     inspiral turns out positive everywhere -- the informative half is then the converse,
     that the inspiral-only b_E has NO feature at that q.

Per q the script computes:
  1. `full`     -- fresh full-window per-q fit of the 4 params.  Fresh rather than
                   reused: the parent flux cache covers only 16 of these 64 q.
  2. `inspiral` -- refit of the same 4 params with the NR arrays cut to t_nr <= T_CUT.
                   The BHPT side is untouched -- only the NR data being compared against
                   is restricted.  Multistart over BOTH signs of b_E, every seed retained
                   as a candidate, so the reported optimum can never be worse than a seed
                   (the failure mode that invalidated `beta_drift_tests/gauge_aligned.py`
                   attempt 1).
  3. `opposite_sign` -- b_E pinned to minus the recovered sign, other 3 params
                   re-optimised, scored on the SAME truncated window: how sharply the
                   inspiral prefers its sign.

TWO CAVEATS THAT BELONG IN ANY WRITEUP.

* `T_ANCHOR = -100 M` (from fit_scaling_PN_opt_creative), so the time-map gauge anchor --
  the point where tau = t0_nr -- lies OUTSIDE a t < -200 M window.  The map is still well
  defined (built by integrating beta over the whole BHPT array), but t0_nr is then fixed
  by extrapolating the map ~100 M past the scored region.  It is a gauge constant, so it
  does not bias b_E; it does mean t0_nr here is not comparable to the parent's.

* For the FLUX model, alpha's coordinate is nearly INERT below the cut: F reaches only
  0.0097 / 0.0159 / 0.0263 (q = 3/5/8) anywhere at t < -200 M, against 0.64-0.85 at
  merger, so `a_C` sits in a near-flat direction and the fit is effectively 3-parameter.
  Its recovered values (and their sign, which flips vs the full-window fit) must NOT be
  read physically.  E(t) by contrast retains 23.1/32.2/41.4% of its range below the cut,
  so b_E IS constrained -- and so is `a_C` for the `anchored` sibling, which is exactly
  why that comparison is worth running.

Usage:
    python fit_scaling_inspiral_fluxanchored.py --check           # 3-q validation
    python fit_scaling_inspiral_fluxanchored.py --nproc 24        # full 64-q grid
    python fit_scaling_inspiral_anchored.py     --nproc 24        # E-driven alpha
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
sys.path.insert(0, str(PARENT))

import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_gwr_energy_global as GG

SEED_CACHE = G.RESULTS_DIR / "per_q_cache_mult.json"

DEFAULT_T_CUT = -200.0
SRC_STRIDE, NR_STRIDE = 6, 10

# alpha's drive coordinate is the ONLY difference between the two models.
MODELS = {
    "fluxanchored": {"drive": "flux_hat", "parent": "gwr_energy_fluxanchored"},
    "anchored":     {"drive": "e_oft",    "parent": "gwr_energy_anchored"},
}

# beta_drift_tests/inspiral_only_sign.py, t_cut = -200 M, flux alpha.  Its `full` values
# come from the 16-point flux cache; ours are refit, so small differences there are
# expected -- the inspiral-only b_E is the number that must reproduce.
REF = {
    3.0: {"full_b_E": +1.2907, "insp_b_E": +3.1677, "cost": 96.2},
    5.0: {"full_b_E": -0.8187, "insp_b_E": +2.0721, "cost": 27.6},
    8.0: {"full_b_E": -1.4777, "insp_b_E": +0.3576, "cost": 1.34},
}

# Independently-measured crossing of full-window b_E through zero, six models
# (RESUME_gwremnant.md): 3.799 ... 4.003, mean 3.92.
FULL_WINDOW_CROSSING = 3.92


def paths_for(model: str):
    d = ROOT / f"inspiral_{model}_results"
    d.mkdir(exist_ok=True)
    return d / "per_q_cache.json", d / "summary.json"


# ---------------------------------------------------------------------------

def full_grid() -> list[float]:
    return sorted(r["q"] for r in json.loads(SEED_CACHE.read_text()).values())


def mult_seeds() -> dict:
    """Cached full-window `mult` per-q solutions: {q: [a_PP, a_E, b_PP, b_E, t0]}."""
    out = {}
    for v in json.loads(SEED_CACHE.read_text()).values():
        p = v["params"]
        out[round(float(v["q"]), 10)] = [p[0], p[1], p[2], p[3], p[4]]
    return out


def truncate(case: dict, t_cut: float) -> dict:
    """Copy of `case` with the NR-side arrays cut at t_cut.  BHPT side untouched."""
    c = dict(case)
    for tk, hk in (("t_nr", "h_nr"), ("fit_t_nr", "fit_h_nr")):
        t = np.asarray(case[tk], float)
        m = t <= t_cut
        c[tk] = t[m]
        c[hk] = np.asarray(case[hk])[m]
    return c


def model_shape(p, t_bhpt, losses, drive):
    """alpha on `drive`, beta always on e_oft.  Generalises FL.model_shape."""
    a_pp, a_c, b_pp, b_e = p
    alpha = a_pp * (1.0 + a_c * losses[drive])
    beta = b_pp * (1.0 + b_e * losses["e_oft"])
    if np.any(beta <= 0):
        return alpha, None, -1.0
    bc = G.creative.cumulative_trapezoid(beta, t_bhpt)
    tau_shape = bc - float(np.interp(G.T_ANCHOR, t_bhpt, bc))
    return alpha, tau_shape, float(np.min(np.diff(tau_shape)))


def mism(p, case, t0_ref, drive, key="fit", span=120.0):
    if key == "fit":
        tb, hb, tn, hn, los = (case["fit_t_bhpt"], case["fit_h_bhpt"],
                               case["fit_t_nr"], case["fit_h_nr"], case["fit_losses"])
    else:
        tb, hb, tn, hn, los = (case["t_bhpt"], case["h_bhpt"],
                               case["t_nr"], case["h_nr"], case["losses"])
    a, tau, dmin = model_shape(p, tb, los, drive)
    if dmin <= 0:
        return 50.0 + 1e3 * (-dmin), t0_ref
    f = lambda t0: GG.err_at_t0(t0, a, tau, hb, tn, hn)
    r = minimize_scalar(f, bounds=(t0_ref - span, t0_ref + span), method="bounded",
                        options={"xatol": 1e-3, "maxiter": 80})
    return float(r.fun), float(r.x)


def perq_fit_energy(case, pm, t0_seed):
    """Full-window per-q fit with the E-driven alpha (`mult` form), multistarted from
    the cached mult solution.  Mirrors FL.perq_fit, which hardcodes flux-scale seeds."""
    best = None
    for sa in (1.0, 0.97, 1.03):
        for se in (1.0, 0.9, 1.1):
            p0 = np.array([pm[0] * sa, pm[1] * se, pm[2], pm[3]], float)
            st = {"t0": t0_seed}

            def obj(p):
                e, t0 = mism(p, case, st["t0"], "e_oft", "fit")
                st["t0"] = t0
                return e
            r = minimize(obj, p0, method="Nelder-Mead",
                         options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-10})
            if best is None or r.fun < best[0]:
                best = (float(r.fun), r.x.copy(), st["t0"])
    return best


def fit_4param(case, pref, t0_ref, drive, be_seeds, fix_be=None):
    """Optimise the 4 params (or 3, with b_E fixed).  Seeds are kept as candidates so
    the returned best can never be worse than a seed."""
    cands, best_seed = [], np.inf
    for be in ([fix_be] if fix_be is not None else be_seeds):
        for s in (1.0, 0.98):
            if fix_be is None:
                x0 = np.array([pref[0] * s, pref[1], pref[2], be], float)
                obj = lambda p: mism(p, case, t0_ref, drive)[0]
            else:
                x0 = np.array([pref[0] * s, pref[1], pref[2]], float)
                obj = lambda u: mism(np.r_[u, fix_be], case, t0_ref, drive)[0]
            e0 = obj(x0)
            cands.append((e0, np.asarray(x0, float).copy()))
            best_seed = min(best_seed, e0)
            r = minimize(obj, x0, method="Nelder-Mead",
                         options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-12})
            cands.append((float(r.fun), np.asarray(r.x, float).copy()))
    e, x = min(cands, key=lambda c: c[0])
    warned = e > best_seed + 1e-15
    p = np.r_[x, fix_be] if fix_be is not None else x
    return e, p, warned


def fit_one_q(q, t_cut, model, pm, t0_seed):
    drive = MODELS[model]["drive"]
    case = FL.add_flux(G.load_case(q, SRC_STRIDE, NR_STRIDE))

    # 1. full-window per-q optimum.  The flux branch delegates to the parent's own
    #    perq_fit so the validated numbers are reproduced bit-for-bit.
    if model == "fluxanchored":
        e_full, p_full, t0_full = FL.perq_fit(q, case, t0_seed, pm[3])
    else:
        e_full, p_full, t0_full = perq_fit_energy(case, pm, t0_seed)

    # 2. inspiral-only fit; both signs seeded so neither is favoured a priori
    tcase = truncate(case, t_cut)
    seeds = [p_full[3], -abs(p_full[3]) if p_full[3] != 0 else -1.0, 0.0, 1.0, 3.0]
    e_insp, p_insp, w1 = fit_4param(tcase, p_full, t0_full, drive, seeds)

    # 3. cost of forcing the opposite sign, on the SAME inspiral-only window
    opp = -abs(p_insp[3]) if p_insp[3] > 0 else abs(p_insp[3])
    e_opp, _, w2 = fit_4param(tcase, p_full, t0_full, drive, [], fix_be=opp)

    los = case["losses"]
    e_tot = float(los["e_oft"][-1] - los["e_oft"][0])
    e_at_cut = float(np.interp(t_cut, case["t_bhpt"], los["e_oft"]))
    m = case["t_bhpt"] <= t_cut
    drive_span = float(np.max(los[drive][m]) - np.min(los[drive][m]))

    return {
        "q": q, "nu": G.get_nu(q), "t_cut": t_cut, "model": model,
        "full": {"err": e_full, "params": p_full.tolist(), "b_E": float(p_full[3])},
        "inspiral": {"err": e_insp, "params": p_insp.tolist(),
                     "b_E": float(p_insp[3])},
        "opposite_sign": {"err": e_opp, "b_E": float(opp)},
        "cost_ratio_opposite": float(e_opp / e_insp) if e_insp > 0 else float("nan"),
        "n_nr_inspiral": int(len(tcase["t_nr"])), "n_nr_total": int(len(case["t_nr"])),
        "frac_window": len(tcase["t_nr"]) / len(case["t_nr"]),
        "E_frac_at_cut": e_at_cut / e_tot,
        "drive_span_below_cut": drive_span,
        "beta_rise_to_cut": float(p_insp[3]) * e_at_cut,
        "b_PP_over_X1_6_5": float(p_insp[2]) / (q / (1.0 + q)) ** 1.2,
        "seed_warning": bool(w1 or w2),
    }


_CTX = {}


def _init(t_cut, model, pm, t0s):
    _CTX.update(t_cut=t_cut, model=model, pm=pm, t0s=t0s)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def _worker(q):
    k = round(q, 10)
    return fit_one_q(q, _CTX["t_cut"], _CTX["model"], _CTX["pm"][k], _CTX["t0s"][k])


# ---------------------------------------------------------------------------

def report(rows, t_cut, model):
    b_full = np.array([r["full"]["b_E"] for r in rows])
    b_insp = np.array([r["inspiral"]["b_E"] for r in rows])
    qs = np.array([r["q"] for r in rows])
    cost = np.array([r["cost_ratio_opposite"] for r in rows])
    ratio = np.array([r.get("b_PP_over_X1_6_5", np.nan) for r in rows])

    n_rising = int(np.sum(b_insp > 0))
    disagree = np.sign(b_full) != np.sign(b_insp)

    cross = float("nan")
    sgn = np.sign(b_full)
    flips = np.where(sgn[:-1] != sgn[1:])[0]
    if len(flips):
        i = flips[0]
        cross = float(np.interp(0.0, [b_full[i + 1], b_full[i]], [qs[i + 1], qs[i]]))

    print(f"\n{'q':>8} {'full b_E':>10} {'insp b_E':>10} {'sign':>8} {'full err':>11} "
          f"{'insp err':>11} {'opp x':>8} {'d beta':>8} {'agree':>6}")
    for r in rows:
        s = ("RISING" if r["inspiral"]["b_E"] > 0
             else "falling" if r["inspiral"]["b_E"] < 0 else "flat")
        ag = "" if np.sign(r["full"]["b_E"]) == np.sign(r["inspiral"]["b_E"]) else "DIFF"
        print(f"{r['q']:>8.4f} {r['full']['b_E']:>+10.4f} {r['inspiral']['b_E']:>+10.4f} "
              f"{s:>8} {r['full']['err']:>11.4e} {r['inspiral']['err']:>11.4e} "
              f"{r['cost_ratio_opposite']:>8.2f} "
              f"{r.get('beta_rise_to_cut', float('nan'))*100:>+7.3f}% {ag:>6}")

    n_mono = int(np.sum(np.diff(b_insp) >= 0))
    print(f"\n=== SUMMARY  model=inspiral_{model}  t_cut={t_cut:g} M  {len(rows)} q ===")
    print(f"inspiral wants RISING beta at {n_rising}/{len(rows)} q "
          f"({n_rising/len(rows)*100:.1f}%)")
    print(f"full-window vs inspiral-only sign DISAGREEMENT at "
          f"{int(disagree.sum())}/{len(rows)} q")
    if np.isfinite(cross):
        print(f"full-window b_E crosses zero at q = {cross:.3f}  "
              f"(independent multi-model value {FULL_WINDOW_CROSSING})")
    if disagree.any():
        print(f"disagreement region: q in [{qs[disagree].min():.3f}, "
              f"{qs[disagree].max():.3f}]")
    print(f"inspiral b_E: {b_insp[0]:+.4f} (q={qs[0]:g}) -> {b_insp[-1]:+.4f} "
          f"(q={qs[-1]:g}); NOT strictly monotone: {n_mono}/{len(rows)-1} steps rise")
    print(f"beta rise over fitted window: {rows[0]['beta_rise_to_cut']*100:+.3f}% "
          f"(q={qs[0]:g}) -> {rows[-1]['beta_rise_to_cut']*100:+.3f}% (q={qs[-1]:g})")
    print(f"opposite-sign cost: median x{np.median(cost):.2f}, min x{cost.min():.2f}, "
          f"max x{cost.max():.2f}")
    print(f"mismatch median: inspiral {np.median([r['inspiral']['err'] for r in rows]):.4e}"
          f"   full-window {np.median([r['full']['err'] for r in rows]):.4e}")
    print(f"b_PP / X1^(6/5): mean |dev| {np.nanmean(np.abs(ratio-1))*100:.3f}%  "
          f"max {np.nanmax(np.abs(ratio-1))*100:.3f}%")
    print(f"alpha drive span below cut: {rows[0]['drive_span_below_cut']:.4f} "
          f"(q={qs[0]:g}) .. {rows[-1]['drive_span_below_cut']:.4f} (q={qs[-1]:g})")
    nw = sum(r.get("seed_warning", False) for r in rows)
    if nw:
        print(f"!! {nw} q where the optimiser never beat its best seed -- inspect")

    return {
        "model": f"inspiral_{model}", "t_cut": t_cut, "n_q": len(rows),
        "n_inspiral_rising": n_rising,
        "frac_inspiral_rising": n_rising / len(rows),
        "n_disagree_with_full_window": int(disagree.sum()),
        "q_full_window_b_E_zero_crossing": cross,
        "q_disagree_min": float(qs[disagree].min()) if disagree.any() else None,
        "q_disagree_max": float(qs[disagree].max()) if disagree.any() else None,
        "b_E_inspiral": {"at_q_min": float(b_insp[0]), "at_q_max": float(b_insp[-1]),
                         "n_nonmonotone_steps": n_mono},
        "beta_rise_to_cut": {"at_q_min": rows[0]["beta_rise_to_cut"],
                             "at_q_max": rows[-1]["beta_rise_to_cut"]},
        "cost_ratio_opposite": {"median": float(np.median(cost)),
                                "min": float(cost.min()), "max": float(cost.max())},
        "err_median": {"full": float(np.median([r["full"]["err"] for r in rows])),
                       "inspiral": float(np.median([r["inspiral"]["err"] for r in rows]))},
        "b_PP_over_X1_6_5": {"mean_abs_dev": float(np.nanmean(np.abs(ratio - 1))),
                             "max_abs_dev": float(np.nanmax(np.abs(ratio - 1)))},
        "alpha_drive_span_below_cut": {"at_q_min": rows[0]["drive_span_below_cut"],
                                       "at_q_max": rows[-1]["drive_span_below_cut"]},
        "n_seed_warnings": int(nw),
    }


def check_against_ref(rows):
    print("\n=== VALIDATION vs beta_drift_tests/inspiral_only_sign.py (t_cut=-200) ===")
    print(f"{'q':>5} {'quantity':>12} {'this run':>12} {'reference':>12} {'delta':>10}")
    ok = True
    for r in rows:
        q = round(r["q"], 4)
        if q not in REF:
            continue
        for lab, got, exp in (("full b_E", r["full"]["b_E"], REF[q]["full_b_E"]),
                              ("insp b_E", r["inspiral"]["b_E"], REF[q]["insp_b_E"]),
                              ("opp cost", r["cost_ratio_opposite"], REF[q]["cost"])):
            d = got - exp
            flag = "" if abs(d) <= max(0.15 * abs(exp), 0.05) else "  <-- CHECK"
            ok = ok and not flag
            print(f"{q:>5g} {lab:>12} {got:>12.4f} {exp:>12.4f} {d:>+10.4f}{flag}")
    print("validation: " + ("consistent with the 3-point reference" if ok else
                            "DIFFERS -- investigate before trusting the grid"))
    return ok


def main(default_model="fluxanchored"):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default=default_model)
    ap.add_argument("--t-cut", type=float, default=DEFAULT_T_CUT)
    ap.add_argument("--q", type=float, nargs="+", default=None,
                    help="subset of q (nearest grid point); default full 64-q grid")
    ap.add_argument("--check", action="store_true",
                    help="only q=3,5,8, and compare against the documented values")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--force", action="store_true",
                    help="refit the selected q even if cached (other q are preserved)")
    args = ap.parse_args()

    cache_path, summary_path = paths_for(args.model)
    grid = full_grid()
    if args.check:
        q_list = [min(grid, key=lambda g: abs(g - t)) for t in (3.0, 5.0, 8.0)]
    elif args.q is not None:
        q_list = sorted({min(grid, key=lambda g: abs(g - t)) for t in args.q})
    else:
        q_list = grid

    pm = mult_seeds()
    t0s = {k: v[4] for k, v in pm.items()}

    # NB: --force refits only the SELECTED q; it must not wipe the cache file, or a
    # `--check --force` would destroy a completed 64-q grid.
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    key_of = lambda q: f"{q:.10f}|{args.t_cut:.1f}"
    todo = q_list if args.force else [q for q in q_list if key_of(q) not in cache]

    print(f"[inspiral_{args.model}] alpha drive = {MODELS[args.model]['drive']}, "
          f"t_cut={args.t_cut:g} M, {len(q_list)} q "
          f"({len(q_list)-len(todo)} cached, {len(todo)} to fit), nproc={args.nproc}",
          flush=True)

    t_start = time.time()
    if todo:
        def emit(n, row):
            cache[key_of(row["q"])] = row
            cache_path.write_text(json.dumps(cache, indent=1))
            s = "RISING" if row["inspiral"]["b_E"] > 0 else "falling"
            print(f"  [{n:>2}/{len(todo)}] q={row['q']:<8.4f} "
                  f"full b_E={row['full']['b_E']:+7.4f} "
                  f"insp b_E={row['inspiral']['b_E']:+7.4f} {s:<8} "
                  f"opp x{row['cost_ratio_opposite']:.2f}", flush=True)

        if args.nproc > 1:
            with Pool(args.nproc, initializer=_init,
                      initargs=(args.t_cut, args.model, pm, t0s)) as pool:
                for n, row in enumerate(pool.imap_unordered(_worker, todo), 1):
                    emit(n, row)
        else:
            _init(args.t_cut, args.model, pm, t0s)
            for n, q in enumerate(todo, 1):
                emit(n, _worker(q))
    print(f"[inspiral_{args.model}] fitting took {time.time()-t_start:.0f}s", flush=True)

    rows = [cache[key_of(q)] for q in q_list]
    summary = report(rows, args.t_cut, args.model)
    if args.model == "fluxanchored" and abs(args.t_cut + 200.0) < 1e-9:
        summary["validation_vs_3point_ref"] = check_against_ref(rows)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {cache_path}\nwrote {summary_path}")


if __name__ == "__main__":
    main()
