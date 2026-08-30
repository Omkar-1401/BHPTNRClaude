"""
merger-ringdown-only per-q fits -- the exact mirror of inspiral_only/, and a test of the
QNM anchor on its home turf.

The NR comparison window is cut to t_nr >= T_CUT, so the INSPIRAL is not scored.  With
T_CUT = -200 M this is the exact complement of `fit_scaling_inspiral_fluxanchored.py`:
the two windows partition the same objective, which is what makes the pair informative.

TWO BETA VARIANTS, and the second is the point:

  --anchor x1   beta(t) = b_PP * (1 + b_E * Ehat(t))
                the inspiral convention: b_PP is beta's EARLY value and measures as
                X1^(6/5).  Kept as the control.  4 free parameters per q.

  --anchor qnm  beta(t) = B1(q) * (1 + b_E * (Ehat(t) - 1))
                B1 = W_Schw*Mf(q)/omega_220(chi_f(q)) from gwModels.remnants.gwModelRemS,
                the SAME derived ringdown anchor `fit_scaling_gwr_betaqnm.py` uses and
                verified there to 0.3-0.9% against a direct omega_pp/omega_NR measurement.
                Ehat -> 1 as E saturates, so beta -> B1 EXACTLY at late times, with
                b_E controlling only the approach.  beta's LEVEL is therefore derived,
                not fitted: 3 free parameters per q.

WHY DROP THE INSPIRAL ANCHOR.  X1^(6/5) is an EARLY-INSPIRAL statement -- RESUME records
beta(start) matching it to 0.08-0.46% at every q.  On a window that never sees the
inspiral it constrains nothing that is being scored, while the QNM value is the anchor
that actually applies in the region being fitted.  The mirror asymmetry is deliberate:
the inspiral-only runs kept the anchor that applied to THEIR window, and this keeps the
one that applies to this one.

THE QUESTION THIS ANSWERS.  RESUME, "WHY the drift flips at q~4": the full-window fits
land beta's terminal value +3.95 / +0.34 / -1.95 / -3.79 / -3.68 / -2.08 % away from B1
at q = 2 / 2.5 / 3 / 4 / 5 / 8, and "B1 appears NOWHERE in mult/anchored/fluxanchored, so
nothing makes the terminal beta land on it."  Here it is imposed, and only the region
where it applies is scored -- so the mismatch difference between `x1` and `qnm` prices
the anchor on its own ground.  `gwr_betaqnm` priced it at 6.2x, but that was full-window
with BOTH anchors imposed and a 3-leg ramp; this isolates it.

ALPHA KEEPS A FREE PREFACTOR.  There is no derived ringdown anchor for alpha, so
alpha = a_RD * (1 + a_C * D(t)) with a_RD free in both variants.  Do not read a_RD as
X1^(6/5): on this window it is a late-time amplitude, a different quantity.

CAVEATS THAT BELONG IN ANY WRITEUP.
  * The MR window carries only ~2% of the radiated energy but 32-38% of the FULL-window
    mismatch numerator.  Mismatches here are normalised to the MR power alone, so they
    are NOT comparable to full-window numbers or to the inspiral-only ones.
  * T_ANCHOR = -100 M lies INSIDE this window for T_CUT = -200 (unlike the inspiral runs,
    where it was outside), so t0_nr is better determined here than there.
  * At T_CUT = -200 the window holds ~5.9% of the NR samples -- the complement of the
    inspiral runs' 94.1%.

Usage:
    python fit_scaling_mr_only.py --check --anchor qnm --nproc 3
    python fit_scaling_mr_only.py --anchor qnm --nproc 24
    python fit_scaling_mr_only.py --anchor x1 --t-cut -50 --nproc 24
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_gwr_energy_global as GG
import gwModels

RESULTS = ROOT / "mr_only_results"
RESULTS.mkdir(exist_ok=True)
SEED_CACHE = G.RESULTS_DIR / "per_q_cache_mult.json"

DEFAULT_T_CUT = -200.0
SRC_STRIDE, NR_STRIDE = 6, 10

W_SCHW = 0.3737
QNM = json.loads((ROOT / "pn_anchored_results" / "coeffs.json").read_text())["qnm"]

DRIVES = {"e_oft": "e_oft", "flux_hat": "flux_hat"}


def B1_of(q):
    """Derived ringdown anchor: W_Schw*Mf/omega_220(chi_f).  Same as gwr_betaqnm."""
    r = gwModels.remnants.gwModelRemS(q, 0.0, 0.0)
    Mf, chif = float(r[0]), float(r[1])
    return W_SCHW * Mf / (QNM["F1"] + QNM["F2"] * (1.0 - chif) ** QNM["F3"])


def add_ehat(case):
    e = np.asarray(case["losses"]["e_oft"], float)
    tot = float(e[-1] - e[0])
    case["losses"]["ehat"] = (e - e[0]) / tot
    ef = np.asarray(case["fit_losses"]["e_oft"], float)
    case["fit_losses"]["ehat"] = (ef - e[0]) / tot
    return case


def truncate_mr(case, t_cut):
    """Keep t_nr >= t_cut.  Mirror of the inspiral runs' truncate()."""
    c = dict(case)
    for tk, hk in (("t_nr", "h_nr"), ("fit_t_nr", "fit_h_nr")):
        t = np.asarray(case[tk], float)
        m = t >= t_cut
        c[tk] = t[m]
        c[hk] = np.asarray(case[hk])[m]
    return c


def model_shape(p, t_bhpt, losses, drive, anchor, B1):
    """p = [a_RD, a_C, b_PP, b_E]  (x1)   or  [a_RD, a_C, b_E]  (qnm)."""
    a_rd, a_c = p[0], p[1]
    alpha = a_rd * (1.0 + a_c * losses[drive])
    if anchor == "qnm":
        beta = B1 * (1.0 + p[2] * (losses["ehat"] - 1.0))
    else:
        beta = p[2] * (1.0 + p[3] * losses["ehat"])
    if np.any(beta <= 0) or not np.all(np.isfinite(np.r_[alpha, beta])):
        return alpha, None, -1.0
    bc = G.creative.cumulative_trapezoid(beta, t_bhpt)
    tau_shape = bc - float(np.interp(G.T_ANCHOR, t_bhpt, bc))
    return alpha, tau_shape, float(np.min(np.diff(tau_shape)))


def mism(p, case, t0_ref, drive, anchor, B1, key="fit", span=120.0):
    if key == "fit":
        tb, hb, tn, hn, los = (case["fit_t_bhpt"], case["fit_h_bhpt"],
                               case["fit_t_nr"], case["fit_h_nr"], case["fit_losses"])
    else:
        tb, hb, tn, hn, los = (case["t_bhpt"], case["h_bhpt"],
                               case["t_nr"], case["h_nr"], case["losses"])
    a, tau, dmin = model_shape(p, tb, los, drive, anchor, B1)
    if dmin <= 0:
        return 50.0 + 1e3 * (-dmin), t0_ref
    r = minimize_scalar(lambda t0: GG.err_at_t0(t0, a, tau, hb, tn, hn),
                        bounds=(t0_ref - span, t0_ref + span), method="bounded",
                        options={"xatol": 1e-3, "maxiter": 80})
    return float(r.fun), float(r.x)


def fit_one_q(q, t_cut, drive, anchor, pm, t0_seed):
    case = add_ehat(FL.add_flux(G.load_case(q, SRC_STRIDE, NR_STRIDE)))
    mcase = truncate_mr(case, t_cut)
    B1 = B1_of(q)
    e_tot = float(case["losses"]["e_oft"][-1] - case["losses"]["e_oft"][0])

    # seed from the cached mult solution: its b_E multiplies E, ours multiplies Ehat,
    # so rescale by E_tot; and for qnm, beta's level is B1 rather than b_PP.
    a_rd, a_c, b_pp, b_e = pm[0], pm[1], pm[2], pm[3] * e_tot
    if anchor == "qnm":
        p0 = np.array([a_rd, a_c, b_e], float)
    else:
        p0 = np.array([a_rd, a_c, b_pp, b_e], float)

    cands = []
    for sa in (1.0, 0.95, 1.05):
        for sb in (1.0, 0.7, 1.3):
            p = p0.copy()
            p[0] *= sa
            p[-1] *= sb
            st = {"t0": t0_seed}

            def obj(pp):
                e, t0 = mism(pp, mcase, st["t0"], drive, anchor, B1, "fit")
                st["t0"] = t0
                return e
            e0 = obj(p)
            cands.append((e0, p.copy()))
            r = minimize(obj, p, method="Nelder-Mead",
                         options={"maxiter": 3000, "xatol": 1e-6, "fatol": 1e-10})
            cands.append((float(r.fun), np.asarray(r.x, float).copy()))
    best_seed = min(c[0] for c in cands[::2])
    e_fit, p_best = min(cands, key=lambda c: c[0])

    e_mr, t0 = mism(p_best, mcase, t0_seed, drive, anchor, B1, "full")
    e_full = mism(p_best, case, t0, drive, anchor, B1, "full")[0]

    los = case["losses"]
    beta = (B1 * (1.0 + p_best[2] * (los["ehat"] - 1.0)) if anchor == "qnm"
            else p_best[2] * (1.0 + p_best[3] * los["ehat"]))
    alpha = p_best[0] * (1.0 + p_best[1] * los[drive])
    base = (q / (1.0 + q)) ** 1.2

    return {
        "q": q, "nu": G.get_nu(q), "t_cut": t_cut, "drive": drive, "anchor": anchor,
        "n_params": int(len(p_best)), "params": p_best.tolist(),
        "err_mr": e_mr, "err_full": e_full, "B1": B1,
        "beta_start": float(beta[0]), "beta_end": float(beta[-1]),
        "beta_end_over_B1": float(beta[-1] / B1),
        "beta_start_over_X1_6_5": float(beta[0]) / base,
        "beta_change_pct": float((beta[-1] / beta[0] - 1.0) * 100.0),
        "alpha_change_pct": float((alpha[-1] / alpha[0] - 1.0) * 100.0),
        "b_E": float(p_best[2] if anchor == "qnm" else p_best[3]),
        "n_nr_mr": int(len(mcase["t_nr"])), "n_nr_total": int(len(case["t_nr"])),
        "frac_window": len(mcase["t_nr"]) / len(case["t_nr"]),
        "seed_warning": bool(e_fit > best_seed + 1e-15),
    }


_CTX = {}


def _init(t_cut, drive, anchor, pm, t0s):
    _CTX.update(t_cut=t_cut, drive=drive, anchor=anchor, pm=pm, t0s=t0s)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def _worker(q):
    k = round(q, 10)
    return fit_one_q(q, _CTX["t_cut"], _CTX["drive"], _CTX["anchor"],
                     _CTX["pm"][k], _CTX["t0s"][k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", choices=("qnm", "x1"), default="qnm")
    ap.add_argument("--drive", choices=sorted(DRIVES), default="e_oft")
    ap.add_argument("--t-cut", type=float, default=DEFAULT_T_CUT)
    ap.add_argument("--q", type=float, nargs="+", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seeds = json.loads(SEED_CACHE.read_text())
    grid = sorted(r["q"] for r in seeds.values())
    pm = {round(float(v["q"]), 10): v["params"][:4] for v in seeds.values()}
    t0s = {round(float(v["q"]), 10): v["params"][4] for v in seeds.values()}

    if args.check:
        q_list = [min(grid, key=lambda g: abs(g - t)) for t in (3.0, 5.0, 8.0)]
    elif args.q is not None:
        q_list = sorted({min(grid, key=lambda g: abs(g - t)) for t in args.q})
    else:
        q_list = grid

    tag = f"{args.anchor}_{args.drive}_tcut{abs(args.t_cut):.0f}"
    cache_path = RESULTS / f"per_q_cache_{tag}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    key_of = lambda q: f"{q:.10f}"
    todo = q_list if args.force else [q for q in q_list if key_of(q) not in cache]

    npar = 3 if args.anchor == "qnm" else 4
    print(f"[mr_only] anchor={args.anchor} ({npar} params/q), drive={args.drive}, "
          f"window t_nr >= {args.t_cut:g} M")
    print(f"[mr_only] {len(q_list)} q ({len(q_list)-len(todo)} cached, {len(todo)} to "
          f"fit), nproc={args.nproc}", flush=True)

    t_start = time.time()
    if todo:
        def emit(n, row):
            cache[key_of(row["q"])] = row
            cache_path.write_text(json.dumps(cache, indent=1))
            print(f"  [{n:>2}/{len(todo)}] q={row['q']:<7.4f} E_mr={row['err_mr']:.4e}  "
                  f"beta_end/B1={row['beta_end_over_B1']:.4f}  "
                  f"b_E={row['b_E']:+.4f}  dbeta={row['beta_change_pct']:+6.2f}%",
                  flush=True)

        if args.nproc > 1:
            with Pool(args.nproc, initializer=_init,
                      initargs=(args.t_cut, args.drive, args.anchor, pm, t0s)) as pool:
                for n, row in enumerate(pool.imap_unordered(_worker, todo), 1):
                    emit(n, row)
        else:
            _init(args.t_cut, args.drive, args.anchor, pm, t0s)
            for n, q in enumerate(todo, 1):
                emit(n, _worker(q))
    print(f"[mr_only] fitting took {time.time()-t_start:.0f}s", flush=True)

    rows = [cache[key_of(q)] for q in q_list]
    err = np.array([r["err_mr"] for r in rows])
    bE = np.array([r["b_E"] for r in rows])
    ratio = np.array([r["beta_end_over_B1"] for r in rows])
    x1r = np.array([r["beta_start_over_X1_6_5"] for r in rows])

    print(f"\n{'q':>8} {'E_mr':>11} {'E_full':>11} {'B1':>8} {'beta_end/B1':>12} "
          f"{'beta_start/X1':>14} {'b_E':>9} {'d beta':>9}")
    for r in rows:
        print(f"{r['q']:>8.4f} {r['err_mr']:>11.4e} {r['err_full']:>11.4e} "
              f"{r['B1']:>8.5f} {r['beta_end_over_B1']:>12.5f} "
              f"{r['beta_start_over_X1_6_5']:>14.5f} {r['b_E']:>+9.4f} "
              f"{r['beta_change_pct']:>+8.2f}%")

    print(f"\n=== mr_only  anchor={args.anchor}  drive={args.drive}  "
          f"t_cut={args.t_cut:g} M  {len(rows)} q ===")
    print(f"MR-window mismatch: median {np.median(err):.4e}  min {err.min():.4e}  "
          f"max {err.max():.4e}")
    print(f"beta rises at {int((bE > 0).sum())}/{len(rows)} q "
          f"(b_E > 0; on this parameterisation beta rises toward the anchor)")
    print(f"beta_end / B1: {ratio.min():.5f} .. {ratio.max():.5f}"
          + ("  (imposed)" if args.anchor == "qnm" else "  (FREE -- this is the test)"))
    print(f"beta_start / X1^(6/5): {x1r.min():.5f} .. {x1r.max():.5f}"
          + ("  (FREE -- inspiral anchor NOT imposed)" if args.anchor == "qnm"
             else "  (imposed)"))
    print(f"window holds {rows[0]['frac_window']*100:.1f}% of the NR samples")
    nw = sum(r["seed_warning"] for r in rows)
    if nw:
        print(f"!! {nw} q where the optimiser never beat its seed -- inspect")

    summary = {
        "model": "mr_only", "anchor": args.anchor, "drive": args.drive,
        "t_cut": args.t_cut, "n_q": len(rows), "n_params_per_q": npar,
        "err_mr": {"median": float(np.median(err)), "min": float(err.min()),
                   "max": float(err.max())},
        "n_b_E_positive": int((bE > 0).sum()),
        "beta_end_over_B1": {"min": float(ratio.min()), "max": float(ratio.max())},
        "beta_start_over_X1_6_5": {"min": float(x1r.min()), "max": float(x1r.max())},
        "frac_window": rows[0]["frac_window"], "n_seed_warnings": int(nw),
    }
    (RESULTS / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {cache_path}\nwrote {RESULTS / f'summary_{tag}.json'}")


if __name__ == "__main__":
    main()
