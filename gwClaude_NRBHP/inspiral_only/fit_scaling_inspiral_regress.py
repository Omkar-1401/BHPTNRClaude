"""
THE FITTED (nu-regressed) inspiral-only models -- the master layer the two per-q
diagnostic scripts deliberately lack.

`fit_scaling_inspiral_fluxanchored.py` and `fit_scaling_inspiral_anchored.py` produce
PER-Q optima only: 4 free parameters at each of 64 mass ratios, no regression, no master
polynomial, nothing evaluable off the grid.  This script adds the regression layer, so
there is an actual model:

    alpha_PP(q) = X1^(6/5) * (1 + c0*nu + c1*nu^2)          INHERITED from the parent
    alpha_C(q)  = A0 / nu                  (E drive)        INHERITED from the parent
                = A0*nu + A1*nu^2          (F drive)        INHERITED from the parent
    beta_PP(q)  = X1^(6/5) * (1 + b*nu)                     1 coef  [derived base]
    P(q)        = P0 + P1*nu   ,  b_E = P / beta_PP         2 coef  [empirical]

    -> 3 FREE coefficients, all of them beta's

ALPHA IS NOT FITTED ON THE INSPIRAL.  It must fall monotonically at every q, and the
truncated window cannot deliver that -- E(t) does only 21-40% of its range below the cut
and alpha's fall lives in the other 60-79%, so the objective has no leverage on alpha_C
and returns a wrong-signed one (+1.89 at q=3 through zero to -0.18 at q=8, against
-5.67..-10.03 full-window).  Constraining alpha_C <= 0 yields the boundary, a FLAT alpha,
which is still not a falling one.  So alpha is taken wholesale from the parent
(`gwr_energy_anchored` / `gwr_energy_fluxanchored`) where it IS measurable, and the
inspiral is used only for what it constrains: beta's sign.  Full argument and the
supporting numbers are in `parent_alpha`'s docstring and in scaling_inspiral_anchored.md.

`--fit-alpha` restores the old behaviour (6/7 coefficients, alpha refitted here) and is
kept as a DIAGNOSTIC of that failure only; its output goes to `coeffs*_fitalpha.json` so
it can never be mistaken for the model.

beta's forms are unchanged from the parents -- see `fit_scaling_gwr_energy_anchored.py`,
whose `seed_theta`/`split`/`params_at` this mirrors.  Keeping them fixed is the point: it
isolates the objective, which is the whole subject of this folder.

WHAT THIS BUYS, AND WHY IT IS THE INTERESTING FIT.  `P` is regressed from
`P = beta_PP * b_E` and the inspiral-only `b_E` is POSITIVE at all 64 q, so `P > 0`
across the training range by construction of the data -- and a line through positive
data stays positive.  `beta` therefore RISES at every q, which is RESUME_gwremnant.md's
open item 6 ("make beta(t) rise at every q, with no post-merger correction") reached
without a gate, a switch, a second drive term or a post-merger branch.  The script prints
`P` and `b_E` at every reported q, including below the training range, so the sign claim
is checkable rather than asserted.

SCORING.  Two numbers per q, both reported, never mixed:
  * `insp` -- mismatch on the TRUNCATED window (t_nr < t_cut).  This is the objective the
    per-q data was fitted to and the one this model is entitled to be judged on.
  * `full` -- mismatch on the whole window, same parameters.  Reported for orientation
    ONLY: it includes merger and ringdown, which this model never saw and does not
    describe.  It will look bad.  That is not a defect, it is the definition.

Do not compare either column against CLAUDE.md's model table, which is full-window.

Usage:
    python fit_scaling_inspiral_regress.py --model anchored
    python fit_scaling_inspiral_regress.py --model anchored --global --maxiter 80
    python fit_scaling_inspiral_regress.py --model fluxanchored --global
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PARENT))
warnings.filterwarnings("ignore")

import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_inspiral_fluxanchored as IN

IN_Q = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
LOW_Q = (2.75, 2.5, 2.25, 2.0)

nu_of = lambda q: np.asarray(q, float) / (1.0 + np.asarray(q, float)) ** 2
X1_of = lambda q: np.asarray(q, float) / (1.0 + np.asarray(q, float))
base_of = lambda q: X1_of(q) ** 1.2              # Newtonian chirp factor

# alpha_C's nu-form per drive, matching each parent exactly.
#   E drive: A0/nu      -- 1/nu is a normalisation convention (only alpha_C*E is physical)
#   F drive: A0*nu+A1*nu^2 -- PP-anchored by construction, F being peak-normalised
NA = {"anchored": 1, "fluxanchored": 2}


def results_dir(model):
    d = ROOT / f"inspiral_{model}_results"
    d.mkdir(exist_ok=True)
    return d


def load_per_q(model, t_cut):
    """The inspiral-only per-q optima, as {q, nu, alpha_PP, alpha_C, beta_PP, P}."""
    path, _ = IN.paths_for(model)
    rows = [r for r in json.loads(path.read_text()).values()
            if abs(r["t_cut"] - t_cut) < 1e-9]
    if not rows:
        sys.exit(f"{path} holds no rows at t_cut={t_cut:g}")
    rows.sort(key=lambda r: r["q"])
    q = np.array([r["q"] for r in rows], float)
    P4 = np.array([r["inspiral"]["params"] for r in rows], float)
    return {"q": q, "nu": nu_of(q), "alpha_PP": P4[:, 0], "alpha_C": P4[:, 1],
            "beta_PP": P4[:, 2], "b_E": P4[:, 3], "P": P4[:, 2] * P4[:, 3],
            "err": np.array([r["inspiral"]["err"] for r in rows], float)}


def parent_alpha(model):
    """alpha's coefficients from the PARENT full-window fit, which is the only window
    where alpha's coupling is measurable.

    WHY alpha IS NOT FITTED HERE.  alpha must fall monotonically at every q: the ppBHPT
    overestimates the amplitude and the correction grows as the binary becomes more
    relativistic, so alpha moves further below 1 with time.  The inspiral cannot deliver
    that.  E(t) does only 21-40% of its range below the cut and alpha's fall happens in
    the other 60-79%, so the truncated objective has no leverage on alpha_C: the per-q
    inspiral fits return +1.89 at q=3 falling THROUGH ZERO to -0.18 at q=8, against
    -5.67 .. -10.03 for the same waveforms scored full-window.  Constraining alpha_C <= 0
    does not fix it either -- where the inspiral prefers positive the constrained optimum
    sits on the boundary at alpha_C -> 0, i.e. a FLAT alpha, which is still not a falling
    one.  So alpha is taken wholesale from the parent, where it is negative at all 64 q,
    monotone in q, and reproduces the per-q full-window optima to ~1%.

    This leaves the inspiral objective doing only what it can do: fixing beta's sign.
    """
    parent = {"anchored": "gwr_energy_anchored",
              "fluxanchored": "gwr_energy_fluxanchored"}[model]
    j = json.loads((PARENT / f"{parent}_results" / "coeffs.json").read_text())
    return np.asarray(j["c"], float), np.asarray(j["A"], float)


def seed_theta(d, model, alpha_fixed=None):
    """Least-squares seed for each form, exactly as the parents build theirs."""
    nu, bs = d["nu"], base_of(d["q"])
    if alpha_fixed is not None:                      # only beta is free: [b, P0, P1]
        b = np.linalg.lstsq(nu[:, None], d["beta_PP"] / bs - 1.0, rcond=None)[0]
        P = np.polyfit(nu, d["P"], 1)[::-1]
        return np.concatenate([b, P])
    c = np.linalg.lstsq(np.vstack([nu, nu ** 2]).T, d["alpha_PP"] / bs - 1.0,
                        rcond=None)[0]
    if model == "anchored":
        A = np.array([float(np.mean(d["alpha_C"] * nu))])           # A0/nu
    else:
        A = np.linalg.lstsq(np.vstack([nu, nu ** 2]).T, d["alpha_C"],
                            rcond=None)[0]                          # A0*nu + A1*nu^2
    b = np.linalg.lstsq(nu[:, None], d["beta_PP"] / bs - 1.0, rcond=None)[0]
    P = np.polyfit(nu, d["P"], 1)[::-1]                             # P0, P1
    return np.concatenate([c, A, b, P])


def split(theta, model, alpha_fixed=None):
    if alpha_fixed is not None:                      # theta = [b, P0, P1]
        c, A = alpha_fixed
        return c, A, float(theta[0]), theta[1:3]
    na = NA[model]
    k = 0
    c = theta[k:k + 2]; k += 2
    A = theta[k:k + na]; k += na
    b = theta[k]; k += 1
    P = theta[k:k + 2]
    return c, A, float(b), P


def params_at(q, theta, model, alpha_fixed=None):
    """[a_PP, a_C, b_PP, b_E] at one q from the master coefficients."""
    c, A, b, P = split(theta, model, alpha_fixed)
    nu = float(nu_of(q)); bs = float(base_of(q))
    a_pp = float(np.clip(bs * (1.0 + c[0] * nu + c[1] * nu ** 2), 0.05, 2.5))
    b_pp = float(np.clip(bs * (1.0 + b * nu), 0.2, 1.6))
    a_c = float(A[0] / nu) if model == "anchored" else float(A[0] * nu + A[1] * nu ** 2)
    p_val = float(P[0] + P[1] * nu)
    return np.array([a_pp, a_c, b_pp, p_val / b_pp]), p_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(NA), default="anchored")
    ap.add_argument("--t-cut", type=float, default=IN.DEFAULT_T_CUT)
    ap.add_argument("--global", dest="do_global", action="store_true",
                    help="joint refit of the master coefficients against the truncated "
                         "windows (Powell), instead of regressing the per-q optima")
    ap.add_argument("--ntrain", type=int, default=12)
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--fit-alpha", action="store_true",
                    help="ALSO fit alpha's coefficients on the inspiral.  Off by default: "
                         "the inspiral cannot measure alpha's coupling and returns the "
                         "wrong sign (see parent_alpha's docstring).  Diagnostic only.")
    args = ap.parse_args()

    model, t_cut = args.model, args.t_cut
    drive = IN.MODELS[model]["drive"]
    d = load_per_q(model, t_cut)
    alpha_fixed = None if args.fit_alpha else parent_alpha(model)
    theta = seed_theta(d, model, alpha_fixed)
    ncoef = len(theta)

    print(f"[inspiral_{model}] FITTED master layer, drive={drive}, t_cut={t_cut:g} M")
    if alpha_fixed is None:
        print(f"  !! --fit-alpha: alpha fitted on the inspiral, where it is NOT "
              f"measurable -- expect a wrong-signed alpha_C.  Diagnostic only.")
        print(f"  {ncoef} coefficients (alpha_PP 2, alpha_C {NA[model]}, beta_PP 1, P 2) "
              f"regressed on {len(d['q'])} per-q optima in [{d['q'][0]:g}, {d['q'][-1]:g}]")
    else:
        c0, A0 = alpha_fixed
        print(f"  alpha INHERITED from the parent full-window fit (not refitted here): "
              f"c={np.round(c0, 6).tolist()}  A={np.round(A0, 6).tolist()}")
        print(f"  {ncoef} free coefficients (beta_PP 1, P 2) "
              f"regressed on {len(d['q'])} per-q optima in [{d['q'][0]:g}, {d['q'][-1]:g}]")
    print(f"  per-q b_E > 0 at {int((d['b_E'] > 0).sum())}/{len(d['b_E'])} q, "
          f"so P > 0 over the training range", flush=True)

    t0_seeds = {k: v[4] for k, v in IN.mult_seeds().items()}
    t0_poly = np.polyfit(nu_of(d["q"]),
                         [t0_seeds[round(q, 10)] for q in d["q"]], 3)

    cases, tcases = {}, {}

    def case_for(q):
        if q not in cases:
            cases[q] = FL.add_flux(G.load_case(q, IN.SRC_STRIDE, IN.NR_STRIDE))
            tcases[q] = IN.truncate(cases[q], t_cut)
        return cases[q], tcases[q]

    def score(q, th):
        """(inspiral-window mismatch, full-window mismatch, P, b_E) at one q."""
        case, tcase = case_for(q)
        p, p_val = params_at(q, th, model, alpha_fixed)
        seed = float(np.polyval(t0_poly, float(nu_of(q))))
        e_fit, t0 = IN.mism(p, tcase, seed, drive, "fit")
        e_insp = IN.mism(p, tcase, t0, drive, "full")[0]
        e_full = IN.mism(p, case, t0, drive, "full")[0]
        return e_insp, e_full, p_val, float(p[3])

    if args.do_global:
        # train on points sampled even in nu, as the parents do
        targets = np.linspace(nu_of(d["q"]).min(), nu_of(d["q"]).max(), args.ntrain)
        train_q = sorted({float(d["q"][np.argmin(np.abs(nu_of(d["q"]) - t))])
                          for t in targets})
        nuis = {q: float(np.polyval(t0_poly, float(nu_of(q)))) for q in train_q}
        for q in train_q:
            case_for(q)

        def obj(th):
            errs = []
            for q in train_q:
                p, _ = params_at(q, th, model, alpha_fixed)
                e, t0 = IN.mism(p, tcases[q], nuis[q], drive, "fit")
                nuis[q] = t0
                errs.append(e)
            errs = np.asarray(errs)
            obj.last = (float(errs.mean()), float(errs.max()))
            return errs.mean()
        obj.last = (np.nan, np.nan)

        obj(theta)
        print(f"\n[global] train q: {[round(q, 2) for q in train_q]}")
        print(f"[global] seed: mean={obj.last[0]:.4e} max={obj.last[1]:.4e}", flush=True)
        t_s = time.time()
        res = minimize(obj, theta, method="Powell",
                       options={"maxiter": args.maxiter, "xtol": 1e-4, "ftol": 1e-6})
        print(f"[global] opt {time.time()-t_s:.0f}s: mean={obj.last[0]:.4e} "
              f"max={obj.last[1]:.4e}", flush=True)
        theta = res.x

    print(f"\n{'q':>6} {'insp (scored)':>14} {'full (orientation)':>19} "
          f"{'P':>10} {'b_E':>9} {'beta rises':>11}")
    final = {}
    for q in list(IN_Q) + list(LOW_Q):
        e_insp, e_full, p_val, b_e = score(q, theta)
        final[q] = {"insp": e_insp, "full": e_full, "P": p_val, "b_E": b_e}
        tag = "" if q in IN_Q else "  [extrap]"
        print(f"{q:>6g} {e_insp:>14.4e} {e_full:>19.4e} {p_val:>+10.4f} {b_e:>+9.4f} "
              f"{'YES' if b_e > 0 else 'NO':>11}{tag}")

    ins = [final[q]["insp"] for q in IN_Q]
    med, mx = float(np.median(ins)), float(np.max(ins))
    perq_med = float(np.median(d["err"]))
    print(f"\nin-range (scored window): median {med:.4e}  max {mx:.4e}")
    print(f"per-q floor (same window, 64 q): median {perq_med:.4e}  "
          f"-> master/floor = {med/perq_med:.2f}x")
    n_rise = sum(final[q]["b_E"] > 0 for q in list(IN_Q) + list(LOW_Q))
    print(f"beta RISES at {n_rise}/{len(IN_Q)+len(LOW_Q)} reported q "
          f"(including the {len(LOW_Q)} extrapolated ones)")

    c, A, b, P = split(theta, model, alpha_fixed)
    out = {"model": f"inspiral_{model}_fitted", "drive": drive, "t_cut": t_cut,
           "n_coefficients": int(ncoef), "global_refit": bool(args.do_global),
           "alpha_fitted_on_inspiral": bool(args.fit_alpha),
           "alpha_source": ("this fit (DIAGNOSTIC, wrong-signed)" if args.fit_alpha
                            else "parent full-window fit"),
           "form": {"alpha_PP": "X1^(6/5)*(1+c0*nu+c1*nu^2)",
                    "alpha_C": "A0/nu" if model == "anchored" else "A0*nu+A1*nu^2",
                    "beta_PP": "X1^(6/5)*(1+b*nu)", "P": "P0+P1*nu",
                    "b_E": "P/beta_PP"},
           "c": c.tolist(), "A": np.asarray(A).tolist(), "b": b, "P": P.tolist(),
           "scored_window": {"median": med, "max": mx},
           "per_q_floor_median": perq_med,
           "per_q": {f"{q:g}": final[q] for q in list(IN_Q) + list(LOW_Q)},
           "beta_rises_at_all_reported_q": bool(n_rise == len(IN_Q) + len(LOW_Q)),
           "caveat": "insp is the scored objective; full includes merger+ringdown which "
                     "this model never saw.  Not comparable to CLAUDE.md's table."}
    suffix = ("_fitalpha" if args.fit_alpha else "")
    path = results_dir(model) / (f"coeffs_global{suffix}.json" if args.do_global
                                 else f"coeffs{suffix}.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
