"""
nu-regression (master layer) for the x-driven model -- the layer fit_scaling_x_drive.py
deliberately lacks.

    s(t)     = x(t) - x(t_start),   x = (0.5 * omega_GW,pp)^(2/3)
    alpha(t) = a_PP(nu) * (1 + sum_{k=1..ka} a_k(nu) * s^k)
    beta (t) = b_PP(nu) * (1 + sum_{k=1..kb} b_k(nu) * s^k)

    a_PP(nu) = X1^(6/5) * (1 + c0*nu + c1*nu^2)                    2   [derived base]
    a_k(nu)  = A_k1*nu + A_k2*nu^2         k = 1..ka           2*ka   [PP-anchored]
    b_PP(nu) = X1^(6/5) * (1 + b*nu)                               1   [derived base]
    b_k(nu)  = B_k1*nu + B_k2*nu^2         k = 1..kb           2*kb   [PP-anchored]

    -> 7 coefficients at ka=kb=1,  11 at ka=kb=2

The prefactors reuse the parents' forms exactly (`fit_scaling_gwr_energy_anchored.py`), so
the comparison against the 6-coefficient energy model is like-for-like.  The SHAPE
coefficients are PP-anchored -- they carry no constant term, so every correction vanishes
as nu -> 0 where the ppBHPT is exact and X1^(6/5) -> 1.  `--free-shape` adds the constant
back if that constraint turns out to distort the in-range fit (the training range is
nu in [0.099, 0.222], nowhere near 0, so anchoring there is pure extrapolation).

REGRESSABILITY, measured before building this (deg-2 fit in nu to the per-q coefficients):

    ka=1/kb=1   a_PP 0.999   a_1 0.998   b_1 0.992     max resid 2.4-4.8% of span
    ka=2/kb=2   a_PP 0.996   a_1 0.924   a_2 0.947 ...  max resid 18% of span

So ka=1/kb=1 is the clean case and ka=2/kb=2 has coefficient scatter of the size that
made `wf_nu_fluxes` unregressable despite its best-in-workspace per-q numbers.  Both are
run; if ka=2/kb=2 gives back its 30% per-q advantage at the master level, that is a
result about the degree choice, not a failure.

Usage:
    python fit_scaling_x_drive_regress.py --ka 1 --kb 1
    python fit_scaling_x_drive_regress.py --ka 2 --kb 2 --global --maxiter 80
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
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_x_drive as XD

IN_Q = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
LOW_Q = (2.75, 2.5, 2.25, 2.0)

nu_of = lambda q: np.asarray(q, float) / (1.0 + np.asarray(q, float)) ** 2
X1_of = lambda q: np.asarray(q, float) / (1.0 + np.asarray(q, float))
base_of = lambda q: X1_of(q) ** 1.2

# reference: the energy family's master numbers on the same footing
REF = {"gwr_energy_anchored (6c)": {"median": 6.2551e-04, "q2": 1.53e-03},
       "gwr_energy_fluxanchored (7c)": {"median": 4.752e-04, "q2": 1.306e-03}}


def n_shape(free_shape):
    return 3 if free_shape else 2


def load_per_q(ka, kb, t_cut=None):
    tag = f"ka{ka}_kb{kb}" + ("" if t_cut is None else f"_tcut{abs(t_cut):.0f}")
    path = XD.RESULTS / f"per_q_cache_{tag}.json"
    if not path.exists():
        sys.exit(f"no per-q cache at {path} -- run fit_scaling_x_drive.py first")
    rows = sorted(json.loads(path.read_text()).values(), key=lambda r: r["q"])
    q = np.array([r["q"] for r in rows], float)
    return {"q": q, "nu": nu_of(q), "tag": tag,
            "a_PP": np.array([r["a_PP"] for r in rows], float),
            "b_PP": np.array([r["b_PP"] for r in rows], float),
            "a_k": np.array([r["a_k"] for r in rows], float),
            "b_k": np.array([r["b_k"] for r in rows], float),
            "err": np.array([r["err_scored"] for r in rows], float)}


def seed_theta(d, ka, kb, free_shape):
    nu, bs = d["nu"], base_of(d["q"])
    ns = n_shape(free_shape)

    def shape_fit(y):
        cols = [nu ** j for j in range(0 if free_shape else 1, 3)]
        return np.linalg.lstsq(np.vstack(cols).T, y, rcond=None)[0]

    c = np.linalg.lstsq(np.vstack([nu, nu ** 2]).T, d["a_PP"] / bs - 1.0, rcond=None)[0]
    b = np.linalg.lstsq(nu[:, None], d["b_PP"] / bs - 1.0, rcond=None)[0]
    parts = [c]
    for k in range(ka):
        parts.append(shape_fit(d["a_k"][:, k]))
    parts.append(b)
    for k in range(kb):
        parts.append(shape_fit(d["b_k"][:, k]))
    return np.concatenate(parts)


def params_at(q, theta, ka, kb, free_shape):
    ns = n_shape(free_shape)
    nu = float(nu_of(q)); bs = float(base_of(q))
    powers = np.array([nu ** j for j in range(0 if free_shape else 1, 3)])

    i = 0
    c = theta[i:i + 2]; i += 2
    a_k = []
    for _ in range(ka):
        a_k.append(float(np.dot(theta[i:i + ns], powers))); i += ns
    b = float(theta[i]); i += 1
    b_k = []
    for _ in range(kb):
        b_k.append(float(np.dot(theta[i:i + ns], powers))); i += ns

    a_pp = float(np.clip(bs * (1.0 + c[0] * nu + c[1] * nu ** 2), 0.05, 2.5))
    b_pp = float(np.clip(bs * (1.0 + b * nu), 0.2, 1.6))
    return np.r_[a_pp, np.array(a_k), b_pp, np.array(b_k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ka", type=int, default=1)
    ap.add_argument("--kb", type=int, default=1)
    ap.add_argument("--t-cut", type=float, default=None)
    ap.add_argument("--free-shape", action="store_true",
                    help="allow a constant term in the shape coefficients (drops the "
                         "PP anchor on them)")
    ap.add_argument("--global", dest="do_global", action="store_true")
    ap.add_argument("--ntrain", type=int, default=12)
    ap.add_argument("--maxiter", type=int, default=80)
    args = ap.parse_args()

    ka, kb, fs = args.ka, args.kb, args.free_shape
    d = load_per_q(ka, kb, args.t_cut)
    theta = seed_theta(d, ka, kb, fs)
    ncoef = len(theta)

    print(f"[x_drive_regress] ka={ka} kb={kb}  shape coeffs "
          f"{'FREE' if fs else 'PP-anchored'}  window "
          f"{'full' if args.t_cut is None else f't_nr <= {args.t_cut:g} M'}")
    print(f"  {ncoef} coefficients (a_PP 2, a_k {n_shape(fs)}x{ka}, b_PP 1, "
          f"b_k {n_shape(fs)}x{kb}) from {len(d['q'])} per-q optima", flush=True)

    cases = {}

    def case_for(q):
        if q not in cases:
            cases[q] = XD.add_x(FL.add_flux(G.load_case(q, XD.SRC_STRIDE, XD.NR_STRIDE)))
        return cases[q]

    def score(q, th):
        case = case_for(q)
        sc = case if args.t_cut is None else XD.truncate(case, args.t_cut)
        p = params_at(q, th, ka, kb, fs)
        e_fit, t0 = XD.mism(p, sc, -30.0, ka, kb, "fit")
        return XD.mism(p, sc, t0, ka, kb, "full")[0]

    if args.do_global:
        targets = np.linspace(d["nu"].min(), d["nu"].max(), args.ntrain)
        train_q = sorted({float(d["q"][np.argmin(np.abs(d["nu"] - t))]) for t in targets})
        for q in train_q:
            case_for(q)

        def obj(th):
            errs = np.array([score(q, th) for q in train_q])
            obj.last = (float(errs.mean()), float(errs.max()))
            return errs.mean()
        obj.last = (np.nan, np.nan)
        obj(theta)
        print(f"\n[global] train q: {[round(q,2) for q in train_q]}")
        print(f"[global] seed: mean={obj.last[0]:.4e} max={obj.last[1]:.4e}", flush=True)
        t_s = time.time()
        res = minimize(obj, theta, method="Powell",
                       options={"maxiter": args.maxiter, "xtol": 1e-4, "ftol": 1e-6})
        print(f"[global] opt {time.time()-t_s:.0f}s: mean={obj.last[0]:.4e} "
              f"max={obj.last[1]:.4e}", flush=True)
        theta = res.x

    print(f"\n{'q':>6} {'master':>12} {'per-q floor':>12} {'ratio':>7}")
    floor = {round(float(q), 4): e for q, e in zip(d["q"], d["err"])}
    final = {}
    for q in list(IN_Q) + list(LOW_Q):
        e = score(q, theta)
        final[q] = e
        f = floor.get(round(q, 4))
        rt = f"{e/f:7.2f}" if f else f"{'--':>7}"
        tag = "" if q in IN_Q else "  [extrap]"
        print(f"{q:>6g} {e:>12.4e} {f if f else float('nan'):>12.4e} {rt}{tag}")

    ins = [final[q] for q in IN_Q]
    med, mx = float(np.median(ins)), float(np.max(ins))
    print(f"\n=== x_drive master  ka={ka} kb={kb}  {ncoef} coefficients ===")
    print(f"in-range: median {med:.4e}  max {mx:.4e}")
    print(f"per-q floor (64 q): median {np.median(d['err']):.4e}  "
          f"-> master/floor {med/np.median(d['err']):.2f}x")
    print(f"q<3: " + "  ".join(f"{q:g}={final[q]:.3e}" for q in LOW_Q))
    for nm, r in REF.items():
        print(f"  ref {nm}: median {r['median']:.4e}  q2 {r['q2']:.3e}")

    out = {"model": "x_drive_master", "ka": ka, "kb": kb, "t_cut": args.t_cut,
           "shape_pp_anchored": not fs, "n_coefficients": int(ncoef),
           "global_refit": bool(args.do_global), "theta": theta.tolist(),
           "in_range": {"median": med, "max": mx},
           "per_q_floor_median": float(np.median(d["err"])),
           "low_q": {f"{q:g}": final[q] for q in LOW_Q},
           "reference": REF}
    tag = d["tag"] + ("_freeshape" if fs else "") + ("_global" if args.do_global else "")
    p = XD.RESULTS / f"coeffs_{tag}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
