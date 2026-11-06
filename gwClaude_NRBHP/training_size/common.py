"""Shared pieces of the training-size study: start points, the generic optimiser, stats, result files.

Start points (choice made on the TRAINING objective only, never on V300):
    t96     the model's shipped T96 coefficients (how every pass of the exploration was started)
    pertN   the same, perturbed with seed N: theta * (1 + 0.25 z1) + 0.02 z2 (z ~ N(0, 1)), redrawn until the start is a
            valid model (finite, beta > 0, monotone clock).  Tests whether a small grid just stays in the T96 basin.
"""
from __future__ import annotations
import json, time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)


def stats(v):
    v = np.asarray(list(v.values()) if isinstance(v, dict) else v, float)
    return dict(n=int(v.size), median=float(np.median(v)), mean=float(v.mean()), max=float(v.max()))


def fmt(s):
    return f"{s['median']:.3e} / {s['mean']:.3e} / {s['max']:.2e} (n={s['n']})"


def kstr(k):
    return f"{k[0]:.6f}_{k[1]:+.3f}"


def start_point(theta, start, valid, free=None):
    """theta for start 't96' or 'pert<seed>'; valid(theta) -> bool.  Only the free entries are perturbed."""
    theta = np.asarray(theta, float)
    if start == "t96":
        return theta.copy()
    assert start.startswith("pert"), start
    r = np.random.default_rng(int(start[4:]))
    free = np.ones(theta.size, bool) if free is None else np.asarray(free, bool)
    for _ in range(200):
        z1, z2 = r.standard_normal(theta.size), r.standard_normal(theta.size)
        th = np.where(free, theta * (1 + 0.25 * z1) + 0.02 * z2, theta)
        if valid(th):
            return th
    raise RuntimeError(f"no valid perturbed start in 200 draws ({start})")


def fit_nm_powell(obj, x0, budget_min, reset=None, log=None, tag=""):
    """Nelder-Mead restarts (adaptive) then a Powell polish, all on obj (mean training error).  Stops at budget_min
    and keeps the best point seen.  reset() is called between restarts (e.g. to drop warm t0 seeds).
    The same recipe as spin_dep/refit_pn_anchored_spin_13free.py."""
    st = {"n": 0, "best": np.inf, "x": np.array(x0, float), "t": time.time()}
    log = [] if log is None else log

    def f(x):
        v = float(obj(x)); st["n"] += 1
        if v < st["best"]:
            st["best"], st["x"] = v, np.array(x, float)
        if time.time() - st["t"] > budget_min * 60:
            raise TimeoutError
        return v
    x = np.array(x0, float); converged = False
    try:
        f0 = f(x); best = f0
        for rep in range(6):
            r = minimize(f, x, method="Nelder-Mead",
                         options={"maxiter": 600 * len(x), "xatol": 1e-8, "fatol": 1e-12, "adaptive": True})
            if reset:
                reset()
            fr = f(r.x); done = fr > best * (1 - 1e-3)
            if fr < best:
                x, best = r.x, fr
            log.append(dict(stage=f"NM {rep}", mean_fit=best, n_eval=st["n"]))
            print(f"  {tag} NM restart {rep}: {best:.4e} ({st['n']} evals, {(time.time() - st['t']) / 60:.1f} min)", flush=True)
            if done:
                break
        r = minimize(f, x, method="Powell", options={"xtol": 1e-5, "ftol": 1e-9, "maxiter": 20000})
        if r.fun < best:
            x, best = r.x, float(r.fun)
        log.append(dict(stage="Powell", mean_fit=best, n_eval=st["n"]))
        converged = True
    except TimeoutError:
        f0 = log[0]["mean_fit"] if log else np.nan
        x, best = st["x"], st["best"]
        print(f"  {tag} budget reached after {st['n']} evals; best {best:.4e}", flush=True)
    return dict(x=x, best=best, converged=converged, n_eval=st["n"], minutes=(time.time() - st["t"]) / 60, log=log)


def write(model, n, start, payload):
    p = RESULTS / f"{model}__G{n}__{start}.json"
    p.write_text(json.dumps(dict(model=model, grid=n, start=start, **payload), indent=1))
    print("wrote", p, flush=True)
    return p
