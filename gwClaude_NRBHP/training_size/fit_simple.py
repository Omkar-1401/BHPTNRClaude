"""Training-size refits of the two single-stage installed spin models.

  pn13        pn_anchored_spin (spin_dep/fit_pn_anchored_spin.py), 13 coefficients; b1, b2, a2 stay at the nonspinning
              pn_anchored values (inherited, as in the installed model: they come from the 1d fit, not from T96); the other
              10 are refitted.  Objective: mean fit-stride mismatch, warm t0 (test_pn_spin_joint2d.mismatch).
              Score: S.full_mm (the stored scorer).
  flux_orbit  flux_anc_spin_with_orbit (spin_dep/fit_flux_anc_spin_with_orbit.py), all 9 coefficients refitted.
              Objective: mean fit-stride mismatch, warm t0 (FO.fast_mismatch, bounded +-60 M around the previous t0;
              first call a coarse 81-point scan).  Score: FO.mismatch (the stored scorer).
Optimiser: common.fit_nm_powell (NM restarts + Powell) for both.
Gate (every run): the shipped coefficients through this scorer reproduce the stored per-case T96 mismatches (ratio
within 1e-3), else exit.

    python fit_simple.py pn13 --grid 24 --start t96 --budget 60
"""
from __future__ import annotations
import argparse, json, sys, time
from multiprocessing import get_context

import numpy as np

import grids as G                              # FIRST: installs the cache routing + sealed guard into S / FS
import common as C
S, FS = G.S, G.FS
import test_pn_spin_joint2d as J               # noqa: E402
import fit_flux_anc_spin_with_orbit as FO      # noqa: E402  (binds FO.build_case to the routed builder)

CASES = {}
M = {}                                         # the active model adapter


def _pn13():
    th = np.r_[S.model9(), np.array(json.loads(S.COEFFS.read_text())["theta_spin"], float)]
    stored = json.loads(S.COEFFS.read_text())["mismatches"]
    free = np.ones(13, bool); free[:3] = False
    return dict(names=list(S.NAMES9) + ["s11", "s12", "s21", "s22"], theta=th, free=free, stored=stored,
                warm=lambda t, k, seed: J.mismatch(t, *k, CASES[k], seed, "fit"),
                full=lambda t, k: S.full_mm(t[:9], t[9:], *k, CASES[k]),
                build=S.build_case, source=str(S.COEFFS))


def _flux_orbit():
    d = json.loads(FO.COEFFS.read_text())
    if d.get("model_form") != FO.MODEL_FORM:
        raise RuntimeError("flux_anc_spin_with_orbit coeffs.json stamped with another form")

    def warm(t, k, seed):
        if seed is None:
            g = np.linspace(-250.0, 150.0, 81)
            seed = float(g[int(np.argmin([FO.fast_mismatch(t, *k, CASES[k], x, "fit")[0] for x in g]))])
        return FO.fast_mismatch(t, *k, CASES[k], seed, "fit")
    return dict(names=list(FO.NAMES), theta=np.array(d["theta"], float), free=np.ones(len(FO.NAMES), bool),
                stored=d["mismatches"], warm=warm, full=lambda t, k: FO.mismatch(t, *k, CASES[k])[0],
                build=FO.build_case, source=str(FO.COEFFS))


ADAPTERS = {"pn13": _pn13, "flux_orbit": _flux_orbit}


def _warm(a):
    k, th, seed = a
    return k, M["warm"](np.asarray(th), k, seed)


def _full(a):
    k, th = a
    return k, M["full"](np.asarray(th), k)


def score(pool, th, keys):
    return dict(pool.map(_full, [(k, list(th)) for k in keys], chunksize=1))


def stored_of(k):
    """stored per-case value for a T96 key (house key format q:.4f_c:+.1f)."""
    return M["stored"][f"{k[0]:.4f}_{k[1]:+.1f}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(ADAPTERS))
    ap.add_argument("--grid", type=int, required=True, choices=list(G.GRIDS))
    ap.add_argument("--start", default="t96")
    ap.add_argument("--budget", type=float, default=60.0)
    ap.add_argument("--nproc", type=int, default=128)
    ap.add_argument("--gate-only", action="store_true")
    a = ap.parse_args()
    M.update(ADAPTERS[a.model]())
    train = G.GRIDS[a.grid]
    for k in sorted(set(train) | set(G.T96) | set(G.V300)):
        CASES[k] = M["build"](*k)                # all before the fork
    with get_context("fork").Pool(a.nproc) as pool:
        E = score(pool, M["theta"], G.T96)
        r = np.array([E[k] / stored_of(k) for k in G.T96])
        print(f"gate: shipped {a.model} recomputed / stored on T96 in [{r.min():.6f}, {r.max():.6f}]", flush=True)
        if np.any(np.abs(r - 1) > 1e-3):
            sys.exit("gate failed: this scorer does not reproduce the shipped model")
        if a.gate_only:
            return

        seeds = {k: None for k in train}
        free = M["free"]; base = M["theta"].copy()

        def full_theta(x):
            th = base.copy(); th[free] = x; return th

        def obj(x):
            out = pool.map(_warm, [(k, list(full_theta(x)), seeds[k]) for k in train], chunksize=1)
            for k, (e, t0) in out:
                if e < 1.0:
                    seeds[k] = t0
            return float(np.mean([e for _, (e, _) in out]))

        def reset():
            seeds.update({k: None for k in train})

        def valid(th):
            reset(); ok = obj(th[free]) < 1.0; reset(); return ok
        th0 = C.start_point(base, a.start, valid, free)
        t = time.time()
        res = C.fit_nm_powell(obj, th0[free], a.budget, reset=reset, tag=f"{a.model} G{a.grid} {a.start}")
        th = full_theta(res["x"])
        print(f"fit: mean training (fit stride) {res['best']:.4e}, converged={res['converged']}, "
              f"{res['n_eval']} evals, {(time.time() - t) / 60:.1f} min", flush=True)
        out = {}
        for g, ks in (("train", train), ("T96", G.T96), ("V300", G.V300)):
            out[g] = score(pool, th, ks)
            print(f"  {g:5s} {C.fmt(C.stats(out[g]))}", flush=True)
        shipped_v = score(pool, M["theta"], G.V300)
        print(f"  shipped T96 coefficients on V300: {C.fmt(C.stats(shipped_v))}", flush=True)
    C.write(a.model, a.grid, a.start, dict(
        names=M["names"], free=list(map(bool, free)), theta=list(map(float, th)), theta_start=list(map(float, th0)),
        theta_shipped=list(map(float, M["theta"])), source=M["source"],
        fit=dict(best_mean_fit=res["best"], converged=res["converged"], n_eval=res["n_eval"], minutes=res["minutes"],
                 log=res["log"]),
        stats={g: C.stats(v) for g, v in out.items()} | {"V300_shipped": C.stats(shipped_v)},
        per_case={g: {C.kstr(k): float(e) for k, e in v.items()} for g, v in out.items()}))


if __name__ == "__main__":
    main()
