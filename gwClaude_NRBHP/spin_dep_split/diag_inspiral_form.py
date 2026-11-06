"""Diagnostic (2026-10-03): why is the split model's inspiral worse than the installed model's?

The S18 two-stage fit reached median E_insp 1.27e-3 (insp-aligned) vs the installed pn_anchored_spin's
2.12e-4, and the excess is in the EARLY inspiral (NR t < -1000 M: 6.4e-4 vs 1.0e-4), unchanged with the
switch turned off.  Two hypotheses:
  (1) optimisation: stage 1 did not reach the best inspiral fit (start too far, local minimum);
  (2) form: the pure inspiral form beta = X1^(6/5) (1 + nu (b1c xc + b2c xc^2)) cannot supply the nearly
      constant offset the installed model gets from its never-off switch (4-6% MR weight early on).
Test, switch OFF (S = 0: pure inspiral form), objective = mean E_insp, 'insp' alignment, S18:
  variant 'base' : the 7 inspiral coefficients, several starts, heavier optimisation  -> tests (1)
  variant 'b0'   : + one constant term nu*b0 inside beta's bracket (8 coefficients)    -> tests (2)
Not a model; nothing is installed.  -> results/diag_inspiral_form.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from multiprocessing import get_context

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_pn_anchored_spin_split as M

KEYS = [(q, c) for q in (3.0, 5.0, 8.0) for c in (-0.6, -0.2, 0.0, 0.4, 0.6, 0.8)]
BANDS = [(-1e9, -1000.0), (-1000.0, -300.0), (-300.0, -100.0), (-100.0, 1e9)]
SEEDS = {}


def ab(p, q, chi, c, key, variant):
    pre = "fit_" if key == "fit" else ""
    b1, b2, a2, s11, s12, s21, s22 = p[:7]
    b0 = p[7] if variant == "b0" else 0.0
    X1 = q / (1 + q); nu = q / (1 + q) ** 2; base = X1 ** 1.2
    xc = M.PC.x_scale(q) * np.minimum(c[pre + "x"], M.S.XCLIP)
    beta = base * (1 + nu * (b0 + (b1 + s11 * chi + s12 * chi ** 2) * xc + (b2 + s21 * chi + s22 * chi ** 2) * xc ** 2))
    alpha = base * (1 + M.S.PN_SLOPE * nu * xc + nu * a2 * xc ** 2)
    return alpha, beta


def _task(a):
    k, p, variant, seed, key, wide = a
    al, be = ab(p, *k, M.CASES[k], key, variant)
    errs, t0 = M.score(al, be, M.CASES[k], key, "insp", seed, wide)
    return k, errs[0], t0


def run(pool, p, variant, key="fit", wide=False):
    out = pool.map(_task, [(k, p, variant, SEEDS.get(k, -50.0), key, wide) for k in KEYS])
    for k, e, t0 in out:
        if e < 1.0:
            SEEDS[k] = t0
    return {k: e for k, e, _ in out}


def fit(pool, p0, variant):
    run(pool, p0, variant, wide=True)
    obj = lambda p: float(np.mean(list(run(pool, p, variant).values())))
    x, best = np.array(p0, float), obj(p0); f0 = best
    for rep in range(6):
        r = minimize(obj, x, method="Nelder-Mead",
                     options={"maxiter": 800 * len(x), "xatol": 1e-8, "fatol": 1e-12, "adaptive": True})
        run(pool, r.x, variant, wide=True)                       # refresh t0 seeds
        fr = obj(r.x)
        if fr < best * (1 - 1e-3):
            x, best = r.x, fr
        else:
            x, best = (r.x, fr) if fr < best else (x, best)
            break
    r = minimize(obj, x, method="Powell", options={"xtol": 1e-6, "ftol": 1e-10, "maxiter": 20000})
    if r.fun < best:
        x, best = r.x, float(r.fun)
    return x, f0, best


def full_res(pool, p, variant):
    run(pool, p, variant, wide=True)
    E = run(pool, p, variant, key="full", wide=True)
    bands = []
    for k in KEYS:                                                # band split of E_insp at full resolution
        c = M.CASES[k]; al, be = ab(p, *k, c, "full", variant)
        t, h, tn, hn = c["t_bhpt"], c["h_bhpt"], c["t_nr"], c["h_nr"]
        tau = SEEDS[k] + M.F.tau_shape(be, t)
        m = (tn >= max(tn[0], tau[0])) & (tn <= min(tn[-1], tau[-1])); tc, r = tn[m], hn[m]
        g = np.interp(tc, tau, al) * (np.interp(tc, tau, h.real) + 1j * np.interp(tc, tau, h.imag))
        ins = tc < c["lm"]["t_cut_nr"]; z = np.vdot(g[ins], r[ins]); g = g * z / abs(z)
        res = np.abs(r - g) ** 2; n = (np.abs(r[ins]) ** 2).sum()
        bands.append([float(res[ins & (tc >= lo) & (tc < hi)].sum() / (2 * n)) for lo, hi in BANDS])
    return E, np.array(bands)


def main():
    for k in KEYS:
        M.CASES[k] = M.prepare(*k)
    inst, _ = M.installed_start()
    proj, _ = M.projected_start(KEYS)               # uses the real window (window start), so before S = 0
    for k in KEYS:                                  # switch OFF: pure inspiral form
        M.CASES[k]["S"] = 0 * M.CASES[k]["S"]; M.CASES[k]["fit_S"] = 0 * M.CASES[k]["fit_S"]
    split = np.array(json.loads((HERE / "results" / "pn_anchored_spin_split_S18.json").read_text())["theta_insp"])
    starts = {"installed": inst, "projected": proj, "split_S18": split,
              "installed_no_spin": np.r_[inst[:3], 0, 0, 0, 0]}
    res = dict(note=__doc__.split("\n\n")[0], runs=[])
    with get_context("fork").Pool(len(KEYS)) as pool:
        for variant in ("base", "b0"):
            best = None
            for name, s in starts.items():
                p0 = s if variant == "base" else np.r_[s, 0.0]
                SEEDS.clear(); t = time.time()
                x, f0, f = fit(pool, p0, variant)
                print(f"{variant:4s} start={name:18s} mean E_insp (fit grid) {f0:.3e} -> {f:.3e}  ({time.time() - t:.0f}s)", flush=True)
                res["runs"].append(dict(variant=variant, start=name, start_mean=f0, end_mean=f, theta=list(map(float, x))))
                if best is None or f < best[0]:
                    best = (f, x)
            SEEDS.clear()
            E, B = full_res(pool, best[1], variant)
            v = np.array(list(E.values()))
            print(f"BEST {variant}: full-res E_insp mean {v.mean():.3e} median {np.median(v):.3e} max {v.max():.2e}; "
                  "median by band: " + ", ".join(f"{lab} {np.median(B[:, j]):.1e}" for j, lab in
                                                 enumerate(["< -1000", "-1000..-300", "-300..-100", "-100..cut"])), flush=True)
            if variant == "b0":
                print(f"     b0 = {best[1][7]:+.4f}  (beta offset at early times = nu*b0, i.e. {100 * 0.16 * best[1][7]:+.2f}% at q=4)")
            res[f"best_{variant}"] = dict(theta=list(map(float, best[1])), E_insp_full={f"{k[0]}_{k[1]}": e for k, e in E.items()},
                                         bands_median=list(np.median(B, axis=0)))
    print("reference: installed pn_anchored_spin, insp-aligned, full res: E_insp mean 7.85e-4, median 2.12e-4; "
          "by band 1.0e-4, 3.9e-5, 2.3e-5, 7.9e-5")
    (HERE / "results" / "diag_inspiral_form.json").write_text(json.dumps(res, indent=1))
    print("wrote results/diag_inspiral_form.json")


if __name__ == "__main__":
    main()
