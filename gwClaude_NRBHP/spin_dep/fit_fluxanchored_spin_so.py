"""fluxanchored_spin_so (2026-10-04; md: scaling_fluxanchored_spin_so.md): the ORIGINAL fluxanchored_spin form + the derived PN spin-orbit term, ONE form for the
whole waveform (no split, no blend), 9 coefficients, fitted the original way: mean full-window mathcalE ('full' alignment).

    alpha = a_pp (1 + a_F F),   beta = b_pp (1 + b_e E) (1 + so(q) chi1 x^(3/2))      (fit_flux_spin_split.ab, S = 0)
    so(q) = (3/2)[113/12 X1^(6/5) - 19/6 X1^2 - 25/4 X1]  -- derived, no fitted multiplier.

Starts: the installed fluxanchored_spin coefficients and the inspiral copy of the T96 split fit.  Fit on T96, scored on
T96 and H270 (both alignments) next to the installed fluxanchored_spin and pn_anchored_spin.

One form, no split, so it lives in spin_dep/ (moved from spin_dep_split/ 2026-10-04, user rule).  It reuses the split
line's case builder and scorer (spin_dep_split/fit_flux_spin_split.py with the blend OFF, S = 0) -- imported, not copied.
-> fluxanchored_spin_so_results/coeffs.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from multiprocessing import get_context

import numpy as np
from scipy.optimize import minimize

OUTDIR = Path(__file__).resolve().parent / "fluxanchored_spin_so_results"
SPLIT = Path("/work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/gwClaude_NRBHP/spin_dep_split")
sys.path.insert(0, str(SPLIT))
import fit_flux_spin_split as FF
import score_h270 as H
M = FF.M
T96 = [(q, c) for q in M.S.TRAIN_Q for c in M.S.CHIS]


def zero_blend(keys):
    for k in keys:
        c = M.CASES[k]; c["S"] = 0 * c["S"]; c["fit_S"] = 0 * c["fit_S"]


def fit(pool, th0):
    FF.KEYS = T96
    FF.run(pool, "split", th0, th0, align="full", wide=True)
    def obj(th):
        R = FF.run(pool, "split", th, th, align="full")
        return float(np.mean([R[k]["errs"][2] for k in T96]))
    x = np.array(th0, float); t = time.time(); f0 = best = obj(x)
    for rep in range(6):
        r = minimize(obj, x, method="Nelder-Mead",
                     options={"maxiter": 900 * len(x), "xatol": 1e-8, "fatol": 1e-12, "adaptive": True})
        FF.run(pool, "split", r.x, r.x, align="full", wide=True); fr = obj(r.x)
        done = fr > best * (1 - 1e-3)
        if fr < best:
            x, best = r.x, fr
        if done:
            break
    print(f"  fit: mean E_full (fit grid) {f0:.4e} -> {best:.4e} ({time.time() - t:.0f}s)", flush=True)
    return x, best


def main():
    for k in T96 + H.H270:
        M.CASES[k] = M.prepare(*k)
    zero_blend(T96 + H.H270)
    starts = {"installed_flux": np.array(json.loads((FF.FS.RESULTS / "coeffs_nu.json").read_text())["theta"]),
              "split_T96_insp": np.array(json.loads((SPLIT / "results" / "flux_spin_split_T96_T-1.json").read_text())["theta_insp"])}
    out = dict(note=__doc__.split("\n\n")[0], runs={})
    with get_context("fork").Pool(128) as pool:
        best = None
        for name, th0 in starts.items():
            FF.SEEDS.clear(); x, f = fit(pool, th0)
            out["runs"][name] = dict(theta=list(map(float, x)), mean_fit=f)
            if best is None or f < best[0]:
                best = (f, x, name)
        th = best[1]
        for grid, keys in (("T96", T96), ("H270", H.H270)):
            FF.KEYS = keys; FF.SEEDS.clear()
            out[grid] = {"fluxanchored_spin_so": FF.final(pool, "split", th, th)}
            for kind in ("flux9", "pn13"):
                out[grid][f"installed_{kind}"] = FF.final(pool, kind, None, None)
            print(f"{grid}:")
            for name, rows in out[grid].items():
                FF.summary(rows, name)
    out["theta_best"] = list(map(float, th)); out["best_start"] = best[2]
    (OUTDIR / "coeffs.json").write_text(json.dumps(out, indent=1))
    print("theta", dict(zip(FF.FS.NAMES, np.round(th, 5))), "\nwrote", OUTDIR / "coeffs.json")


if __name__ == "__main__":
    main()
