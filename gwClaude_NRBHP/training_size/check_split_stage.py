"""Why does the G96 split refit not reproduce shipped A / B?  Evaluate each STAGE objective at the shipped and the refit
coefficients on T96 (fresh or old cache, per TS_FRESH):
  stage 1 (harness protocol)   mean E_insp with the MR copy = the inspiral copy (explore_split.stage1)
  stage 1 (A's origin)         mean E_insp with tm = flux_spin_split_T96_T-1.json theta_mr, window T = -1
  stage 2                      mean E_MR of the slim MR branch at the model's window
If shipped < refit on the harness's own stage objective, the harness optimiser stops early; if shipped > refit, the two
are different local optima of a staged fit.

    [TS_FRESH=1] python check_split_stage.py
"""
from __future__ import annotations
import json, os
from multiprocessing import get_context

import numpy as np

import fit_split as FSP                        # routing + split modules (as the harness)
G, X, FF, M, CV = FSP.G, FSP.X, FSP.FF, FSP.M, FSP.CV
R = G.HERE / "results"


def mean_err(pool, ti, tm, i):
    FF.KEYS = G.T96; FF.SEEDS.clear()
    Rr = FF.run(pool, "split", ti, tm, wide=True)
    return float(np.mean([Rr[k]["errs"][i] for k in FF.KEYS]))


def main():
    print(f"TS_FRESH={os.environ.get('TS_FRESH')}", flush=True)
    for k in G.T96:
        M.CASES[k] = M.prepare(*k)
    origin = json.loads((FSP.SPLIT / "results" / "flux_spin_split_T96_T-1.json").read_text())
    tm_origin = np.array(origin["theta_mr"])
    for m in ("splitA", "splitB"):
        d = json.loads((R / f"{m}__G96__t96.json").read_text())
        th = {"shipped": (np.array(d["theta_insp_shipped"]), d["x_mr_shipped"]), "refit": (np.array(d["theta_insp"]), d["x_mr"])}
        X.set_window(-1.0)
        with get_context("fork").Pool(96) as pool:
            for lab, (ti, _) in th.items():
                print(f"{m} {lab:7s} stage-1 harness E_insp {mean_err(pool, ti, ti, 0):.5e}   "
                      f"origin (tm = T96_T-1 MR copy, T = -1) {mean_err(pool, ti, tm_origin, 0):.5e}", flush=True)
        FSP.MODELS[m]["window"]()
        with get_context("fork").Pool(96) as pool:
            for lab, (ti, x) in th.items():
                print(f"{m} {lab:7s} stage-2 E_MR {mean_err(pool, ti, ('form', 'slim', list(x)), 1):.5e}   "
                      f"full E mean {mean_err(pool, ti, ('form', 'slim', list(x)), 2):.5e} (fit stride, insp-aligned seeds)", flush=True)


if __name__ == "__main__":
    main()
