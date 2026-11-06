"""Score FINAL split candidates on the fresh held-out set F100 (2026-10-13).  Run ONCE, after the choice is made on T96 CV
(CV_LOG.md).  No fitting.  Every run appends a "look" line to CV_LOG.md.

    python score_f100.py --candidate NAME [--candidate ...]
NAME: an explore_split variant with a full-T96 fit in results/explore/NAME.json, or "tline" (results/cv/tline__foldall.json).
H270 is scored too, for the tline (the explore variants already carry it).  -> results/f100/<NAME>.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from multiprocessing import get_context

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fresh_set as FR                       # installs the F100-aware get_case (before cv_split / explore_split use it)
import cv_split as C
X, FF, M = C.X, C.FF, C.M
OUT = HERE / "results" / "f100"; OUT.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--candidate", action="append", required=True)
    a = ap.parse_args()
    grids = {"F100": FR.F100, "H270": X.H.H270}
    for k in FR.F100 + X.H.H270:
        M.CASES[k] = M.prepare(*k); X.rem(*k)
    for name in a.candidate:
        if name == "tline":
            d = json.loads((HERE / "results" / "cv" / "tline__foldall.json").read_text())
            C.set_tline(d["chosen"]["T0"], d["chosen"]["T1"]); mr = d["spec"]["mr"]; use = ["F100", "H270"]
        else:
            d = json.loads((X.OUT / f"{name}.json").read_text())
            X.set_window(d["spec"].get("window", -1.0)); mr = d["spec"]["mr"]; use = ["F100"]
        ti, x = np.array(d["theta_insp"]), d["x_mr"]
        with get_context("fork").Pool(128) as pool:
            res = {g: C.score(pool, ti, mr, x, grids[g]) for g in use}
        st = {g: X.stats(r) for g, r in res.items()}
        (OUT / f"{name}.json").write_text(json.dumps(dict(candidate=name, ncoef=d["ncoef"], stats=st, cases=res), indent=1))
        msg = "; ".join(f"{g} {s['median']:.3e} / {s['mean']:.3e} / {s['max']:.2e}" for g, s in st.items())
        print(f"F100 {name}: {msg}", flush=True)
        with (HERE / "CV_LOG.md").open("a") as f:
            f.write(f"* look {time.strftime('%Y-%m-%d %H:%M')}: {name} ({d['ncoef']}c): {msg}\n")


if __name__ == "__main__":
    main()
