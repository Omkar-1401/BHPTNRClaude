"""EXPLORATION pass 9 -- does the x_pp clip (XCLIP = 0.26) cost the chi1 = +0.6, +0.8 columns?
At chi1 = +0.8 the ppBHPT's own ISCO is at x_pp ~ 0.31, so the clip freezes the spin-orbit factor and x over the late
inspiral -- where that column's floor loses (pass 6 band analysis: [-50, -15] M).  Re-runs the p5b per-case M2 floor at
chi1 = +0.6, +0.8 (all 12 q) with XCLIP = 0.6 (no effective clip before the peak).  Writes exploration/p9/p9_xclip.json."""
import json, sys
from pathlib import Path
import multiprocessing as mp
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p5b_floor as F
F.S.XCLIP = 0.6
OUT = HERE / "p9"; OUT.mkdir(exist_ok=True)

if __name__ == "__main__":
    keys = [(q, c) for q in F.S.TRAIN_Q for c in (0.6, 0.8)]
    old = {(r["q"], r["chi1"]): r for r in json.loads((HERE / "p5b" / "p5b_floor.json").read_text())}
    with mp.get_context("fork").Pool(len(keys)) as pool:
        rows = pool.map(F.fit_case, keys)
    (OUT / "p9_xclip.json").write_text(json.dumps(rows, indent=1))
    for c in (0.6, 0.8):
        a = [old[(r["q"], r["chi1"])]["M2"]["E"] for r in rows if r["chi1"] == c]
        b = [r["M2"]["E"] for r in rows if r["chi1"] == c]
        fa = [old[(r["q"], r["chi1"])]["FLUX"]["E"] for r in rows if r["chi1"] == c]
        fb = [r["FLUX"]["E"] for r in rows if r["chi1"] == c]
        print(f"XCLIP chi1 {c:+.1f}: M2 floor median 0.26 -> 0.6: {np.median(a):.3e} -> {np.median(b):.3e}; "
              f"FLUX floor {np.median(fa):.3e} -> {np.median(fb):.3e}", flush=True)
