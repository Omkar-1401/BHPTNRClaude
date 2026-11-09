"""Per-mode FD matches at chi1 = 0, Zhong et al. 2025 Fig. 14 style: (2,2), (3,3), (4,4) each matched on its own against
NRHybSur3dq8 (aLIGO ZDHP, maximised over time and phase, mean over 4 signal phases; fd_core.match), for q = 3..8.
Our higher modes are the zero-parameter transfer of the (2,2) calibration (hm_run.maps).  f_low = 20 Hz unless the mode's
own frequency at the end of the start taper requires more (flagged).
    python fdm/mode_match.py MODEL     -> fdm/results/modes_<MODEL>.json            (M = 100, 200: fd_run.MTOT)
    FDM_MTOT=50 python fdm/mode_match.py MODEL  -> fdm/results/modes_<MODEL>_M50.json   (other masses; the window
    -5000..+100 M then starts above 20 Hz, so f_low rises per mode and is recorded per point)
"""
from __future__ import annotations
import json, os, time
import multiprocessing as mp
import numpy as np

import hm_run as HR                             # model setup (sys.argv[1]) + maps()
import fd_core as FC
FR = HR.FR
MODES = [(2, 2), (3, 3), (4, 4)]
PTS = [k for k in FR.PTS if k[1] == 0.0]
_env = os.environ.get("FDM_MTOT")
MASSES = [float(x) for x in _env.split(",")] if _env else list(FR.MTOT)
SUFFIX = "" if MASSES == list(FR.MTOT) else "_" + "_".join(f"M{int(M)}" for M in MASSES)


def _work(k):
    tc, nr, mod = HR.maps(k)
    row = dict(q=k[0], chi1=k[1])
    for mo in MODES:
        for M in MASSES:
            r = FC.match(tc, nr[mo], mod[mo], M)
            row[f"{mo[0]}{mo[1]}_M{int(M)}"] = dict(O=1 - r["mm"], mm=r["mm"], f_low=r["f_low"], flagged=r["flagged"])
    return row


def main():
    t0 = time.time()
    with mp.get_context("fork").Pool(min(96, len(PTS))) as pool:
        e = np.array(pool.map(FR._E, FR.CHK))
        r = e / np.array([FR.REF[k] for k in FR.CHK])
        print(f"{HR.MODEL}: reproduce stored T96 on {len(FR.CHK)} points: {r.min():.5f}..{r.max():.5f}", flush=True)
        assert np.allclose(r, 1, atol=2e-3), r
        rows = pool.map(_work, PTS, chunksize=1)
    if 100.0 in MASSES:                                                              # (2,2) must equal the fd_run result
        f22 = json.loads((FR.OUT / f"fd_{HR.MODEL}.json").read_text())["rows"]
        ref = {(x["q"], x["chi1"]): x["M100"]["mm"] for x in f22}
        d = max(abs(x["22_M100"]["mm"] / ref[(x["q"], x["chi1"])] - 1) for x in rows)
        print(f"(2,2) vs fd_run: max relative difference {d:.1e}", flush=True); assert d < 1e-6, d
    for mo in MODES:
        for M in MASSES:
            v = [x[f"{mo[0]}{mo[1]}_M{int(M)}"]["O"] for x in rows]; fl = [x[f"{mo[0]}{mo[1]}_M{int(M)}"]["f_low"] for x in rows]
            print(f"  {mo} M = {M:.0f}: O {min(v):.4f}..{max(v):.4f}; f_low {min(fl):.1f}..{max(fl):.1f} Hz", flush=True)
    (FR.OUT / f"modes_{HR.MODEL}{SUFFIX}.json").write_text(json.dumps(dict(model=HR.MODEL, label=FR.LABEL, masses=MASSES,
                                                                          rows=rows), indent=1))
    print(f"done {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
