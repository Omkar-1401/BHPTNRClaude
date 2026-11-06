"""Score the PUBLISHED calibrated BHPTNRSur2dq1e3 (Rink et al. 2024, arXiv:2407.18319; constant-in-time alpha, beta;
12 coefficients for the (2,2) mode; calibrated on 90 points, q = 3..10, chi1 = -0.6..0.8) with this repo's scorer:
NRHybSur3dq8 (2,2), NR window [-5000.1, +100] M, t0 and phi0 optimised, full-window mathcalE (+ the ISCO-split errors).

The calibrated waveform comes straight from the surrogate (calibrated=True: alpha, beta already applied, time in M), so
the model map here is the identity (alpha = beta = 1) and only the alignment is optimised.
Check: alpha, beta read from the surrogate's h5 (nr_calib_params) applied to the uncalibrated waveform with this
repo's map must give the same error on a few cases (verifies the conventions).
Points: T96, H270, and q = 2, 2.5 (8 spins).  -> results/rink_scores.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from multiprocessing import get_context

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extrap_lowq as XL                       # its get_case reads spin_dep caches, computes missing drives locally
M, D = XL.M, XL.D
import score_h270 as H
sys.path.insert(0, "/work2/11702/omkarsridhar1401/stampede3/BHPTNRSurrogate/surrogates")
H5 = "/work2/11702/omkarsridhar1401/stampede3/BHPTNRSurrogate/data/BHPTNRSur2dq1e3.h5"
T96 = [(q, c) for q in M.S.TRAIN_Q for c in M.S.CHIS]
LOWQ = XL.CASES_LIST


def calibrated_case(k):
    import BHPTNRSur2dq1e3 as ppsur
    q, chi = k; c = dict(M.CASES[k])
    t, hd = ppsur.generate_surrogate(q=q, spin1=chi, calibrated=True, modes=[(2, 2)], neg_modes=False)
    t = np.asarray(t, float); h = np.asarray(hd[(2, 2)])
    si = np.unique(np.r_[np.arange(0, len(t), M.S.SRC_STRIDE), len(t) - 1])
    c.update(t_bhpt=t, h_bhpt=h, fit_t_bhpt=t[si], fit_h_bhpt=h[si])
    return c


def _task(k):
    c = calibrated_case(k)
    one = np.ones_like(c["fit_t_bhpt"]); onef = np.ones_like(c["t_bhpt"])
    out = {}
    for align in ("insp", "full"):
        _, t0 = M.score(one, one, c, "fit", align, -50.0, wide=True)
        (ei, em, ef), t0 = M.score(onef, onef, c, "full", align, t0, wide=True)
        out.update({f"{align}_E_insp": ei, f"{align}_E_MR": em, f"{align}_E_full": ef})
    return dict(q=k[0], chi1=k[1], **out)


def _check(k):
    """alpha, beta from the h5 on the UNCALIBRATED waveform with this repo's map vs the calibrated output."""
    import h5py
    with h5py.File(H5, "r") as f:
        A = np.array(f["nr_calib_params"]["(2,2)"]["alpha"]); B = np.array(f["nr_calib_params"]["(2,2)"]["beta"])
    q, chi = k
    def val(co):
        P = sum(co[n] * q ** -(n + 1) for n in range(4)); S = sum(co[4 + n] * chi ** (n + 1) for n in range(2))
        return 1 + P * (1 + S)
    al, be = val(A), val(B)
    c = M.CASES[k]
    a = np.full_like(c["fit_t_bhpt"], al); b = np.full_like(c["fit_t_bhpt"], be)
    _, t0 = M.score(a, b, c, "fit", "full", -50.0, wide=True)
    a = np.full_like(c["t_bhpt"], al); b = np.full_like(c["t_bhpt"], be)
    (_, _, ef), _ = M.score(a, b, c, "full", "full", t0, wide=True)
    return k, al, be, ef


def main():
    keys = T96 + H.H270 + LOWQ
    for k in keys:
        M.CASES[k] = M.prepare(*k)
    with get_context("fork").Pool(128) as pool:
        rows = pool.map(_task, keys, chunksize=1)
        chk = pool.map(_check, [(5.0, 0.4), (3.0, -0.6), (8.0, 0.8)])
    R = {(r["q"], r["chi1"]): r for r in rows}
    for k, al, be, ef in chk:
        print(f"check q={k[0]} chi={k[1]:+.1f}: h5 alpha={al:.5f} beta={be:.5f} (X1^(6/5)={(k[0]/(1+k[0]))**1.2:.5f}); "
              f"E_full via repo map {ef:.4e} vs calibrated output {R[k]['full_E_full']:.4e}  ratio {ef / R[k]['full_E_full']:.4f}")
    out = {}
    for name, ks in (("T96", T96), ("H270", H.H270), ("q2", [k for k in LOWQ if k[0] == 2.0]), ("q2.5", [k for k in LOWQ if k[0] == 2.5])):
        v = np.array([R[k]["full_E_full"] for k in ks]); out[name] = [R[k] for k in ks]
        print(f"Rink et al. calibrated, {name:5s}: E_full (full-aligned) median {np.median(v):.3e} mean {v.mean():.3e} max {v.max():.2e}"
              f" | insp-aligned E_insp median {np.median([R[k]['insp_E_insp'] for k in ks]):.2e}, E_MR {np.median([R[k]['insp_E_MR'] for k in ks]):.2e}")
    (HERE / "results" / "rink_scores.json").write_text(json.dumps(out, indent=1))
    print("wrote results/rink_scores.json")


if __name__ == "__main__":
    main()
