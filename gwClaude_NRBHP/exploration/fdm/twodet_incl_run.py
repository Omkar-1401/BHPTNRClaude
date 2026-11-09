"""Two-detector FD mismatch (twodet_core.match2, Rink et al. 2024 / Blackman et al. 2017 procedure) at INCLINED
orientations, where the higher modes contribute: iota = pi/3 and pi/2, all 8 cached modes, M = 10..300 Msun, same 126
points and model waveforms as hm_run.py, f_min by Rink's rule.
Two detectors see both polarizations, so the signal's polarization drops out (gate in twodet_run); the signal's orbital
phase phi_s does not, so it is sampled at N_PHI_S = 8 values.  Reported per point: the SNR^3-weighted mismatch over phi_s
(Harry et al. 2018; rho = network SNR of the signal), plus the mean and the worst phi_s.  The template's orbital phase
is optimised (grid of 24 + bounded refinement), its polarization analytically, time by FFT.
Gates before anything is written: stored T96 (fd_run's gate); on q = 5, chi1 = 0, M = 50, iota = pi/3: NR vs NR = 0;
the mismatch is invariant under a rotation of the signal's polarization; face-on output equals twodet_<model>.json.
    python fdm/twodet_incl_run.py MODEL   -> fdm/results/twodet_incl_<MODEL>.json
"""
from __future__ import annotations
import json, time
import multiprocessing as mp
import numpy as np

import hm_run as HR
import twodet_core as TD
FR = HR.FR; MODEL = HR.MODEL
MASSES = [10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0]
INCS = {"pi/3": np.pi / 3, "pi/2": np.pi / 2}
N_PHI_S = 8
PHIS = [2 * np.pi * j / N_PHI_S for j in range(N_PHI_S)]


def combine(res):
    o = np.array([1 - r["mm"] for r in res]); rho = np.array([r["rho"] for r in res])
    w = 1.0 - (np.sum(o ** 3 * rho ** 3) / np.sum(rho ** 3)) ** (1 / 3)
    return dict(mm=float(w), mm_mean=float(np.mean(1 - o)), mm_worst=float(np.max(1 - o)),
                mm_phi=[float(1 - x) for x in o], snr_rel=(rho / rho.max()).tolist(), f_min=res[0]["f_min"])


def _work(k):
    try:
        tc, nr, mod = HR.maps(k)
        row = dict(q=k[0], chi1=k[1])
        for name, io in INCS.items():
            for M in MASSES:
                row[f"{name}_M{int(M)}"] = combine([TD.match2(tc, nr, mod, M, iota=io, phi_s=p) for p in PHIS])
        return row
    except Exception as exc:
        print(f"  [fail] {k}: {exc!r}", flush=True); return dict(q=k[0], chi1=k[1], error=repr(exc))


def selftest(k):
    tc, nr, mod = HR.maps(k); M = 50.0; io = np.pi / 3
    z = TD.match2(tc, nr, nr, M, iota=io, phi_s=0.4)["mm"]
    a = TD.match2(tc, nr, mod, M, iota=io, phi_s=0.4)["mm"]; b = TD.match2(tc, nr, mod, M, iota=io, phi_s=0.4, psi_s=0.7)["mm"]
    f0 = json.loads((FR.OUT / f"twodet_{MODEL}.json").read_text())["rows"]
    ref = {(r["q"], r["chi1"]): r["rink_M50"]["mm"] for r in f0 if "error" not in r}[k]
    c = TD.match2(tc, nr, mod, M, iota=0.0)["mm"]
    print(f"selftest {k} M=50 iota=pi/3: NR vs NR {z:.1e}; signal-polarization invariance {a:.8e} vs {b:.8e}; "
          f"face-on {c:.8e} vs twodet_{MODEL}.json {ref:.8e}", flush=True)
    assert z < 1e-8 and abs(a / b - 1) < 1e-6 and abs(c / ref - 1) < 1e-9, (z, a, b, c, ref)


def main():
    t0 = time.time()
    with mp.get_context("fork").Pool(96) as pool:
        e = np.array(pool.map(FR._E, FR.CHK))
        r = e / np.array([FR.REF[k] for k in FR.CHK])
        print(f"{MODEL}: reproduce stored T96 on {len(FR.CHK)} points: {r.min():.5f}..{r.max():.5f}", flush=True)
        assert np.allclose(r, 1, atol=2e-3), r
        selftest(FR.PTS[len(FR.QS) * 2 + 8])                                  # chi1 = 0, q = 5
        rows = pool.map(_work, FR.PTS, chunksize=1)
    ok = [x for x in rows if "error" not in x]
    for name in INCS:
        for M in MASSES:
            v = np.array([x[f"{name}_M{int(M)}"]["mm"] for x in ok])
            print(f"  iota {name} M = {M:5.0f}: median {np.median(v):.3e} max {v.max():.2e}", flush=True)
    (FR.OUT / f"twodet_incl_{MODEL}.json").write_text(json.dumps(dict(
        model=MODEL, label=FR.LABEL, inclinations=list(INCS), phis=PHIS, masses=MASSES, fmin="rink", pad=TD.PAD,
        psd="aLIGOZeroDetHighPower", weighting="SNR^3 over phi_s (Harry et al. 2018)", rows=rows), indent=1))
    print(f"{MODEL}: {len(ok)}/{len(rows)} points, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
