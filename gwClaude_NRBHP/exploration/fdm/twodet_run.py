"""Rink et al. 2024 Fig. 12 style: two-detector FD mismatch vs total mass, face-on (iota = 0, phi = 0), all 8 cached modes
(face-on only the m = 2 harmonics survive: (2,2), (3,2), (4,2)).  Same 126 points and model waveforms as hm_run.py.
Two band choices per point and mass: f_min = Rink's (the waveform's (2,2) frequency after a 1.5-cycle taper) and
f_min = max(20 Hz, that).  twodet_core.py has the definitions.
Gates before anything is written: the model's stored T96 scores (fd_run's gate); on q = 5, chi1 = 0, M = 50:
  NR vs NR = 0; the mismatch is invariant under a rotation of the SIGNAL's polarization (two detectors see both
  polarizations equally); the analytic polarization maximisation equals an explicit rotation of the template on a grid.
    python fdm/twodet_run.py MODEL     MODEL in p40 | p40lean | p49b12 | p49b9 | flux_orbit | rink
    -> fdm/results/twodet_<MODEL>.json
"""
from __future__ import annotations
import json, time
import multiprocessing as mp
import numpy as np

import hm_run as HR                             # model setup (sys.argv[1]) + maps(); imports fd_run
import twodet_core as TD
FR = HR.FR; MODEL = HR.MODEL
MASSES = [10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0]
FMINS = {"rink": "rink", "f20": 20.0}


def _work(k):
    try:
        tc, nr, mod = HR.maps(k)
        row = dict(q=k[0], chi1=k[1])
        for M in MASSES:
            for name, fm in FMINS.items():
                row[f"{name}_M{int(M)}"] = TD.match2(tc, nr, mod, M, iota=0.0, phi_s=0.0, fmin=fm)
        return row
    except Exception as exc:
        print(f"  [fail] {k}: {exc!r}", flush=True); return dict(q=k[0], chi1=k[1], error=repr(exc))


def selftest(k):
    tc, nr, mod = HR.maps(k); M = 50.0
    z = TD.match2(tc, nr, nr, M)["mm"]
    a = TD.match2(tc, nr, mod, M)["mm"]; b = TD.match2(tc, nr, mod, M, psi_s=0.7)["mm"]
    br, an = TD.brute_psi(tc, nr, mod, M)
    st = TD.setup(tc, nr[(2, 2)], M)
    print(f"selftest {k} M=50: NR vs NR {z:.1e}; signal-polarization invariance {a:.8e} vs {b:.8e}; "
          f"explicit psi grid {br:.8e} vs analytic {an:.8e}; band {st['f_min']:.1f}..{st['f_max']:.0f} Hz at fs {st['fs']:.0f}",
          flush=True)
    assert z < 1e-8 and abs(a / b - 1) < 1e-6 and abs(br / an - 1) < 1e-5, (z, a, b, br, an)


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
    for name in FMINS:
        for M in MASSES:
            v = np.array([x[f"{name}_M{int(M)}"]["mm"] for x in ok])
            fl = [x[f"{name}_M{int(M)}"]["f_min"] for x in ok]
            print(f"  {name:4s} M = {M:5.0f}: median {np.median(v):.3e} max {v.max():.2e}; f_min {min(fl):.1f}..{max(fl):.1f} Hz",
                  flush=True)
    (FR.OUT / f"twodet_{MODEL}.json").write_text(json.dumps(dict(
        model=MODEL, label=FR.LABEL, iota=0.0, phi=0.0, masses=MASSES, fmins=list(FMINS), pad=TD.PAD,
        psd="aLIGOZeroDetHighPower", rows=rows), indent=1))
    print(f"{MODEL}: {len(ok)}/{len(rows)} points, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
