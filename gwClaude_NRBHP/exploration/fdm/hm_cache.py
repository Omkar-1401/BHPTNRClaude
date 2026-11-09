"""Higher-mode waveforms for the FD study (one-off, compute node): per point, the 8 modes common to BHPTNRSur2dq1e3 and
NRHybSur3dq8, MODES = (2,2),(2,1),(3,1),(3,2),(3,3),(4,2),(4,3),(4,4)  (m > 0; m < 0 by h_{l,-m} = (-1)^l conj(h_lm)).
Stored: raw ppBHPT modes (calibrated=False, m1 time), the published calibrated BHPTNRSur2dq1e3 modes (Rink et al. 2024,
calibrated=True, M time) and NRHybSur3dq8 modes on the (2,2) cases' NR window and grid (dt = 0.1 M, f_low = 5e-3).
Gate: the regenerated (2,2) raw ppBHPT and NR modes equal the (2,2) case pair (FS.build_case) to 1e-6 relative (surrogate\nre-evaluation differs at 1e-9..1e-7); the cached case's (2,2) is then stored, so (2,2) is identical to the single-mode study.
    python fdm/hm_cache.py      -> fdm/cache/hm/hm_q*_chi*.npz
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from multiprocessing import get_context
import numpy as np

import fd_route
S, FS = fd_route.S, fd_route.FS
HERE = Path(__file__).resolve().parent
HM = HERE / "cache" / "hm"; HM.mkdir(parents=True, exist_ok=True)
MODES = [(2, 2), (2, 1), (3, 1), (3, 2), (3, 3), (4, 2), (4, 3), (4, 4)]
QS = [round(float(q), 6) for q in np.linspace(3.0, 8.0, 21)]
CHIS = [-0.6, -0.3, 0.0, 0.3, 0.6, 0.8]
PTS = [(q, c) for c in CHIS for q in QS]
path = lambda q, c: HM / f"hm_q{q:.6f}_chi{c:+.3f}.npz"
_NR = None


def build(k):
    global _NR
    q, c = k; p = path(q, c)
    if p.exists():
        return k, 0.0
    t0 = time.time()
    import BHPTNRSur2dq1e3 as ppsur, gwsurrogate
    if _NR is None:
        _NR = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
    tr, hr = ppsur.generate_surrogate(q=q, spin1=c, calibrated=False, modes=MODES, neg_modes=False)
    tc, hc = ppsur.generate_surrogate(q=q, spin1=c, calibrated=True, modes=MODES, neg_modes=False)
    tn, hn, _ = _NR(q, [0, 0, c], [0, 0, 0.0], dt=0.1, f_low=S.F_LOW)
    m = (tn >= S.WF.qdep.NR_T_START) & (tn <= S.WF.qdep.NR_T_END)
    st = dict(t_pp=np.asarray(tr, float), t_rink=np.asarray(tc, float), t_nr=np.asarray(tn, float)[m])
    for l, mm in MODES:
        st[f"pp_{l}{mm}"] = np.asarray(hr[(l, mm)]); st[f"rink_{l}{mm}"] = np.asarray(hc[(l, mm)])
        st[f"nr_{l}{mm}"] = np.asarray(hn[(l, mm)])[m]
    case = FS.build_case(q, c)                                   # gate against the (2,2) case pair
    for a, b, name in ((st["t_pp"], case["t_bhpt"], "t_pp"), (st["pp_22"], case["h_bhpt"], "pp_22"),
                       (st["t_nr"], case["t_nr"], "t_nr"), (st["nr_22"], case["h_nr"], "nr_22")):
        assert a.shape == b.shape and np.max(np.abs(a - b)) <= 1e-6 * np.max(np.abs(b)), (k, name)   # re-evaluation noise ~1e-7
    st["pp_22"], st["nr_22"] = case["h_bhpt"], case["h_nr"]     # the (2,2) used everywhere else, bit for bit
    np.savez_compressed(p, **st)
    return k, time.time() - t0


def main():
    t0 = time.time()
    with get_context("fork").Pool(24) as pool:
        for i, (k, dt) in enumerate(pool.imap_unordered(build, PTS), 1):
            if i % 20 == 0 or dt > 60:
                print(f"{i:3d}/{len(PTS)} {k}  {dt:.1f} s", flush=True)
    n = len(list(HM.glob("hm_*.npz")))
    print(f"{n} HM files, gate passed on every new file; {(time.time() - t0) / 60:.1f} min", flush=True)
    assert n >= len(PTS)


if __name__ == "__main__":
    main()
