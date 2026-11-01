"""EXPLORATION pass 44 -- the pass-40 leader beyond the training range: q = 9, 10 (section 8 (c) prediction at q = 10).

No fitting.  q = 9, 10 x chi1 in the T96 spins (16 cases).  NRHybSur3dq8 is itself extrapolated above q = 8 (its training
range), so these numbers mix model and NR-surrogate extrapolation; read them as an upper bound on the model's error.
Waveforms and drives are built into exploration/p44/cache (never spin_dep/).  The stored T96 is reproduced first.
    python p44_highq.py      -> exploration/p44/p44.json
"""
from __future__ import annotations
import json, sys
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_guard as TG                         # noqa: E402  (the sealed points stay refused)
S, FS = TG._S, TG._FS
OUT = HERE / "p44"; WF, LOSS = OUT / "cache" / "wf", OUT / "cache" / "loss"
WF.mkdir(parents=True, exist_ok=True); LOSS.mkdir(parents=True, exist_ok=True)
CHIS = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
KEYS = [(q, c) for q in (9.0, 10.0) for c in CHIS]
_KS = set(KEYS)
_b0, _wf0, _l0 = FS.build_case, S.CACHE, FS.LOSSCACHE


def build_case(q, chi1):
    mine = (round(float(q), 6), round(float(chi1), 3)) in _KS
    S.CACHE, FS.LOSSCACHE = (WF, LOSS) if mine else (_wf0, _l0)
    try:
        return _b0(q, chi1)
    finally:
        S.CACHE, FS.LOSSCACHE = _wf0, _l0


FS.build_case = build_case
sys.argv = [sys.argv[0], "L4"]
import p40_r2closed as C                        # noqa: E402
P6 = C.P6
R = json.loads((HERE / "p39" / "p39_L4.json").read_text()); TH = list(map(float, R["theta"]))


def _build(k):
    FS.build_case(*k); return k


def _full(k):
    return P6._full((TH, k))


def main():
    with mp.get_context("fork").Pool(16) as pool:
        pool.map(_build, KEYS)
    P6.get = C.L.get; P6.ab = C.L.ab; C.use("O2n")
    ref = json.loads((HERE / "p40" / "p40.json").read_text())["O2n"]["T96_cases"]
    chk = P6.T96[::8]
    with mp.get_context("fork").Pool(48) as pool:
        r = np.array(pool.map(_full, chk)) / np.array([ref[f"{q:.6f}_{c:+.4f}"] for q, c in chk])
        print(f"reproduce pass 40 T96: {r.min():.4f}..{r.max():.4f}", flush=True)
        assert np.allclose(r, 1, atol=2e-3), r
        e = dict(zip(KEYS, pool.map(_full, KEYS)))
    out = {f"{q:.6f}_{c:+.4f}": float(v) for (q, c), v in e.items()}
    (OUT / "p44.json").write_text(json.dumps(dict(model="pass 40", cases=out), indent=1))
    t8 = {c: ref[f"8.000000_{c:+.4f}"] for c in CHIS}
    print("chi1 :  q = 8 (training edge) | q = 9 | q = 10", flush=True)
    for c in CHIS:
        print(f"{c:+.1f} :  {t8[c]:.2e} | {e[(9.0, c)]:.2e} | {e[(10.0, c)]:.2e}", flush=True)
    for q in (9.0, 10.0):
        v = np.array([e[(q, c)] for c in CHIS]); print(f"q = {q:g}: median {np.median(v):.3e} max {v.max():.3e}", flush=True)
    v8 = np.array(list(t8.values())); print(f"q = 8: median {np.median(v8):.3e} max {v8.max():.3e}", flush=True)


if __name__ == "__main__":
    main()
