"""Untouched TEST set for the final comparison (2026-10-13; user: H270 was used for model selection, so the final
comparison needs points no fit and no choice has seen, as in an ML train / validation / test split).

    R100  100 points uniform in 3 < q < 8, -0.6 < chi1 < 0.8 (the T96 / H270 box, interpolation)
    L20    20 points uniform in 2.5 < q < 3, -0.6 < chi1 < 0.8 (below the training box, extrapolation)

Drawn once with numpy default_rng(SEED), rounded to the cache-name precision (q 1e-6, chi1 1e-3).  The seed is not
F100's (spin_dep_split/fresh_set.py, the other session's set), so the two sets are independent.
RULE: no fit, no prune and no model choice may use these points.  They are scored once per final candidate
(t100_score.py) and every look is recorded in EXPLORATION.md.

Caches: exploration/t100/cache/{wf,loss}.  Importing this module patches FS.build_case so TEST keys are built / read
there and every other key still reads the spin_dep caches (read-only).  Nothing is written into spin_dep/.

    python t100_set.py           # build the 120 waveform pairs + energy / flux drives (one-off, compute node)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from multiprocessing import get_context

import numpy as np

HERE = Path(__file__).resolve().parent; REPO = HERE.parent
for p in (HERE, REPO / "spin_dep", REPO):
    sys.path.insert(0, str(p))
import fit_pn_anchored_spin as S              # noqa: E402
import fit_fluxanchored_spin as FS            # noqa: E402
import test_guard as TG                       # noqa: E402
TG.ALLOW = True                               # the only module that may build / score the sealed points

SEED = 8675309
CACHE = HERE / "t100" / "cache"; WF_DIR = CACHE / "wf"; LOSS_DIR = CACHE / "loss"


def _draw(n, qlo, qhi, rng):
    out = set()
    while len(out) < n:
        q, c = round(float(rng.uniform(qlo, qhi)), 6), round(float(rng.uniform(-0.6, 0.8)), 3)
        if qlo < q < qhi and -0.6 < c < 0.8:
            out.add((q, c))
    return sorted(out)


_rng = np.random.default_rng(SEED)
R100 = _draw(100, 3.0, 8.0, _rng)
L20 = _draw(20, 2.5, 3.0, _rng)
TEST = R100 + L20
_T = set(TEST)
_WF0, _LOSS0 = S.CACHE, FS.LOSSCACHE
_build0 = FS.build_case


def build_case(q, chi1):
    test = (round(float(q), 6), round(float(chi1), 3)) in _T
    S.CACHE, FS.LOSSCACHE = (WF_DIR, LOSS_DIR) if test else (_WF0, _LOSS0)
    try:
        return _build0(q, chi1)
    finally:
        S.CACHE, FS.LOSSCACHE = _WF0, _LOSS0


FS.build_case = build_case


def min_distance():
    """Smallest (q, chi1) distance from a TEST point to T96 / H270 (q scaled by 1/5, chi1 by 1/1.4: unit box)."""
    T96 = [(q, c) for q in S.TRAIN_Q for c in S.CHIS]
    QS, CS = np.linspace(2, 8, 20), np.linspace(-0.6, 0.8, 20)
    H270 = [(float(q), float(c)) for q in QS for c in CS[1:-1] if 3 < q < 8]
    ref = np.array(T96 + H270); t = np.array(TEST)
    d = np.hypot((t[:, None, 0] - ref[None, :, 0]) / 5, (t[:, None, 1] - ref[None, :, 1]) / 1.4)
    return float(d.min())


def _build(k):
    t = time.time(); build_case(*k); return k, time.time() - t


def main():
    assert len(R100) == 100 and len(L20) == 20 and len(_T) == 120
    WF_DIR.mkdir(parents=True, exist_ok=True); LOSS_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE.parent / "TEST_points.txt").write_text(
        f"# seed {SEED}; R100 then L20\n" + "".join(f"{q:.6f} {c:+.3f}\n" for q, c in TEST))
    print(f"min unit-box distance to T96/H270: {min_distance():.4f}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(32) as pool:
        for i, (k, dt) in enumerate(pool.imap_unordered(_build, TEST), 1):
            print(f"{i:3d}/120 q={k[0]:.6f} chi1={k[1]:+.3f}  {dt:5.1f} s", flush=True)
    n = len(list(WF_DIR.glob("wf_*.npz"))), len(list(LOSS_DIR.glob("loss_*.npz")))
    print(f"caches: {n[0]} waveform pairs, {n[1]} drives; {(time.time() - t0) / 60:.1f} min", flush=True)
    assert n == (120, 120), n


if __name__ == "__main__":
    main()
