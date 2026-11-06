"""Fresh held-out set F100 (2026-10-13; user: never select on H270, it is held out).

H270 was used to choose between split variants in EXPLORE_LOG rounds 1-4, so it no longer gives an untouched error
estimate for them.  F100 replaces it as the final test: 100 (q, chi1) points drawn once, uniform in 3 < q < 8,
-0.6 < chi1 < 0.8 (the H270 box), numpy default_rng(20261013), rounded to the cache-name precision (q 1e-6, chi1 1e-3).
RULE: no fit and no choice may use F100.  Score it once, for the final pick(s) only, and record every look in CV_LOG.md.

Caches live in spin_dep_split/fresh_cache/{wf,loss}; nothing is written into spin_dep/.  Importing this module installs
a get_case that serves F100 keys from there and every other key through extrap_lowq.get_case.

    python fresh_set.py          # build the 100 waveform pairs + energy/flux drives (one-off)
"""
from __future__ import annotations
import sys
from pathlib import Path
from multiprocessing import get_context

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extrap_lowq as XL
D = XL.D
WF_DIR = HERE / "fresh_cache" / "wf"; LOSS_DIR = HERE / "fresh_cache" / "loss"


def draw(n=100, seed=20261013):
    r = np.random.default_rng(seed)
    pts = sorted({(round(float(q), 6), round(float(c), 3)) for q, c in zip(r.uniform(3, 8, 4 * n), r.uniform(-0.6, 0.8, 4 * n))
                  if 3 < q < 8 and -0.6 < c < 0.8}, key=lambda k: (k[0], k[1]))
    pts = [pts[i] for i in r.permutation(len(pts))[:n]]        # dedup, then a fixed random subset of n
    return sorted(pts)


F100 = draw()
_F = set(F100)


def get_case(q, chi, where="grid"):
    if (round(float(q), 6), round(float(chi), 3)) not in _F:
        return XL.get_case(q, chi, where)
    S, FS = D.S, D.FS
    S.CACHE, FS.LOSSCACHE = WF_DIR, LOSS_DIR
    return FS.build_case(q, chi)


D.get_case = get_case


def _build(k):
    get_case(*k)
    return k


def main():
    assert len(F100) == 100 and len(_F) == 100
    WF_DIR.mkdir(parents=True, exist_ok=True); LOSS_DIR.mkdir(parents=True, exist_ok=True)
    (HERE / "fresh_cache" / "F100.txt").write_text("".join(f"{q:.6f} {c:+.3f}\n" for q, c in F100))
    with get_context("fork").Pool(32) as pool:
        for i, k in enumerate(pool.imap_unordered(_build, F100), 1):
            print(f"{i:3d}/100 built q={k[0]:.6f} chi1={k[1]:+.3f}", flush=True)
    n = len(list(WF_DIR.glob("wf_*.npz"))), len(list(LOSS_DIR.glob("loss_*.npz")))
    print(f"caches: {n[0]} waveform pairs, {n[1]} drives", flush=True)
    assert n == (100, 100), n


if __name__ == "__main__":
    main()
