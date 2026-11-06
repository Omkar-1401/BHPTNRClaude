"""Fresh-G96 check, part 1 (user, 2026-10-18): regenerate the 96 T96 waveform pairs + energy/flux drives from scratch
(TS_FRESH=1 -> cache_fresh/), exactly as the new grid points were built, and compare every array with spin_dep's old cache.
Part 2 is the harness gates run with TS_FRESH=1 (the shipped models must reproduce their stored T96 numbers on the
fresh waveforms) -- see ../ls6/training_size/run_fresh96.slurm.

    TS_FRESH=1 python check_fresh96.py
"""
from __future__ import annotations
import os, sys
from multiprocessing import get_context

import numpy as np

assert os.environ.get("TS_FRESH") == "1", "run with TS_FRESH=1"
import grids as G                              # noqa: E402

G.WF_FRESH.mkdir(parents=True, exist_ok=True); G.LOSS_FRESH.mkdir(parents=True, exist_ok=True)


def _build(k):
    G.FS.build_case(*k)
    return k


def compare(a_path, b_path):
    """max over arrays of max|a - b| / max|b| (non-numeric entries must be equal)."""
    A, B = np.load(a_path, allow_pickle=True), np.load(b_path, allow_pickle=True)
    assert sorted(A.files) == sorted(B.files), (a_path.name, A.files, B.files)
    worst = (0.0, "")
    for n in A.files:
        a, b = A[n], B[n]
        if a.dtype.kind in "fc" and b.dtype.kind in "fc":
            if a.shape != b.shape:
                return (np.inf, f"{n} shape {a.shape} vs {b.shape}")
            d = float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)) if b.size else 0.0
            worst = max(worst, (d, n))
        elif not np.array_equal(a, b):
            return (np.inf, f"{n} differs")
    return worst


def main():
    with get_context("fork").Pool(96) as pool:
        pool.map(_build, G.T96, chunksize=1)
    rows = []
    for q, c in G.T96:
        tag = f"q{q:.6f}_chi{c:+.3f}.npz"
        rows.append(((q, c), compare(G.WF_FRESH / f"wf_{tag}", G.WF_SHARED / f"wf_{tag}"),
                     compare(G.LOSS_FRESH / f"loss_{tag}", G.LOSS_SHARED / f"loss_{tag}")))
    for lab, i in (("waveform", 1), ("drive", 2)):
        w = max(rows, key=lambda r: r[i][0])
        print(f"{lab:8s}: worst relative difference fresh vs old cache {w[i][0]:.3e} (array {w[i][1]!r}, q={w[0][0]}, "
              f"chi1={w[0][1]:+.1f}); cases > 1e-10: {sum(r[i][0] > 1e-10 for r in rows)}/96", flush=True)


if __name__ == "__main__":
    main()
