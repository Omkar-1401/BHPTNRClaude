"""Training-size study (2026-10-18, user request): how do model mismatches change with 24 / 48 / 96 / 192 training points?

Grids (user choice: keep the T96 aspect, ~3 q : 2 chi1), q in [3, 8] even in nu, chi1 in [-0.6, 0.8] evenly spaced:
    G24  =  6 q x  4 chi1          G48 = 8 q x 6 chi1
    G96  = T96 itself (12 q x 8 chi1; its q are the historical 'even-nu' list, even in nu to ~1e-3, not exactly)
    G192 = 16 q x 12 chi1
Keys are rounded to the cache precision (q 6 dp, chi1 3 dp): the waveform cache file names use exactly that.

Scoring set V300 (user choice: a fresh set, not H270): 300 points uniform in 3 < q < 8, -0.6 < chi1 < 0.8, drawn once
here.  RULE: V300 is used only to SCORE the refitted models.  No fit, start point or form choice may use it; any choice
inside this study (e.g. which of two start points to keep) is made on the training objective.
The sealed exploration test set (exploration/t100) is guarded: every builder call goes through exploration/test_guard.

Caches: a (q, chi1) already in spin_dep/{spin_wf_cache, flux_loss_cache} is read from there; anything else is built into
training_size/cache/{wf,loss}.  Nothing is written into spin_dep/.  Importing this module installs that routing.

    python grids.py            # print the grids, check the guard, build every missing case (run on a compute node)
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from multiprocessing import get_context

import numpy as np

HERE = Path(__file__).resolve().parent; REPO = HERE.parent
for p in (REPO / "exploration", REPO / "spin_dep", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import fit_pn_anchored_spin as S              # noqa: E402
import fit_fluxanchored_spin as FS            # noqa: E402
import test_guard as TG                       # noqa: E402  (wraps S/FS.build_case: refuses sealed test points)

WF_SHARED, LOSS_SHARED = REPO / "spin_dep" / "spin_wf_cache", REPO / "spin_dep" / "flux_loss_cache"
WF_DIR, LOSS_DIR = HERE / "cache" / "wf", HERE / "cache" / "loss"
# TS_FRESH=1: never read spin_dep's caches; keys found there are rebuilt into cache_fresh/ (T96 regenerated exactly as the
# new grid points were, to check old cache == fresh build and that the shipped models reproduce on fresh waveforms)
FRESH = os.environ.get("TS_FRESH") == "1"
WF_FRESH, LOSS_FRESH = HERE / "cache_fresh" / "wf", HERE / "cache_fresh" / "loss"
Q_LO, Q_HI, C_LO, C_HI = 3.0, 8.0, -0.6, 0.8


def nu_of(q):
    return q / (1.0 + q) ** 2


def q_of_nu(nu):
    return (1.0 - 2.0 * nu + np.sqrt(1.0 - 4.0 * nu)) / (2.0 * nu)


def key(q, c):
    return (round(float(q), 6), round(float(c), 3))


def grid(nq, nc):
    qs = [q_of_nu(n) for n in np.linspace(nu_of(Q_LO), nu_of(Q_HI), nq)]
    qs[0], qs[-1] = Q_LO, Q_HI                     # exact ends (q_of_nu round-off)
    return [key(q, c) for q in qs for c in np.linspace(C_LO, C_HI, nc)]


T96 = [key(q, c) for q in S.TRAIN_Q for c in S.CHIS]
GRIDS = {24: grid(6, 4), 48: grid(8, 6), 96: T96, 192: grid(16, 12)}


def _draw_v300(n=300):
    r = np.random.default_rng(20261018)
    pts = sorted({key(q, c) for q, c in zip(r.uniform(Q_LO, Q_HI, 4 * n), r.uniform(C_LO, C_HI, 4 * n))
                  if Q_LO < q < Q_HI and C_LO < c < C_HI})
    return sorted(pts[i] for i in r.permutation(len(pts))[:n])


V300 = _draw_v300()

_orig_S, _orig_FS = S.build_case, FS.build_case    # already wrapped by test_guard


def _route(q, chi1):
    tag = f"q{q:.6f}_chi{chi1:+.3f}.npz"
    S.CACHE = (WF_FRESH if FRESH else WF_SHARED) if (WF_SHARED / f"wf_{tag}").exists() else WF_DIR
    FS.LOSSCACHE = (LOSS_FRESH if FRESH else LOSS_SHARED) if (LOSS_SHARED / f"loss_{tag}").exists() else LOSS_DIR


def _S_build(q, chi1, *a, **k):
    _route(q, chi1)
    return _orig_S(q, chi1, *a, **k)


def _FS_build(q, chi1, *a, **k):
    _route(q, chi1)
    return _orig_FS(q, chi1, *a, **k)


_S_build._sealed_guard = _FS_build._sealed_guard = True
S.build_case, FS.build_case = _S_build, _FS_build


def all_keys():
    return sorted(set(V300).union(*GRIDS.values()))


def _build(k):
    FS.build_case(*k)
    return k


def main():
    WF_DIR.mkdir(parents=True, exist_ok=True); LOSS_DIR.mkdir(parents=True, exist_ok=True)
    for n, g in GRIDS.items():
        assert len(g) == len(set(g)) == n, n
        qs = sorted({k[0] for k in g}); cs = sorted({k[1] for k in g})
        print(f"G{n}: {len(qs)} q x {len(cs)} chi1   q = {qs}\n      chi1 = {cs}\n      "
              f"d(nu) = {np.round(np.diff(nu_of(np.array(qs))), 5).tolist()}", flush=True)
    assert len(V300) == 300
    keys = all_keys()
    for k in keys:
        TG.check(*k)                                # raises on a sealed point (the message never prints it)
    print(f"guard: all {len(keys)} keys clear of the sealed set", flush=True)
    tr = set().union(*GRIDS.values())
    dmin = min(np.hypot((a[0] - b[0]) / 5, (a[1] - b[1]) / 1.4) for a in V300 for b in tr)
    print(f"V300: min unit-box distance to any training grid point {dmin:.4f}; overlap {len(set(V300) & tr)}", flush=True)
    (HERE / "V300.txt").write_text("".join(f"{q:.6f} {c:+.3f}\n" for q, c in V300))
    miss = [k for k in keys if not (WF_SHARED / f"wf_q{k[0]:.6f}_chi{k[1]:+.3f}.npz").exists()]
    print(f"{len(keys)} keys, {len(miss)} not in spin_dep cache -> building into {WF_DIR.parent}", flush=True)
    with get_context("fork").Pool(64) as pool:
        for i, k in enumerate(pool.imap_unordered(_build, keys), 1):
            if i % 50 == 0 or i == len(keys):
                print(f"{i:4d}/{len(keys)} built/loaded", flush=True)
    print(f"caches here: {len(list(WF_DIR.glob('wf_*.npz')))} waveform pairs, "
          f"{len(list(LOSS_DIR.glob('loss_*.npz')))} drives", flush=True)


if __name__ == "__main__":
    main()
