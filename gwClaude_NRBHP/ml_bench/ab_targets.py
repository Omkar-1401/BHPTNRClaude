"""Paradigm B targets: model-free alpha(t), beta(t) per case by the equal-phase construction (as exploration/p0_measure.py,
spin_dep/spin_measure_alpha_beta.py): phases of the ppBHPT and NR (2,2) modes zeroed at each |h| peak, ppBHPT time t paired
with the NR time tau(t) of EQUAL phase;  beta = dtau/dt,  alpha = |h_NR(tau(t))| / |h_pp(t)|.
By construction these reproduce NR exactly through the repo's map (up to interpolation), in one gauge (phase offset fixed
at the peak) shared by every case -- which is what a regression target needs.  No fitting.

Grid: the hand models' clock s = t - t_peak,pp in m1 (S_GRID).  Where the paired tau leaves the NR window (early inspiral at
large s, late ringdown) values are held at the edge (those samples are outside the scored overlap anyway).
Non-monotone phase (retrograde ringdown beating) is handled by a running maximum of the phase magnitude.

    python ab_targets.py selftest | extract --nproc 64      -> results/ab_targets.npz (train pool only)
"""
from __future__ import annotations
import argparse, time
from multiprocessing import get_context

import numpy as np
from scipy.interpolate import CubicSpline

import common as C
from direct_targets import _tpeak

OUT = C.HERE / "results"
S_GRID = np.r_[np.arange(-7600.0, -300.0, 2.0), np.arange(-300.0, 120.0 + 1e-9, 0.25)]      # m1


def _phase(t, h):
    tp = _tpeak(t, h); s = t - tp
    ph = np.unwrap(np.angle(h)); ph = ph - np.interp(0.0, s, ph)
    sg = -np.sign(ph[-1] - ph[0]) if ph[-1] != ph[0] else 1.0          # make the phase increase with time
    ph = -sg * ph if sg > 0 else ph
    if ph[-1] < ph[0]:
        ph = -ph
    return s, ph


def _mono(x):
    return np.maximum.accumulate(x)


def targets(k, with_tau=False):
    """ln alpha, ln beta on S_GRID (m1); with_tau: also the clock tau(s) - s (NR time paired with s, edge-extended
    LINEARLY so beta = dtau/ds stays positive outside the NR window)."""
    c = C.case(k)
    tp_, hp = c["t_bhpt"], c["h_bhpt"]; tn, hn = c["t_nr"], c["h_nr"]
    sp, php = _phase(tp_, hp); sn, phn = _phase(tn, hn)
    php, phn = _mono(php), _mono(phn)
    phs = np.interp(S_GRID, sp, php)                        # pp phase on the target grid
    # invert NR phase: tau(phase), strictly increasing after a tiny ramp to break ties
    phn_s = phn + 1e-12 * np.arange(len(phn))
    tau = np.interp(phs, phn_s, sn)                         # held at the window edges outside the NR phase range
    beta = np.gradient(tau, S_GRID)
    alpha = np.interp(tau, sn, np.abs(hn)) / np.interp(S_GRID, sp, np.abs(hp))
    inside = (phs > phn_s[0]) & (phs < phn_s[-1])
    lb = np.log(np.clip(beta, 1e-3, None)); la = np.log(np.clip(alpha, 1e-6, None))
    i0, i1 = np.argmax(inside), len(inside) - 1 - np.argmax(inside[::-1])
    lb[:i0], la[:i0] = lb[i0], la[i0]; lb[i1 + 1:], la[i1 + 1:] = lb[i1], la[i1]
    if not with_tau:
        return la, lb
    tt = tau.copy(); b0, b1 = np.exp(lb[i0]), np.exp(lb[i1])
    tt[:i0] = tau[i0] + b0 * (S_GRID[:i0] - S_GRID[i0]); tt[i1 + 1:] = tau[i1] + b1 * (S_GRID[i1 + 1:] - S_GRID[i1])
    return la, lb, tt - S_GRID


def ab_on(k, c, la, lb, key):
    pre = "fit_" if key == "fit" else ""
    t = c[pre + "t_bhpt"]; tpk = _tpeak(c["t_bhpt"], c["h_bhpt"]); s = t - tpk
    return np.exp(CubicSpline(S_GRID, la)(np.clip(s, S_GRID[0], S_GRID[-1]))), \
        np.exp(CubicSpline(S_GRID, lb)(np.clip(s, S_GRID[0], S_GRID[-1])))


def score(k, la, lb):
    c = C.case(k)
    return C.score_ab(lambda key: ab_on(k, c, la, lb, key), c)


def ab_on_tau(k, c, la, taudev, key):
    """alpha from ln alpha, beta = dtau/ds from the regressed clock (target 'abtau')"""
    pre = "fit_" if key == "fit" else ""
    t = c[pre + "t_bhpt"]; tpk = _tpeak(c["t_bhpt"], c["h_bhpt"]); s = np.clip(t - tpk, S_GRID[0], S_GRID[-1])
    # beta floor 1e-3 as for the ln beta target (retrograde ringdown phase plateaus make tau flat; spline overshoot < 0)
    return np.exp(CubicSpline(S_GRID, la)(s)), np.clip(1.0 + CubicSpline(S_GRID, taudev).derivative()(s), 1e-3, None)


def score_tau(k, la, taudev):
    c = C.case(k)
    return C.score_ab(lambda key: ab_on_tau(k, c, la, taudev, key), c)


def _one(k):
    try:
        a, b, tt = targets(k, with_tau=True); return k, a, b, tt
    except Exception as e:                                   # noqa: BLE001
        print("fail", k, e, flush=True); return k, None, None, None


def extract(nproc):
    keys = C.train_pool(); t0 = time.time()
    with get_context("fork").Pool(nproc) as p:
        res = [r for r in p.map(_one, keys, chunksize=4) if r[1] is not None]
    np.savez_compressed(OUT / "ab_targets.npz", keys=np.array([r[0] for r in res]),
                        lna=np.array([r[1] for r in res]), lnb=np.array([r[2] for r in res]),
                        tau=np.array([r[3] for r in res]), s=S_GRID)
    print(f"ab_targets: {len(res)} cases, {time.time() - t0:.0f} s", flush=True)


def selftest():
    for k in [C.T96[0], C.T96[7], C.T96[50], C.T96[88], C.T96[-1]]:
        a, b, tt = targets(k, with_tau=True)
        print(f"{k}: own-target E  beta {score(k, a, b):.3e}   tau {score_tau(k, a, tt):.3e}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["extract", "selftest"])
    ap.add_argument("--nproc", type=int, default=64); a = ap.parse_args()
    extract(a.nproc) if a.cmd == "extract" else selftest()
