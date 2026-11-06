"""Paradigm A targets (Zhong+ 2503.04534 style, no alpha-beta map, no fitting): amplitude and phase corrections from the
ppBHPT (2,2) mode to NR on a common peak-aligned grid in TOTAL-mass units.

    s = t - t_peak (each waveform on its own clock; ppBHPT t [m1] -> t * X1 [M], the unit conversion m1 = X1 M)
    dlnA(s) = ln|h_NR(s)| - ln|h_pp(s)|,    dphi(s) = phi_NR(s) - phi_pp(s)     (phi = unwrapped arg h, 0 at the peak)
Model waveform for any (q, chi1) given predicted dlnA, dphi on S_GRID:
    h_model(s) = h_pp(s) exp(dlnA(s) + i dphi(s))        on the ppBHPT samples inside the window, NR time = s
scored with common.score_wave (same E, phi0 analytic, t0 refit).  t_peak by a quadratic refinement of argmax |h|.

    python direct_targets.py extract --nproc 64     -> results/direct_targets.npz (train pool) + direct_targets_V300.npz
    python direct_targets.py selftest               -> own-target reconstruction E on a few cases (should be ~0)
"""
from __future__ import annotations
import argparse, sys, time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

import common as C

OUT = C.HERE / "results"; OUT.mkdir(exist_ok=True)
# M; ppBHPT ends at +94 M at q = 3, NR window is [-5000, +100].  1 M in the inspiral, 0.2 M from -200 M: the retrograde
# ringdown (pp |h| -> 5e-6) needs it (q = 3, chi1 = -0.6 own-target E: 1 M everywhere 5.8e-5, 0.25 M 3.7e-7, 0.1 M 3.1e-8)
S_GRID = np.r_[np.arange(-5020.0, -200.0, 1.0), np.arange(-200.0, 94.0 + 1e-9, 0.2)]


def _tpeak(t, h):
    a = np.abs(h); i = int(np.argmax(a)); y0, y1, y2 = a[i - 1], a[i], a[i + 1]
    d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return t[i] + d * (t[i + 1] - t[i])


def _aligned(t, h):
    tp = _tpeak(t, h); s = t - tp
    ph = np.unwrap(np.angle(h)); ph = ph - np.interp(0.0, s, ph)
    return s, np.abs(h), ph


def pp_aligned(k, c, key="full"):
    pre = "fit_" if key == "fit" else ""
    X1 = k[0] / (1 + k[0])
    tf, hf = c["t_bhpt"], c["h_bhpt"]
    tp = _tpeak(tf * X1, hf)                      # peak from full resolution, also for the fit stride
    t, h = c[pre + "t_bhpt"] * X1, c[pre + "h_bhpt"]
    s = t - tp
    phf = np.unwrap(np.angle(hf)); ph0 = np.interp(0.0, tf * X1 - tp, phf)
    ph = np.interp(s, tf * X1 - tp, phf) - ph0 if key == "fit" else phf - ph0
    return s, h, np.abs(h), ph


def targets(k):
    c = C.case(k)
    sp, ap, php = _aligned(c["t_bhpt"] * k[0] / (1 + k[0]), c["h_bhpt"])
    sn, an, phn = _aligned(c["t_nr"], c["h_nr"])
    dlnA = np.log(np.interp(S_GRID, sn, an)) - np.log(np.interp(S_GRID, sp, ap))
    dphi = np.interp(S_GRID, sn, phn) - np.interp(S_GRID, sp, php)
    return dlnA, dphi


def render(k, c, dlnA, dphi, key):
    """model waveform (NR time = s) from predicted curves on S_GRID"""
    s, h, _, _ = pp_aligned(k, c, key)
    m = (s >= S_GRID[0]) & (s <= S_GRID[-1])
    fa = CubicSpline(S_GRID, dlnA)(s[m]); fp = CubicSpline(S_GRID, dphi)(s[m])
    return s[m], h[m] * np.exp(fa + 1j * fp)


def score(k, dlnA, dphi):
    c = C.case(k)
    return C.score_wave(lambda key: render(k, c, dlnA, dphi, key), c)


def _one(k):
    try:
        a, p = targets(k); return k, a, p
    except Exception as e:                         # noqa: BLE001
        print("fail", k, e, flush=True); return k, None, None


def extract(nproc):
    for name, keys in (("direct_targets", C.train_pool()), ("direct_targets_V300", C.V300)):
        t0 = time.time()
        with get_context("fork").Pool(nproc) as p:
            res = [r for r in p.map(_one, keys, chunksize=4) if r[1] is not None]
        K = np.array([r[0] for r in res]); A = np.array([r[1] for r in res]); P = np.array([r[2] for r in res])
        np.savez_compressed(OUT / f"{name}.npz", keys=K, dlnA=A, dphi=P, s=S_GRID)
        print(f"{name}: {len(K)} cases, {time.time() - t0:.0f} s", flush=True)


def selftest():
    for k in [C.T96[0], C.T96[50], C.T96[-1]]:
        a, p = targets(k); t0 = time.time()
        print(f"{k}: own-target E {score(k, a, p):.3e}  ({time.time() - t0:.1f} s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["extract", "selftest"])
    ap.add_argument("--nproc", type=int, default=64); a = ap.parse_args()
    extract(a.nproc) if a.cmd == "extract" else selftest()
