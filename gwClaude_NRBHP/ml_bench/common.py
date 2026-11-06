"""ml_bench shared pieces: data access (cache routing + sealed guard via training_size/grids), the training pool,
and the scorer.

Scoring = the exploration chain's protocol (exploration/p6_m2_global.err): S._err_at_t0 (imported, never
re-implemented), t0 from a 126-point grid on [-300, 200] + bounded refine on the fit stride, then the full-resolution
error seeded with that t0.  Two entry points:
    score_ab(alpha, beta, case)       alpha, beta on the ppBHPT grid (m1); tau = t0 + int beta dt, anchored at T_ANCHOR
    score_wave(tm, hm, case)          any model waveform hm on its own NR-time grid tm (alpha = 1, ts = tm), e.g. a
                                      direct amplitude/phase model; same E, same phi0, same t0 search
Both take {"fit": (...), "full": (...)} arrays; see ab_both / wave_both.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent; REPO = HERE.parent
for p in (REPO / "training_size", REPO / "exploration", REPO / "spin_dep", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import grids as G                      # noqa: E402  FIRST: cache routing + sealed guard on S / FS
S, FS = G.S, G.FS
import fit_scaling_PN_opt_creative as creative   # noqa: E402

Q_LO, Q_HI, C_LO, C_HI = G.Q_LO, G.Q_HI, G.C_LO, G.C_HI
V300 = list(G.V300)
T96 = list(G.T96)
_V300 = set(V300)


def nu_of(q):
    return q / (1.0 + q) ** 2


def cached_keys():
    """Every (q, chi1) with a cached waveform pair in range [3, 8] x [-0.6, 0.8] (spin_dep + training_size caches)."""
    out = set()
    for d in (G.WF_SHARED, G.WF_DIR):
        for f in d.glob("wf_q*_chi*.npz"):
            s = f.stem[4:]; q = float(s.split("_chi")[0]); c = float(s.split("_chi")[1])
            if Q_LO - 1e-9 <= q <= Q_HI + 1e-9 and C_LO - 1e-9 <= c <= C_HI + 1e-9:
                out.add(G.key(q, c))
    return sorted(out)


def train_pool():
    """Training candidates: cached keys minus V300 (score-only) minus anything the sealed guard refuses."""
    keys = []
    for k in cached_keys():
        if k in _V300:
            continue
        try:
            G.TG.check(*k)
        except G.TG.SealedTestPoint:
            continue
        keys.append(k)
    return keys


def case(k):
    """Waveform pair + derived arrays (FS.build_case: S.build_case + energy/flux losses), via the guarded routing."""
    G._route(*k)
    return FS.build_case(*k)


def _t0_err(f, seed):
    if seed is None:
        grid = np.linspace(-300, 200, 126); seed = grid[int(np.argmin([f(g) for g in grid]))]
    r = minimize_scalar(f, bounds=(seed - 6, seed + 6), method="bounded", options={"xatol": 1e-3})
    return float(r.fun), float(r.x)


def score_ab_key(alpha, beta, c, key, seed=None):
    pre = "fit_" if key == "fit" else ""
    t, h, tn, hn = c[pre + "t_bhpt"], c[pre + "h_bhpt"], c[pre + "t_nr"], c[pre + "h_nr"]
    if alpha is None or not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(beta)):
        return 1.0, 0.0
    bc = creative.cumulative_trapezoid(beta, t); ts = bc - np.interp(S.T_ANCHOR, t, bc)
    if np.min(np.diff(ts)) <= 0:
        return 1.0, 0.0
    return _t0_err(lambda x0: S._err_at_t0(x0, alpha, ts, t, h, tn, hn), seed)


def score_ab(ab_fun, c):
    """ab_fun(key) -> (alpha, beta) on case[pre + 't_bhpt'].  Returns full-resolution E (fit-stride t0 seed)."""
    e, t0 = score_ab_key(*ab_fun("fit"), c, "fit")
    return score_ab_key(*ab_fun("full"), c, "full", seed=t0)[0]


def score_wave_key(tm, hm, c, key, seed=None):
    pre = "fit_" if key == "fit" else ""
    tn, hn = c[pre + "t_nr"], c[pre + "h_nr"]
    if not np.all(np.isfinite(hm)):
        return 1.0, 0.0
    one = np.ones_like(tm)
    return _t0_err(lambda x0: S._err_at_t0(x0, one, tm, tm, hm, tn, hn), seed)


def score_wave(wave_fun, c):
    """wave_fun(key) -> (tm, hm): model waveform on its own NR-time grid.  Full-resolution E (fit-stride t0 seed)."""
    e, t0 = score_wave_key(*wave_fun("fit"), c, "fit")
    return score_wave_key(*wave_fun("full"), c, "full", seed=t0)[0]


def stats(v):
    v = np.asarray(v, float)
    return dict(median=float(np.median(v)), mean=float(np.mean(v)), max=float(np.max(v)), n=int(len(v)))


def fmt(s):
    return f"{s['median']:.3e} / {s['mean']:.3e} / {s['max']:.2e} (n={s['n']})"
