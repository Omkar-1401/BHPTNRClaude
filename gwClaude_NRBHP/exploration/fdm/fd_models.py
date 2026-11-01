"""Constant-coefficient comparison models on the repo scorer (import fd_route first).

zhong   Zhong, Han & Sun 2025 (arXiv:2503.04534), the (alpha', beta') extracted from their non-spinning CfC model,
        Tables II (l = 2) and III, Eq. alpha_beta form:  alpha'(q), beta'(q) = 1 + A/q + B/q^2 + C/q^3 + D/q^4.
        Their map (Sec. "Scaling parameters extraction"): h = alpha' A_pp(t) exp(-i phi_pp(t) / beta'), i.e. the PHASE
        is scaled on the ppBHPT's own time grid (raw BHPTNRSur output, m1 units, used as the NR time axis).
        'zhong' = that literal map; 'zhong_t' = beta' used as a TIME scaling instead (t_NR = beta' t_pp, the
        BHPTNRSur convention, common_utils/nr_calibration.beta_scaling_time), a diagnostic only.
        Spin-blind: the same alpha'(q), beta'(q) are applied to the spinning ppBHPT input (BHPTNRSur2dq1e3, chi1).
        8 coefficients for (2,2).
rink    the published calibrated BHPTNRSur2dq1e3 (Rink et al. 2024): calibrated=True output, identity map, t0 / phi0
        optimised -- exactly exploration/t100_score.py's 'rink' branch.
Scoring = S._err_at_t0 (full-window mathcalE vs NRHybSur3dq8, (2,2)), t0 by grid search on the fit stride, then bounded
refinement at full resolution, phi0 analytic.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar

import fd_route                                # noqa: F401  (cache routing; must precede model imports)
from fd_route import S, FS

Z_A = [-1.488, 2.341, -4.911, 6.302]           # alpha'_2: A, B, C, D (Table II, l = 2)
Z_B = [-0.944, 1.618, -2.277, 1.457]           # beta':    A, B, C, D (Table III)
poly = lambda c, q: 1.0 + sum(c[n] * q ** -(n + 1) for n in range(4))


def fit_t0(ts, h, alpha, c, grid=(-300.0, 200.0, 251)):
    si = np.unique(np.r_[np.arange(0, len(ts), S.SRC_STRIDE), len(ts) - 1])
    tf, hf, af = ts[si], h[si], alpha[si]
    f = lambda x: S._err_at_t0(x, af, tf, tf, hf, c["fit_t_nr"], c["fit_h_nr"])
    g = np.linspace(*grid); cen = float(g[int(np.argmin([f(v) for v in g]))])
    step = (grid[1] - grid[0]) / (grid[2] - 1)
    x = minimize_scalar(f, bounds=(cen - 3 * step, cen + 3 * step), method="bounded", options={"xatol": 1e-4}).x
    f2 = lambda x0: S._err_at_t0(x0, alpha, ts, ts, h, c["t_nr"], c["h_nr"])
    r = minimize_scalar(f2, bounds=(x - 3, x + 3), method="bounded", options={"xatol": 1e-4})
    return float(r.fun), float(r.x)


def wave(model, q, chi):
    """(ts, h, alpha, case, grid) for the constant-coefficient models; NR time = t0 + ts."""
    c = FS.build_case(q, chi)
    if model == "rink":
        import BHPTNRSur2dq1e3 as ppsur
        t, hd = ppsur.generate_surrogate(q=q, spin1=chi, calibrated=True, modes=[(2, 2)], neg_modes=False)
        t = np.asarray(t, float); h = np.asarray(hd[(2, 2)])
        return t - S.T_ANCHOR, h, np.ones_like(t), c, (-300.0, 200.0, 251)
    t, h = c["t_bhpt"], c["h_bhpt"]
    a, b = poly(Z_A, q), poly(Z_B, q)
    if model == "zhong":
        hz = np.abs(h) * np.exp(1j * np.unwrap(np.angle(h)) / b)
        return t - S.T_ANCHOR, hz, np.full_like(t, a), c, (-1500.0, 1500.0, 1201)
    if model == "zhong_t":
        return b * (t - S.T_ANCHOR), h, np.full_like(t, a), c, (-300.0, 200.0, 251)
    raise ValueError(model)


def score(model, q, chi):
    ts, h, al, c, grid = wave(model, q, chi)
    return fit_t0(ts, h, al, c, grid)[0]


def calibrated(model, q, chi):
    """NR and the aligned model on the common NR window (same window and alignment as the E score)."""
    ts, h, al, c, grid = wave(model, q, chi)
    e, t0 = fit_t0(ts, h, al, c, grid)
    tn, hn = c["t_nr"], c["h_nr"]; tau = t0 + ts
    m = (tn >= max(tn[0], tau[0])) & (tn <= min(tn[-1], tau[-1]))
    tc, r = tn[m], hn[m]
    g = np.interp(tc, tau, al) * (np.interp(tc, tau, h.real) + 1j * np.interp(tc, tau, h.imag))
    Z = np.sum(r * np.conj(g)); n1, n2 = np.sum(np.abs(r) ** 2), np.sum(np.abs(g) ** 2)
    e2 = float((n1 + n2 - 2 * abs(Z)) / (2 * n1))
    assert abs(e2 / e - 1) < 1e-6, (model, q, chi, e, e2)
    return dict(tc=tc, hnc=r, hcal=g * Z / abs(Z), E=e)
