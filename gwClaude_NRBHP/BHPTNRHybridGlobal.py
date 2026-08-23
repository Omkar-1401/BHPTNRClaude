"""BHPTNRHybridGlobal — globally-jointly-fit wf-nu-hybrid BHPT->NR (2,2) waveform.

Usage
-----
    from BHPTNRHybridGlobal import generate_hybrid_global_calibrated
    t, h = generate_hybrid_global_calibrated(q_input=5.0)

    t : 1-D array, time in units of M (merger at t = 0, uniform dt = 0.1)
    h : 1-D complex array, calibrated (2,2) mode waveform

This is the `wf_nu_hybrid_global` model: the SAME 11-parameter wf-nu-hybrid
architecture as `wf_nu_hybrid_q_dep` (logistic switch between an inspiral segment
with a linear alpha drift and a remnant segment), but the master polynomial
coefficients are obtained by a GLOBAL JOINT FIT of the coefficients directly against
all q in [3, 8] waveforms at once (global_joint_fit.py --sample nu), instead of
per-q fit -> regress. Training points are sampled evenly in nu (the regression variable),
which keeps the fit constrained near the q=3 boundary that governs q<3 extrapolation. This
yields a more uniform in-range fit (in-range mathcalE max 9.6e-4 -> 5.2e-4) and smoother
alpha/beta shapes across q, while clearing the 1e-2 gate across all of q in [2, 2.75]
(q=2 extrapolation mathcalE ~ 4.0e-3, q=2.25 ~ 9.4e-3). Coefficients:
wf_nu_hybrid_global_results/coeffs.json.

Model equations (nu = q/(1+q)^2; all params are cubic polynomials in nu, PP-anchored):
    S      = 1 / (1 + exp(-(p_loss - p0)/w))
    dp_hat = (p_loss - p0)/(p_loss[0] - p0)                 # in [0,1] during inspiral
    alpha  = alpha_i + (1-S)*alpha_L*dp_hat + S*(alpha_E*dE + alpha_J*dJ)
    beta   = (1-S)*(beta_i + beta_L*dp) + S*beta_r
    tau(t) = t0 + integral beta dt     (t0 chosen so the merger maps to t=0)
    h(tau) = alpha * exp(i*phi0) * h_BHPT

Loss coordinates (Ehat, Jhat, p_loss) come from the ppBHPT (2,2) GW flux (no surfinBH).
The overall alignment (t0, phi0) is a free convention here: the merger is placed at
t = 0 and phi0 = 0 (no NR is used at generation). Calibrated on q in [3, 8]; q = 2 is a
supported extrapolation. Below q ~ 2 the model degrades quickly.

The first call for a given q loads the BHPT surrogate (~1 min); subsequent calls with
the same q return from the in-process cache.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "BHPTNRSurrogate" / "surrogates"))
sys.path.insert(0, str(ROOT.parent / "BHPTutils"))

import fit_scaling_PN_opt_creative as _creative
import fit_scaling_wf_nu_q_dep as _wfnu
import fit_scaling_wf_nu_hybrid_q_dep as _H

_MODE = (2, 2)
_COEFF_JSON = ROOT / "wf_nu_hybrid_global_results" / "coeffs.json"
_T_START_VALIDATED = _creative.NR_T_START          # ~-5000 M
_COEFFS: dict | None = None


def _get_coeffs() -> dict:
    global _COEFFS
    if _COEFFS is None:
        d = json.loads(_COEFF_JSON.read_text())
        _COEFFS = {n: np.array(d[n], dtype=float) for n in _H.PARAM_NAMES}
    return _COEFFS


@lru_cache(maxsize=16)
def _load_bhpt(q: float) -> tuple[np.ndarray, np.ndarray]:
    import BHPTNRSur1dq1e4 as bhptsur
    t, h_dict = bhptsur.generate_surrogate(
        q=q, calibrated=False, modes=[_MODE], neg_modes=False
    )
    return t, h_dict[_MODE]


def _params(q: float) -> np.ndarray:
    """9 physical params from the global coeffs, clipped as in training; t0/phi0 unused."""
    coeffs = _get_coeffs()
    p = np.array([_wfnu.eval_poly(q, coeffs[n]) for n in _H.PARAM_NAMES], dtype=float)
    p[0] = np.clip(p[0], -3.0, 0.05)                    # p0
    p[1] = np.clip(p[1], _creative.W_MIN, 0.45)         # w
    p[2] = np.clip(p[2], 0.05, 2.5)                     # alpha_i
    p[6] = np.clip(p[6], 0.2, 1.6)                      # beta_i
    p[7] = np.clip(p[7], 0.2, 1.6)                      # beta_r
    p[10] = np.clip(p[10], -0.05, 0.05)                 # beta_L
    return p


def _alpha_beta(params, t_bhpt, h_bhpt, losses):
    """Build alpha(t), beta(t) from the hybrid equations (no NR)."""
    p0, w, ai, aL, aE, aJ, bi, br, _t0, _ph, bL = params
    pl, eh, jh = losses["p_loss"], losses["e_hat"], losses["j_hat"]
    S = _creative.sigmoid((pl - p0) / w)
    dp = pl - p0
    e0 = float(np.interp(p0, pl, eh)); j0 = float(np.interp(p0, pl, jh))
    dE = eh - e0; dJ = jh - j0
    denom = pl[0] - p0
    if abs(denom) < 1e-8:
        denom = -1.0
    dp_hat = dp / denom
    alpha = ai + (1.0 - S) * aL * dp_hat + S * (aE * dE + aJ * dJ)
    beta = (1.0 - S) * (bi + bL * dp) + S * br
    return alpha, beta


def _build(q_input: float, t_start: float | None = _T_START_VALIDATED):
    """Reconstruct on the BHPT sample grid. Returns (tau, h_scaled, use)."""
    q = float(q_input)
    t_bhpt, h_bhpt = _load_bhpt(q)
    meta = {"nu": q / (1.0 + q) ** 2,
            "t_bhpt_merger": float(t_bhpt[np.argmax(np.abs(h_bhpt))]),
            "t_nr_merger": 0.0}
    losses = _wfnu.wf_loss_coordinates(t_bhpt, h_bhpt, meta)

    params = _params(q)
    alpha, beta = _alpha_beta(params, t_bhpt, h_bhpt, losses)

    beta_cum = _creative.cumulative_trapezoid(beta, t_bhpt)
    m_b = int(np.argmax(np.abs(h_bhpt)))
    tau = beta_cum - beta_cum[m_b]              # merger (BHPT amp peak) -> t = 0
    h_scaled = alpha * h_bhpt                    # phi0 = 0 convention

    use = np.r_[True, np.diff(tau) > 0]          # monotone time map
    if t_start is not None:
        use &= tau >= t_start
    if np.count_nonzero(use) < 100:
        raise RuntimeError(f"Time map has too little coverage for q={q_input}.")
    if np.any(np.diff(tau[use]) <= 0.0):
        raise RuntimeError(f"Time map is non-monotone for q={q_input}.")
    return tau, h_scaled, use


def generate_hybrid_global_calibrated(
    q_input: float,
    dt: float = 0.1,
    t_start: float | None = _T_START_VALIDATED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the wf_nu_hybrid_global calibrated BHPT (2,2) waveform.

    Parameters
    ----------
    q_input : float
        Mass ratio q = m1/m2 >= 1. Calibration range 3 <= q <= 8; q in [2, 2.75] is a
        supported extrapolation (q=2 mathcalE ~ 4.0e-3, all under the 1e-2 gate). Values
        below ~2 degrade quickly.
    dt : float
        Output time step in units of M. Default 0.1, matching NRHybSur3dq8.
    t_start : float or None
        Earliest output time (M). Defaults to the NR-validated window (~-5000 M).
        Pass None to emit the full BHPT inspiral (early portion is unvalidated).

    Returns
    -------
    t : np.ndarray
        Uniform time array in units of M, merger at t = 0.
    h : np.ndarray (complex)
        Calibrated (2,2) waveform. Overall alignment (time, phase) is by convention
        (merger at t = 0, phi0 = 0); optimize t0/phi0 when comparing to a reference.
    """
    tau, h_scaled, use = _build(q_input, t_start=t_start)
    t_out = np.arange(tau[use][0], tau[use][-1], dt)
    h_out = _creative.interp_complex(tau[use], h_scaled[use], t_out)
    return t_out, h_out


def get_alpha_beta(q_input: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_NR, alpha(t), beta(t)) for the model at q_input (merger at t=0)."""
    q = float(q_input)
    t_bhpt, h_bhpt = _load_bhpt(q)
    meta = {"nu": q / (1.0 + q) ** 2,
            "t_bhpt_merger": float(t_bhpt[np.argmax(np.abs(h_bhpt))]),
            "t_nr_merger": 0.0}
    losses = _wfnu.wf_loss_coordinates(t_bhpt, h_bhpt, meta)
    params = _params(q)
    alpha, beta = _alpha_beta(params, t_bhpt, h_bhpt, losses)
    beta_cum = _creative.cumulative_trapezoid(beta, t_bhpt)
    m_b = int(np.argmax(np.abs(h_bhpt)))
    tau = beta_cum - beta_cum[m_b]
    use = np.r_[True, np.diff(tau) > 0] & (tau >= _T_START_VALIDATED)
    return tau[use], alpha[use], beta[use]
