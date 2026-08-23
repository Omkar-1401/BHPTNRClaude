"""
BHPTNRClaude — creative time-dependent BHPT-NR calibrated waveform.

Usage
-----
    from BHPTNRClaude import generate_claude_calibrated
    t, h = generate_claude_calibrated(q_input=5.0)

    t  : 1-D array, time in units of M (NR-aligned, uniform dt=0.1)
    h  : 1-D complex array, calibrated (2,2) mode waveform

The calibration uses the 10-parameter logistic-switch model fit over
40 q values in [3, 8], with parameters represented as cubic polynomials
in 1/q. Coefficients are read from scaling_PN_opt_creative_q_dep.md in
the same directory as this file.

BHPT waveform loading is slow (~1 min on first call for a given q).
Subsequent calls with the same q return immediately from the in-process
cache.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "BHPTNRSurrogate" / "surrogates"))

import fit_scaling_PN_opt_creative as _creative
import fit_scaling_PN_opt_creative_q_dep as _qdep

_MD_PATH = ROOT / "scaling_PN_opt_creative_q_dep.md"
_MODE = (2, 2)

# ---------------------------------------------------------------------------
# Coefficient loading (once at first call)
# ---------------------------------------------------------------------------

_COEFFS: dict | None = None


def _get_coeffs() -> dict:
    global _COEFFS
    if _COEFFS is not None:
        return _COEFFS

    text = _MD_PATH.read_text()
    heading = "## Selected Master Coefficients"
    start = text.index(heading)
    lines = text[start:].splitlines()

    table_lines = []
    in_table = False
    for line in lines:
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            table_lines.append(s)
            in_table = True
        elif in_table:
            break

    rows = []
    for line in table_lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(c.replace(":", "").replace("-", "") == "" for c in cells):
            continue
        rows.append(cells)

    if not rows:
        raise RuntimeError(f"Could not parse coefficient table from {_MD_PATH}")

    coeffs: dict[str, np.ndarray] = {}
    for row in rows[1:]:
        coeffs[row[0]] = np.array([float(v) for v in row[1:]], dtype=float)

    missing = [n for n in _qdep.PARAM_NAMES if n not in coeffs]
    if missing:
        raise RuntimeError(f"Missing parameters in markdown table: {missing}")

    _COEFFS = {n: coeffs[n] for n in _qdep.PARAM_NAMES}
    return _COEFFS


# ---------------------------------------------------------------------------
# BHPT waveform loading (cached per q)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _load_bhpt(q: float) -> tuple[np.ndarray, np.ndarray]:
    import BHPTNRSur1dq1e4 as bhptsur
    t, h_dict = bhptsur.generate_surrogate(
        q=q, calibrated=False, modes=[_MODE], neg_modes=False
    )
    return t, h_dict[_MODE]


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

def _build_model(q_input: float) -> tuple:
    """Return (tau, h_scaled, alpha, beta, use) on the valid NR window."""
    coeffs = _get_coeffs()
    params = _qdep.constrained_master_params(q_input, coeffs)

    t_bhpt, h_bhpt = _load_bhpt(float(q_input))

    meta = {
        "nu": q_input / (1.0 + q_input) ** 2,
        "t_bhpt_merger": float(t_bhpt[np.argmax(np.abs(h_bhpt))]),
        "t_nr_merger": 0.0,
    }
    losses = _creative.pn_loss_coordinates(t_bhpt, h_bhpt, meta)

    tau, h_scaled, alpha, beta, _masks = _creative.model_arrays(
        params, t_bhpt, h_bhpt, losses
    )

    use = (tau >= _creative.NR_T_START) & (tau <= _creative.NR_T_END)
    if np.count_nonzero(use) < 100:
        raise RuntimeError(
            f"Calibrated time map has too little coverage for q={q_input}. "
            "q may be outside the supported range [3, 8]."
        )
    if np.any(np.diff(tau[use]) <= 0.0):
        raise RuntimeError(
            f"Calibrated time map is non-monotone for q={q_input}."
        )

    return tau, h_scaled, alpha, beta, use


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate_claude_calibrated(
    q_input: float,
    dt: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the creative time-dependent calibrated BHPT (2,2) waveform.

    Parameters
    ----------
    q_input : float
        Mass ratio q = m1/m2 >= 1. The calibration polynomial covers
        3 <= q <= 8; values outside this range are extrapolated.
    dt : float
        Output time step in units of M. Default 0.1, matching NRHybSur3dq8.

    Returns
    -------
    t : np.ndarray
        Uniform time array in units of M, NR-time-aligned.
    h : np.ndarray (complex)
        Calibrated (2,2) mode waveform on the t grid.

    Notes
    -----
    The time and phase alignment (t0_nr, phi0) are read from the
    polynomial fit. They are accurate to the polynomial interpolation
    error but are not re-optimised against NR for the specific q_input.
    For the calibration grid points this error is at the ~1e-4 level.

    The first call for a given q loads the BHPT surrogate (~1 min).
    Subsequent calls with the same q return from the in-process cache.
    """
    tau, h_scaled, _alpha, _beta, use = _build_model(q_input)
    t_out = np.arange(tau[use][0], tau[use][-1], dt)
    h_out = _creative.interp_complex(tau[use], h_scaled[use], t_out)
    return t_out, h_out


def get_alpha_beta(
    q_input: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the time-dependent amplitude and time-stretch scaling functions.

    Parameters
    ----------
    q_input : float
        Mass ratio q = m1/m2 >= 1. Valid calibration range: 3 <= q <= 8.

    Returns
    -------
    t : np.ndarray
        Time array in units of M (NR-aligned, non-uniform — one point per
        BHPT sample mapped through the beta time integral).
    alpha : np.ndarray
        Amplitude scaling alpha(t): the factor by which the BHPT waveform
        amplitude is multiplied at each instant.
    beta : np.ndarray
        Time-stretch beta(t): the local ratio of NR to BHPT time at each
        instant. Integrating beta maps BHPT coordinate time to NR time.

    Notes
    -----
    The time axis t is tau[use] — the BHPT time grid mapped to NR-aligned
    time via the beta integral. It is not uniform but is monotone and
    suitable for plotting. alpha and beta are evaluated at the same points.
    """
    tau, _h_scaled, alpha, beta, use = _build_model(q_input)
    return tau[use], alpha[use], beta[use]
