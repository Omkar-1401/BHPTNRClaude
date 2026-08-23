"""BHPTNRPeaks — peak-ratio BHPT->NR calibrated (2,2) waveform.

Usage
-----
    from BHPTNRPeaks import generate_peaks_calibrated
    t, h = generate_peaks_calibrated(q_input=5.0)

    t : 1-D array, time in units of M (NR-aligned, uniform dt=0.1, merger at t=0)
    h : 1-D complex array, calibrated (2,2) mode waveform

This is the companion generator to `BHPTNRClaude.py`, but the amplitude/time
rescalings alpha(t), beta(t) come from the paper's peak-ratio measurement
(Islam & Khanna arXiv:2307.03155 Sec. II.A.3) regressed across mass ratio, NOT
from an L2 waveform fit.  See `fit_scaling_peaks_stage2.py` for the model and
`scaling_peaks.md` for the writeup and honest performance envelope.

SCOPE / COVERAGE.  The peaks method is an inspiral method: this generator
reconstructs the inspiral through merger (~98% of the (2,2) energy).  The
post-merger ringdown (~2% of energy, the last ~100 M) is NOT modelled here and is
truncated at merger.  Calibrated on q in [2.25, 8]; usable down to q~2 (q=2 is a
one-step extrapolation, mathcalE ~ 0.55% over the covered window).

The overall phase is a free convention: h is returned with phi0 = 0 relative to the
BHPT merger phase (no NR is used at generation time).

The first call for a given q loads the BHPT surrogate (~1 min); subsequent calls with
the same q return from the in-process cache.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "BHPTNRSurrogate" / "surrogates"))

import fit_scaling_peaks_stage2 as _s2

_MODE = (2, 2)
_MODEL: dict | None = None


def _get_model() -> dict:
    global _MODEL
    if _MODEL is None:
        _MODEL = _s2.load_coeffs()
    return _MODEL


@lru_cache(maxsize=16)
def _load_bhpt(q: float) -> tuple[np.ndarray, np.ndarray]:
    import BHPTNRSur1dq1e4 as bhptsur
    t, h_dict = bhptsur.generate_surrogate(
        q=q, calibrated=False, modes=[_MODE], neg_modes=False
    )
    return t, h_dict[_MODE]


# Earliest NR-validated time (M). The BHPT surrogate provides a much longer inspiral,
# but below the training x-range alpha,beta are frozen at the clip boundary and the
# constant-beta error accumulates phase over the long early inspiral, so the default
# output is capped at the calibration window. Pass t_start=None to emit the full length.
_T_START_VALIDATED = _s2.creative.NR_T_START      # -5000.1


def _build(q_input: float, t_start: float | None = _T_START_VALIDATED):
    """Reconstruct on the BHPT sample grid. Returns (tau_abs, h_scaled, use)."""
    model = _get_model()
    t_bhpt, h_bhpt = _load_bhpt(float(q_input))
    # NR-aligned so merger sits at t = 0 (generator convention; no NR needed)
    tau_abs, h_scaled, mono = _s2.reconstruct(
        model, float(q_input), t_bhpt, h_bhpt, t_nr_merger=0.0, phi0=0.0
    )
    use = mono & (tau_abs <= 0.0 + 1e-9)     # inspiral through merger; drop post-merger
    if t_start is not None:
        use &= tau_abs >= t_start            # cap at the NR-validated window
    if np.count_nonzero(use) < 100:
        raise RuntimeError(
            f"Peak-ratio time map has too little coverage for q={q_input}; "
            "q may be far outside the supported range."
        )
    if np.any(np.diff(tau_abs[use]) <= 0.0):
        raise RuntimeError(f"Time map is non-monotone for q={q_input}.")
    return tau_abs, h_scaled, use


def generate_peaks_calibrated(
    q_input: float,
    dt: float = 0.1,
    t_start: float | None = _T_START_VALIDATED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the peak-ratio calibrated BHPT (2,2) waveform (inspiral through merger).

    Parameters
    ----------
    q_input : float
        Mass ratio q = m1/m2 >= 1. Calibration range 2.25 <= q <= 8; q down to ~2 is
        a supported one-step extrapolation. Values further out are extrapolated and
        degrade quickly.
    dt : float
        Output time step in units of M. Default 0.1, matching NRHybSur3dq8.
    t_start : float or None
        Earliest output time in units of M. Defaults to the NR-validated window start
        (~-5000 M). Pass None to emit the full BHPT inspiral, but note that the early
        inspiral below the training frequency range is unvalidated extrapolation
        (frozen alpha,beta) and accumulates phase error.

    Returns
    -------
    t : np.ndarray
        Uniform time array in units of M, ending at merger (t = 0).
    h : np.ndarray (complex)
        Calibrated (2,2) mode waveform on the t grid. Overall phase convention
        phi0 = 0 relative to the BHPT merger phase.
    """
    tau_abs, h_scaled, use = _build(q_input, t_start=t_start)
    t_out = np.arange(tau_abs[use][0], tau_abs[use][-1], dt)
    h_out = _s2.creative.interp_complex(tau_abs[use], h_scaled[use], t_out)
    return t_out, h_out


def get_alpha_beta(
    q_input: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the peak-ratio amplitude and time-stretch scalings at the model peaks.

    Parameters
    ----------
    q_input : float
        Mass ratio q = m1/m2 >= 1.

    Returns
    -------
    trel_bhpt : np.ndarray
        Merger-relative BHPT peak times (M), the nodes of the time map (<= 0).
    alpha : np.ndarray
        Amplitude scaling alpha at each peak (|h|_NR / |h|_ppBHPT).
    beta : np.ndarray
        Time-stretch beta_abs at each peak (t_NR / t_ppBHPT, merger-relative).
    """
    model = _get_model()
    t_bhpt, h_bhpt = _load_bhpt(float(q_input))
    nu = q_input / (1.0 + q_input) ** 2
    tm_b, tpk_b, om_pk = _s2.phase_peaks_full(t_bhpt, h_bhpt)
    trel_pk = tpk_b - tm_b
    x_pk = (om_pk / 2.0) ** (2.0 / 3.0)
    a_pk, b_pk = _s2.alpha_beta_at(model, nu, x_pk)
    o = np.argsort(trel_pk)
    return trel_pk[o], a_pk[o], b_pk[o]
