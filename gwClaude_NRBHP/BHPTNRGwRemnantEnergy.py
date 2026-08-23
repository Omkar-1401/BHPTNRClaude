"""BHPTNRGwRemnantEnergy — energy-driven ppBHPT->NR (2,2) calibrated waveform.

Usage
-----
    from BHPTNRGwRemnantEnergy import generate_gwr_energy_calibrated
    t, h = generate_gwr_energy_calibrated(q_input=5.0)

    t : 1-D array, time in units of M (merger at t = 0, uniform dt = 0.1)
    h : 1-D complex array, calibrated (2,2) mode waveform

Model.  Switchless, no PN expressions, no logistic gate, no remnant fits.  The ONLY
time dependence in the scaling is the energy the binary has radiated, taken from
`gw_remnant` (arXiv:2301.07215 / 1802.04276) -- the quantity behind
`RemnantMassCalculator.plot_mass_energy()`:

    dE/dt = (1/16 pi) sum_lm |dh_lm/dt|^2      # modes (2,2) and (2,-2) of the ppBHPT
    E(t)  = cumtrapz(dE/dt, t)                 # calc.Eoft, units of M, E(t_start) = 0
    M(t)  = 1 - E(t)                           # calc.Moft, the Bondi mass

    nu = q/(1+q)^2
    alpha(t, nu) = alpha_PP(nu) * (1 + alpha_E(nu) * E(t))
    beta (t, nu) = beta_PP (nu) * (1 + beta_E (nu) * E(t))
    tau(t) = integral beta dt   (merger -> t = 0)
    h(tau) = alpha * h_BHPT     (phi0 = 0 convention, as in the other model modules)

alpha_PP, beta_PP are PP-anchored (c0 = 1 at nu = 0, so the uncalibrated ppBHPT
waveform is recovered exactly in the test-particle limit).  The couplings are NOT
anchored and do not need to be: E_rad -> 0 as nu -> 0 (measured E_tot ~ nu^2.31 over
q in [3,10]), so the correction term vanishes on its own.  Because E(t) is zero at the
start of the surrogate window, alpha_PP is literally the amplitude scaling there.

Each of the four functions is a **degree-3** polynomial in nu, fitted per-q on the 64
cached mass ratios in [3, 8] and then regressed across nu.  Degree 4 was tested and
rejected: it is marginally better in-range but destroys q < 3 (q=2.5 goes 1.3e-3 ->
6.7e-2).

Accuracy (mathcalE vs NRHybSur3dq8, calibrated=False ppBHPT input):
    in-range [3,8]   median 6.16e-4,  max 9.87e-4   (uniform: max/median = 1.6)
    q = 2.75         1.01e-3
    q = 2.5          1.32e-3          <- lowest q inside the BHPT surrogate's domain
    q = 2.25         4.14e-3
    q = 2.0          2.43e-2

CAVEATS (see scaling_gw_remnant_energy_mult.md).
  * q < 3 is a genuine extrapolation of the nu polynomials; only [3,8] was trained on.
  * `BHPTNRSur1dq1e4` is declared valid only for q >= 2.5 (X_min = log10(2.5)) and the
    bound is a printed warning, not an error -- so at q = 2.25 and 2.0 the BHPT *input*
    is itself an extrapolation of the surrogate's splines in log q.
  * The q=2 result is limited by a degeneracy, not by the model form: fitting q=2 on its
    own reaches 8.5e-4.  beta_PP and P = beta_PP*beta_E trade off, beta_PP needs ~0.5%
    accuracy at q=2, and P crosses zero at q ~ 4.1 which leaves its slope at the q=3
    boundary unconstrained.  A global joint fit is the indicated fix.

The first call for a given q loads the BHPT surrogate (~1 min); cached thereafter.
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

import fit_scaling_PN_opt_creative as _creative

from gw_remnant.remnant_calculators.remnant_mass_calculator import (
    RemnantMassCalculator,
)

_MODE = (2, 2)
_COEFF_JSON = ROOT / "gw_remnant_energy_results" / "coeffs_mult.json"
_T_START_VALIDATED = _creative.NR_T_START            # ~ -5000 M
_T_END = _creative.NR_T_END                          # 100 M
_PARAM_ORDER = ("alpha_PP", "alpha_E", "beta_PP", "beta_E")
_C: dict | None = None


def _cfg() -> dict:
    global _C
    if _C is None:
        d = json.loads(_COEFF_JSON.read_text())
        deg = str(d["selected_degree"])
        _C = {"deg": int(deg),
              "coeffs": {k: np.asarray(v, float) for k, v in d["coeffs"][deg].items()},
              "training_range": d["training_range"]}
    return _C


def _nu(q: float) -> float:
    return q / (1.0 + q) ** 2


def _eval_poly(q: float, c: np.ndarray) -> float:
    nu = _nu(q)
    return float(sum(c[k] * nu ** k for k in range(len(c))))


def master_params(q_input: float) -> dict[str, float]:
    """The four nu-functions evaluated at q_input (alpha_PP, alpha_E, beta_PP, beta_E)."""
    cfg = _cfg()
    p = {k: _eval_poly(float(q_input), cfg["coeffs"][k]) for k in _PARAM_ORDER}
    p["alpha_PP"] = float(np.clip(p["alpha_PP"], 0.05, 2.5))
    p["beta_PP"] = float(np.clip(p["beta_PP"], 0.2, 1.6))
    return p


@lru_cache(maxsize=16)
def _load_bhpt(q: float):
    import BHPTNRSur1dq1e4 as bhptsur
    t, h = bhptsur.generate_surrogate(q=q, calibrated=False, modes=[_MODE], neg_modes=False)
    return t, h[_MODE]


def radiated_energy(t_bhpt: np.ndarray, h_bhpt: np.ndarray, q: float) -> np.ndarray:
    """gw_remnant cumulative radiated energy E(t) in units of M; E(t_start) = 0."""
    calc = RemnantMassCalculator(
        time=t_bhpt, h_dict={_MODE: h_bhpt, (_MODE[0], -_MODE[1]): np.conj(h_bhpt)},
        q=float(q), E_initial=0.0, L_initial=0.0, M_initial=1.0, use_filter=False,
    )
    return np.asarray(calc.Eoft, dtype=float)


def _alpha_beta(q: float, e_oft: np.ndarray):
    p = master_params(q)
    alpha = p["alpha_PP"] * (1.0 + p["alpha_E"] * e_oft)
    beta = p["beta_PP"] * (1.0 + p["beta_E"] * e_oft)
    return alpha, beta


def _build(q_input: float, t_start=_T_START_VALIDATED):
    q = float(q_input)
    t_bhpt, h_bhpt = _load_bhpt(q)
    e_oft = radiated_energy(t_bhpt, h_bhpt, q)
    alpha, beta = _alpha_beta(q, e_oft)

    if np.any(beta <= 0):
        raise RuntimeError(f"Non-positive beta for q={q_input}.")

    bc = _creative.cumulative_trapezoid(beta, t_bhpt)
    m_b = int(np.argmax(np.abs(h_bhpt)))
    tau = bc - bc[m_b]                                   # merger -> t = 0
    h_scaled = alpha * h_bhpt                            # phi0 = 0 convention

    use = np.r_[True, np.diff(tau) > 0]
    if t_start is not None:
        use &= tau >= t_start
    use &= tau <= _T_END
    if np.count_nonzero(use) < 100 or np.any(np.diff(tau[use]) <= 0.0):
        raise RuntimeError(f"Time map invalid for q={q_input} (coverage/monotonicity).")
    return tau, h_scaled, use, alpha, beta, e_oft


def generate_gwr_energy_calibrated(q_input: float, dt: float = 0.1,
                                   t_start=_T_START_VALIDATED):
    """Generate the energy-calibrated BHPT (2,2) waveform (merger at t=0, dt=0.1).

    q_input : mass ratio >= 1.  Calibrated on [3,8]; q in [2.5, 3) is a supported
              extrapolation (q=2.5 mathcalE ~ 1.3e-3).  Below q=2.5 the BHPT surrogate
              is itself out of domain -- see the module docstring.
              t_start=None emits the full inspiral.
    """
    tau, h_scaled, use, *_ = _build(q_input, t_start=t_start)
    t_out = np.arange(tau[use][0], tau[use][-1], dt)
    return t_out, _creative.interp_complex(tau[use], h_scaled[use], t_out)


def get_alpha_beta(q_input: float):
    """Return (t_NR, alpha(t), beta(t)) for the model at q_input (merger at t=0)."""
    tau, _, use, alpha, beta, _ = _build(q_input)
    return tau[use], alpha[use], beta[use]


def get_energy(q_input: float):
    """Return (t_NR, E(t)) -- the gw_remnant radiated energy on the mapped time axis."""
    tau, _, use, _, _, e_oft = _build(q_input)
    return tau[use], e_oft[use]
