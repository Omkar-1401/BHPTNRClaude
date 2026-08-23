"""BHPTNRPNAnchored — PN-anchored ppBHPT->NR (2,2) calibrated waveform.

Usage
-----
    from BHPTNRPNAnchored import generate_pn_anchored_calibrated
    t, h = generate_pn_anchored_calibrated(q_input=5.0)

    t : 1-D array, time in units of M (merger at t = 0, uniform dt = 0.1)
    h : 1-D complex array, calibrated (2,2) mode waveform

Model (per the "Physical Anchors" note, alpha_beta_pn_scaling_note_revised.pdf).
The q-dependence of the leading scaling is IMPOSED analytically rather than fitted:
the inspiral amplitude and time rescalings are anchored to X1^(6/5) (the Newtonian
chirp + quadrupole result) with the leading finite-mass amplitude slope fixed at the
1PN value 55/42, and only small, nu-suppressed residuals are calibrated. Because the
q-dependence is a KNOWN function of q rather than an extrapolated polynomial, q<3 is
an evaluation, not an extrapolation -- which is why it stays smooth below the training
range (no q=2.25 notch).

    X1 = q/(1+q);  nu = q/(1+q)^2;  base = X1^(6/5)
    x  = (M Omega_orb)^(2/3) = (omega_gw/2)^(2/3)  from the ppBHPT (2,2) phase
    xc = min(x, x_clip)
    p_loss = wf-flux loss coordinate  (0.5*(Ehat+Jhat), no surfinBH)  -- switch only
    S  = sigmoid( (p_loss - (p0_0 + p0_1*nu)) / w0 )

    beta_insp = base * [ 1 + nu*(b1*xc + b2*xc^2) ]
    alpha_insp= base * [ 1 + (55/42)*nu*xc + nu*a2*xc^2 ]      # 55/42 FIXED (1PN)
    beta_r    = beta_r_physical(q) * r0        # QNM remnant time-stretch (NRSur3dq8Remnant)
    alpha_mr  = base * (m0 + m1*nu)

    beta  = (1-S)*beta_insp + S*beta_r
    alpha = (1-S)*alpha_insp + S*alpha_mr
    tau(t)= integral beta dt   (merger -> t = 0);   h(tau) = alpha * h_BHPT   (phi0 = 0)

Fit: global joint (Powell) of the 9 residual params over the even-nu [3,8] set.
Coefficients: pn_anchored_results/coeffs.json.

CAVEATS (documented in scaling_pn_anchored.md): the inspiral is faithfully PN-anchored
and carries ~87% of the (2,2) signal power; the merger-ringdown branch (beta_r QNM,
alpha_mr) is present but UNDER-ENGAGED in this fit (the switch stays weak), so the merger
is largely shaped by the inspiral polynomial + the x_clip, and alpha_mr(q=2) ~ 1.0 is not
physical. The MR sector is thus not yet a genuine remnant anchor.

The first call for a given q loads the BHPT surrogate (~1 min); cached thereafter.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "BHPTNRSurrogate" / "surrogates"))
sys.path.insert(0, str(ROOT.parent / "BHPTutils"))

import fit_scaling_PN_opt_creative as _creative
import fit_scaling_wf_nu_q_dep as _wfnu

_MODE = (2, 2)
_COEFF_JSON = ROOT / "pn_anchored_results" / "coeffs.json"
_T_ANCHOR = -100.0
_T_START_VALIDATED = _creative.NR_T_START           # ~ -5000 M
_T_END = _creative.NR_T_END                          # 100 M
_PN_SLOPE = 55.0 / 42.0
_XCLIP = 0.26
_C: dict | None = None
_SFBH = None


def _cfg() -> dict:
    global _C
    if _C is None:
        d = json.loads(_COEFF_JSON.read_text())
        _C = {"theta": np.array(d["theta"], float), "qnm": d["qnm"],
              "xclip": float(d["fixed"]["x_clip"]), "slope": float(d["fixed"]["alpha_pn_slope"])}
    return _C


def _remnant(q: float):
    global _SFBH
    if _SFBH is None:
        import surfinBH
        _SFBH = surfinBH.LoadFits("NRSur3dq8Remnant")
    mf, _ = _SFBH.mf(q, [0, 0, 0], [0, 0, 0])
    chif, _ = _SFBH.chif(q, [0, 0, 0], [0, 0, 0])
    return float(mf), float(chif[2])


def _beta_r_phys(q: float) -> float:
    mf, chif = _remnant(q)
    qn = _cfg()["qnm"]
    return (0.3683 * (1.0 + q) / q) / ((qn["F1"] + qn["F2"] * (1.0 - chif) ** qn["F3"]) / mf)


@lru_cache(maxsize=16)
def _load_bhpt(q: float):
    import BHPTNRSur1dq1e4 as bhptsur
    t, h = bhptsur.generate_surrogate(q=q, calibrated=False, modes=[_MODE], neg_modes=False)
    return t, h[_MODE]


def _get_x(t: np.ndarray, h: np.ndarray) -> np.ndarray:
    ph = np.unwrap(np.angle(h))
    n = len(ph); win = min(401, n - (1 - n % 2)); win = win - 1 if win % 2 == 0 else win
    if win >= 11:
        ph = savgol_filter(ph, win, 3, mode="interp")
    return np.clip((0.5 * np.abs(np.gradient(ph, t))) ** (2.0 / 3.0), 1e-8, 0.6)


def _alpha_beta(q: float, t: np.ndarray, h: np.ndarray, p_loss: np.ndarray):
    cfg = _cfg(); b1, b2, a2, p00, p01, w0, r0, m0, m1 = cfg["theta"]
    X1 = q / (1.0 + q); nu = q / (1.0 + q) ** 2; base = X1 ** 1.2
    xc = np.minimum(_get_x(t, h), cfg["xclip"])
    S = _creative.sigmoid((p_loss - (p00 + p01 * nu)) / max(w0, 1e-3))
    beta = (1 - S) * base * (1 + nu * (b1 * xc + b2 * xc ** 2)) + S * _beta_r_phys(q) * r0
    alpha = (1 - S) * base * (1 + cfg["slope"] * nu * xc + nu * a2 * xc ** 2) + S * base * (m0 + m1 * nu)
    return alpha, beta


def _build(q_input: float, t_start=_T_START_VALIDATED):
    q = float(q_input)
    t_bhpt, h_bhpt = _load_bhpt(q)
    meta = {"nu": q / (1 + q) ** 2, "t_bhpt_merger": float(t_bhpt[np.argmax(np.abs(h_bhpt))]),
            "t_nr_merger": 0.0}
    p_loss = _wfnu.wf_loss_coordinates(t_bhpt, h_bhpt, meta)["p_loss"]
    alpha, beta = _alpha_beta(q, t_bhpt, h_bhpt, p_loss)
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
    return tau, h_scaled, use


def generate_pn_anchored_calibrated(q_input: float, dt: float = 0.1,
                                    t_start=_T_START_VALIDATED):
    """Generate the PN-anchored calibrated BHPT (2,2) waveform (merger at t=0, dt=0.1).

    q_input : mass ratio >= 1. Calibrated on [3,8]; q in [2, 2.75] is a supported
              extrapolation (q=2 mathcalE ~ 2.6e-3). t_start=None emits the full inspiral.
    """
    tau, h_scaled, use = _build(q_input, t_start=t_start)
    t_out = np.arange(tau[use][0], tau[use][-1], dt)
    return t_out, _creative.interp_complex(tau[use], h_scaled[use], t_out)


def get_alpha_beta(q_input: float):
    """Return (t_NR, alpha(t), beta(t)) for the model at q_input (merger at t=0)."""
    q = float(q_input)
    t_bhpt, h_bhpt = _load_bhpt(q)
    meta = {"nu": q / (1 + q) ** 2, "t_bhpt_merger": float(t_bhpt[np.argmax(np.abs(h_bhpt))]),
            "t_nr_merger": 0.0}
    p_loss = _wfnu.wf_loss_coordinates(t_bhpt, h_bhpt, meta)["p_loss"]
    alpha, beta = _alpha_beta(q, t_bhpt, h_bhpt, p_loss)
    bc = _creative.cumulative_trapezoid(beta, t_bhpt)
    m_b = int(np.argmax(np.abs(h_bhpt)))
    tau = bc - bc[m_b]
    use = np.r_[True, np.diff(tau) > 0] & (tau >= _T_START_VALIDATED) & (tau <= _T_END)
    return tau[use], alpha[use], beta[use]
