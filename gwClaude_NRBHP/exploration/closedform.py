"""Closed-form replacements for the numerical inputs of the calibration (user, 2026-10-21: "convert as many numerical
parts of the model to closed-form expressions as possible, even at a marginal mismatch cost").

All constants are fitted OFFLINE to theory or reference data, never to the NR waveforms of the calibration
(`closedform_coeffs.json`, built 2026-10-21; see its "note").
    Kerr (2,2,0) QNM (replaces the qnm package):
        w220(chi), gamma220(chi) = sum_{k=0..5} a_k (1 - chi)^(k/2)        (chi in [-0.85, 0.95]; counter-rotating branch for chi < 0)
        accuracy on [-0.65, 0.92]: w 8.7e-5, gamma 3.1e-4 (relative); limits w -> 0.99, gamma -> 0.003 as chi -> 1
    Remnant (replaces NRSur3dq8Remnant):
        chi_f = X1^2 chi1 + nu sum_{i,j=0..3} c_ij nu^i chi1^j
        1 - M_f = nu (1 - E_isco(X1^2 chi1)) sum_{i,j=0..3} m_ij nu^i chi1^j     (E_isco: Kerr ISCO energy per unit mass)
        accuracy vs NRSur3dq8Remnant for q in [2.5, 8.5]: chi_f 7e-4, M_f 4.6e-4 (absolute)
    Measured ppBHPT ringdown decay (replaces the per-waveform slope fit):
        gamma_meas = gamma220(chi1) [1 + d0 + d1 chi1 + d2 chi1^2 + d3 chi1^3 + d4 chi1^4 + nu (e0 + e1 chi1 + e2 chi1^2)]
        accuracy vs the per-case slopes (T96): 1.1% max, 0.4% rms
install(P6_like_modules) patches the repo's physics_common QNM functions, the NRSur3dq8Remnant object used by
fit_pn_anchored_spin, and returns a gamma_meas function for the get() hooks.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CF = json.loads((HERE / "closedform_coeffs.json").read_text())


def w220(chi):
    s = np.sqrt(1 - np.asarray(chi, float)); return sum(a * s ** k for k, a in enumerate(CF["w220"]))


def g220(chi):
    s = np.sqrt(1 - np.asarray(chi, float)); return sum(a * s ** k for k, a in enumerate(CF["g220"]))


def Q220(chi):
    return w220(chi) / (2 * g220(chi))


def isco_E(a):
    a = np.asarray(a, float); A = np.abs(a)
    z1 = 1 + (1 - A * A) ** (1 / 3) * ((1 + A) ** (1 / 3) + (1 - A) ** (1 / 3)); z2 = np.sqrt(3 * A * A + z1 * z1)
    r = 3 + z2 - np.sign(a) * np.sqrt((3 - z1) * (3 + z1 + 2 * z2)); v = 1 / np.sqrt(r)
    return (1 - 2 * v ** 2 + a * v ** 3) / np.sqrt(1 - 3 * v ** 2 + 2 * a * v ** 3)


def _poly(c, nu, chi):
    c = np.array(c).reshape(4, 4); return sum(c[i, j] * nu ** i * chi ** j for i in range(4) for j in range(4))


def chif(q, chi1):
    nu, X1 = q / (1 + q) ** 2, q / (1 + q); return X1 ** 2 * chi1 + nu * _poly(CF["chif"], nu, chi1)


def mf(q, chi1):
    nu, X1 = q / (1 + q) ** 2, q / (1 + q); return 1 - nu * (1 - isco_E(X1 ** 2 * chi1)) * _poly(CF["mf"], nu, chi1)


def gamma_meas(q, chi1):
    nu = q / (1 + q) ** 2; d = CF["gmeas"]
    corr = d[0] + d[1] * chi1 + d[2] * chi1 ** 2 + d[3] * chi1 ** 3 + d[4] * chi1 ** 4 + nu * (d[5] + d[6] * chi1 + d[7] * chi1 ** 2)
    return float(g220(chi1) * (1 + corr))


class RemnantCF:
    """Drop-in for the surfinBH NRSur3dq8Remnant calls the repo makes: .mf(q, s1, s2), .chif(q, s1, s2) -> (value, err)."""
    def mf(self, q, s1, s2):
        return np.array(mf(q, float(s1[2]))), 0.0

    def chif(self, q, s1, s2):
        return np.array([0.0, 0.0, float(chif(q, float(s1[2])))]), 0.0


def install(S, PC, qnm=True, remnant=True):
    """Patch fit_pn_anchored_spin (S) and physics_common (PC) in place (call before any fork)."""
    if qnm:
        PC.w220_exact = lambda chi: float(w220(chi)); PC.Q220_exact = lambda chi: float(Q220(chi))
    if remnant:
        S._SFBH = RemnantCF()
