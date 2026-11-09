"""Multimode frequency-domain match (aLIGO zero-detuned high-power PSD), extending fd_core.py to higher modes.

Strain.  H(t; iota, phi) = h_+ - i h_x = sum_{l, |m|} h_lm(t) _{-2}Y_lm(iota, phi), with h_{l,-m} = (-1)^l conj(h_lm)
(aligned spins).  Any detector response F+ h+ + Fx hx = Re[K H] for one complex constant K (sky + polarisation).
  signal    s = Re[e^{i kappa} H_NR(iota, phi_s)]  for kappa in KAPPAS and phi_s in PHIS (fixed, not maximised).
            Reported: the SNR-weighted mismatch over these orientations, 1 - [sum O_i^3 rho_i^3 / sum rho_i^3]^(1/3)
            (Harry et al. 2018; rho_i = the signal's optimal SNR, so orientations where the detector sees almost no
            signal -- e.g. edge-on with the (2,2) contribution in the null polarisation -- do not dominate), plus the
            unweighted mean and the worst orientation.
  template  same iota; maximised over the time shift (FFT), the polarisation / amplitude (exact 2-D quadrature span
            {Re H, Im H}, as in fd_core) and the orbital phase phi_t (coarse grid + bounded refinement).
No per-mode phase freedom: the relative phases of the modes are physical, so a convention mismatch between the model's
and NR's modes would show up as a large mismatch (the published Rink model is the check).
f_low = 20 Hz unless the highest-|m| mode's frequency at the end of the start taper exceeds 0.9 * 20 Hz; then raised to
that frequency / 0.9 (flagged).  Tapers, sampling and band as fd_core.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar

import fd_core as FC

KAPPAS = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
PHIS = (0.0, np.pi / 3, 2 * np.pi / 3)


def sylm(iota, l, m):
    import lal
    return complex(lal.SpinWeightedSphericalHarmonic(float(iota), 0.0, -2, int(l), int(m)))


def _mgroups(modes, iota):
    """A_m(t) = sum_l h_lm Y_lm(iota, 0) for m = +-|m|; H(phi) = sum_m e^{i m phi} A_m."""
    G = {}
    for (l, m), h in modes.items():
        for mm, hh in ((m, h), (-m, (-1) ** l * np.conj(h))):
            G[mm] = G.get(mm, 0) + hh * sylm(iota, l, mm)
    return G


def mm_match(tc, nr, mod, m_tot, iota, f_low=FC.F_LOW, fs=FC.FS_HZ, n_phi=24):
    """nr, mod: {(l, m): complex array on tc} (m > 0).  Returns dict(mm mean, mm_worst, f_low, flagged)."""
    tsec = (tc - tc[0]) * m_tot * FC.MTSUN
    tu = np.arange(0.0, tsec[-1], 1.0 / fs)
    ip = lambda h: np.interp(tu, tsec, h.real) + 1j * np.interp(tu, tsec, h.imag)
    dt_m = 1.0 / fs / (m_tot * FC.MTSUN)
    i_on = int(round(FC.T_ON / dt_m))
    w = FC.planck(len(tu), i_on, int(round(FC.T_OFF / dt_m)))
    h22 = ip(nr[(2, 2)]); ph = np.unwrap(np.angle(h22))
    f22 = abs(np.gradient(ph, tu)[min(i_on, len(tu) - 1)]) / (2 * np.pi)
    mmax = max(m for _, m in nr)
    flo = max(f_low, (mmax / 2) * f22 / 0.9)
    n = 1 << int(np.ceil(np.log2(2 * len(tu))))
    f = np.fft.rfftfreq(n, 1.0 / fs); df = f[1] - f[0]
    W = np.where((f >= flo) & (f <= 0.45 * fs), 1.0 / FC.psd_zdhp(f), 0.0)
    ft = lambda x: np.fft.rfft(x * w, n) / fs
    inner = lambda a, b: 4 * df * float(np.real(np.sum(np.conj(a) * b * W)))
    # template: per-m FFTs of Re and Im of A_m, so H_t(phi) is assembled linearly in the frequency domain
    Gt = {m: ip(a) for m, a in _mgroups(mod, iota).items()}
    FR = {m: ft(a.real) for m, a in Gt.items()}; FI = {m: ft(a.imag) for m, a in Gt.items()}

    def quads(phi):                                   # FFT of Re H_t(phi), Im H_t(phi)
        U0 = sum(np.cos(m * phi) * FR[m] - np.sin(m * phi) * FI[m] for m in FR)
        U1 = sum(np.sin(m * phi) * FR[m] + np.cos(m * phi) * FI[m] for m in FR)
        return U0, U1

    Gs = {m: ip(a) for m, a in _mgroups(nr, iota).items()}
    out, rho = [], []
    for ps in PHIS:
        Hs = sum(np.exp(1j * m * ps) * a for m, a in Gs.items())
        for ka in KAPPAS:
            Sg = ft(np.real(np.exp(1j * ka) * Hs)); ss = inner(Sg, Sg); rho.append(np.sqrt(ss))

            def mism(phi):
                U = quads(phi)
                G = np.array([[inner(U[i], U[j]) for j in range(2)] for i in range(2)]); Gi = np.linalg.inv(G)
                c = np.array([2 * df * n * np.fft.irfft(np.conj(Sg) * Ui * W, n) for Ui in U])
                m2 = np.einsum("in,ij,jn->n", c, Gi, c) / ss
                k = int(np.argmax(m2)); y0, y1, y2 = m2[k - 1], m2[k], m2[(k + 1) % n]
                den = y0 - 2 * y1 + y2
                pk = y1 - 0.125 * (y2 - y0) ** 2 / den if den < 0 else y1
                return 1.0 - np.sqrt(min(pk, 1.0))
            grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
            v = [mism(p) for p in grid]; j = int(np.argmin(v)); h = grid[1] - grid[0]
            r = minimize_scalar(mism, bounds=(grid[j] - h, grid[j] + h), method="bounded", options={"xatol": 1e-4})
            out.append(min(float(r.fun), v[j]))
    out, rho = np.array(out), np.array(rho)
    snrw = 1.0 - (np.sum((1 - out) ** 3 * rho ** 3) / np.sum(rho ** 3)) ** (1 / 3)
    return dict(mm=float(snrw), mm_mean=float(out.mean()), mm_worst=float(out.max()), mm_cfg=out.tolist(),
                snr_rel=(rho / rho.max()).tolist(), f_low=float(flo), flagged=bool(flo > f_low))
