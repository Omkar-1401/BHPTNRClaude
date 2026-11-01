"""Frequency-domain match between NR and a calibrated model, (2,2) mode, aLIGO zero-detuned high-power PSD.

Detector strain.  For an aligned-spin (2,2)-only signal, h_+ - i h_x = h22 Y22(iota, phi) + h2-2 Y2-2 with h2-2 = conj(h22),
so ANY detector response F+ h+ + Fx hx = Re[K h22(t)] for one complex constant K(iota, phi, psi, sky).  |K| cancels in
the normalised match, so the signal's orientation enters only through arg K = delta.  The template's polarisation and
phase are maximised over the whole 2-D span {a Re h_model + b Im h_model}:
    max_{a,b,tau} <s | a u0 + b u1>_tau / (|s| |a u0 + b u1|) = max_tau sqrt(c(tau)^T G^-1 c(tau)) / |s|,
c_i(tau) = <s | u_i shifted by tau>, G_ij = <u_i | u_j>.  This is exact (it does not assume the two quadratures have equal
norm).  The signal's delta is not maximised: the match is reported for delta in DELTAS (mean and worst).
<a|b> = 4 Re int a~* b~ / S_n df over [f_low, f_high].  Time in seconds = t[M] * M_tot * MTSUN.
Tapers: Planck window over the first T_ON M and the last T_OFF M of the common NR / model window.
f_low = F_LOW Hz unless the NR (2,2) frequency at the end of the start taper exceeds 0.9 F_LOW; then f_low is raised to
f_taper / 0.9 and the case is flagged.
"""
from __future__ import annotations
import numpy as np

MTSUN = 4.925490947641267e-06          # G Msun / c^3 [s]
F_LOW, FS_HZ, T_ON, T_OFF = 20.0, 4096.0, 400.0, 20.0
DELTAS = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
_PSD = {}


def psd_zdhp(f):
    import lalsimulation as ls
    key = (len(f), float(f[1] - f[0]))
    if key not in _PSD:
        out = np.full(len(f), np.inf)
        for i, x in enumerate(f):
            if x > 0:
                out[i] = ls.SimNoisePSDaLIGOZeroDetHighPower(float(x))
        _PSD[key] = out
    return _PSD[key]


def planck(n, n_on, n_off):
    w = np.ones(n)
    for sl, m, rev in ((slice(0, n_on), n_on, False), (slice(n - n_off, n), n_off, True)):
        if m < 2:
            continue
        x = np.arange(1, m) / m
        with np.errstate(over="ignore", divide="ignore"):
            r = 1.0 / (1.0 + np.exp(1.0 / x - 1.0 / (1.0 - x)))
        seg = np.r_[0.0, r]
        w[sl] = seg[::-1] if rev else seg
    return w


def match(tc, h_sig, h_tmp, m_tot, psd="zdhp", f_low=F_LOW, fs=FS_HZ, deltas=DELTAS):
    """tc [M] uniform-or-not common grid; h_sig (NR), h_tmp (model) complex (2,2) on tc.  Returns dict."""
    tsec = (tc - tc[0]) * m_tot * MTSUN
    tu = np.arange(0.0, tsec[-1], 1.0 / fs)
    ip = lambda h: np.interp(tu, tsec, h.real) + 1j * np.interp(tu, tsec, h.imag)
    hs, ht = ip(h_sig), ip(h_tmp)
    dt_m = 1.0 / fs / (m_tot * MTSUN)                                  # sample spacing in M
    w = planck(len(tu), int(round(T_ON / dt_m)), int(round(T_OFF / dt_m)))
    # f_low guard: NR (2,2) frequency at the end of the start taper
    ph = np.unwrap(np.angle(hs)); i_on = int(round(T_ON / dt_m))
    f_on = abs(np.gradient(ph, tu)[min(i_on, len(tu) - 1)]) / (2 * np.pi)
    flo = max(f_low, f_on / 0.9)
    n = 1 << int(np.ceil(np.log2(2 * len(tu))))
    f = np.fft.rfftfreq(n, 1.0 / fs); df = f[1] - f[0]
    S = psd_zdhp(f) if psd == "zdhp" else np.ones_like(f)
    W = np.where((f >= flo) & (f <= 0.45 * fs), 1.0 / S, 0.0)
    U = [np.fft.rfft(x * w, n) / fs for x in (ht.real, ht.imag)]        # continuous-FT normalisation
    inner = lambda a, b: 4 * df * float(np.real(np.sum(np.conj(a) * b * W)))
    G = np.array([[inner(U[i], U[j]) for j in range(2)] for i in range(2)]); Gi = np.linalg.inv(G)
    out = []
    for d in deltas:
        Sg = np.fft.rfft(np.real(np.exp(1j * d) * hs) * w, n) / fs
        ss = inner(Sg, Sg)
        c = np.array([2 * df * n * np.fft.irfft(np.conj(Sg) * Ui * W, n) for Ui in U])     # c_i(tau) for all shifts
        m2 = np.einsum("in,ij,jn->n", c, Gi, c) / ss
        k = int(np.argmax(m2)); y0, y1, y2 = m2[k - 1], m2[k], m2[(k + 1) % n]          # parabolic refine
        den = y0 - 2 * y1 + y2
        peak = y1 - 0.125 * (y2 - y0) ** 2 / den if den < 0 else y1
        out.append(1.0 - np.sqrt(min(peak, 1.0)))
    out = np.array(out)
    return dict(mm=float(out.mean()), mm_worst=float(out.max()), mm_delta=out.tolist(), f_low=float(flo),
                flagged=bool(flo > f_low), f_taper=float(f_on))


def detector_strain_check(tc, h22, iota=1.0, phi=0.4, psi=0.7, theta=0.9, varphi=2.1):
    """Explicit h+ / hx -> F+ h+ + Fx hx for an arbitrary orientation; returns the real strain (for the degeneracy test)."""
    c = np.cos(iota)
    Y22 = np.sqrt(5 / (64 * np.pi)) * (1 + c) ** 2 * np.exp(2j * phi)
    Y2m2 = np.sqrt(5 / (64 * np.pi)) * (1 - c) ** 2 * np.exp(-2j * phi)
    hc = h22 * Y22 + np.conj(h22) * Y2m2                                 # h+ - i hx
    hp, hx = hc.real, -hc.imag
    Fp = 0.5 * (1 + np.cos(theta) ** 2) * np.cos(2 * varphi) * np.cos(2 * psi) - np.cos(theta) * np.sin(2 * varphi) * np.sin(2 * psi)
    Fx = 0.5 * (1 + np.cos(theta) ** 2) * np.cos(2 * varphi) * np.sin(2 * psi) + np.cos(theta) * np.sin(2 * varphi) * np.cos(2 * psi)
    return Fp * hp + Fx * hx
