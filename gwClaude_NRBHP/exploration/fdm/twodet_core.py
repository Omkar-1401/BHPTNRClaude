"""Two-detector frequency-domain match, following Rink et al. 2024 (arXiv:2407.18319, Sec. near Fig. 12), which follows
Blackman et al. 2017: one detector sees only h_+, the other only h_x, so the two polarizations count equally.

  network inner product   <a|b> = <a_+|b_+> + <a_x|b_x>,   <u|v> = 4 Re int u~* v~ / S_n df over [f_min, f_max]
  strain                  H(t; iota, phi) = h_+ - i h_x = sum_{l,m} h_lm(t) _{-2}Y_lm(iota, phi),  h_{l,-m} = (-1)^l conj(h_lm)
  optimised over          time shift (FFT), template polarization angle psi (analytic: a rotation of (h_+, h_x) by 2 psi,
                          so max_psi = sqrt(A^2 + B^2) with A = <s_+|t_+> + <s_x|t_x>, B = <s_+|t_x> - <s_x|t_+>), and the
                          template's initial orbital phase (grid + bounded refinement; skipped when every contributing mode
                          has the same m, e.g. face-on, where it is degenerate with psi).
  signal                  fixed orientation (iota, phi_s); not optimised.
  windowing (Rink)        Planck taper over the first 1.5 cycles of the NR (2,2) and the last 20 M; zero-pad to the next
                          power of two, then by a further factor PAD = 8.
  band (Rink)             f_min = twice the orbital angular frequency at the end of the start taper (= the (2,2) frequency
                          there) -- fmin="rink"; or max(20 Hz, that) -- fmin=20.0.  f_max = 8 x the orbital angular
                          frequency at the (2,2) peak, / 2 pi.
  PSD                     aLIGO design sensitivity, taken as lalsimulation aLIGOZeroDetHighPower (Rink: "Advanced LIGO
                          design sensitivity"; the exact curve is not named).
Sampling rate: the smallest power of two >= 4096 Hz with f_max <= 0.45 fs.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar

import fd_core as FC
from hm_core import sylm

PAD = 8
T_OFF_M = 20.0
_PSD = {}


def psd(f):
    import lalsimulation as ls
    key = (len(f), float(f[1] - f[0]))
    if key not in _PSD:
        out = np.full(len(f), np.inf)
        nz = f > 0
        out[nz] = [ls.SimNoisePSDaLIGOZeroDetHighPower(float(x)) for x in f[nz]]
        _PSD[key] = out
    return _PSD[key]


def mgroups(modes, iota):
    """{m: A_m(t)} with H(phi) = sum_m e^{i m phi} A_m; modes whose harmonic vanishes at this iota are dropped."""
    G = {}
    for (l, m), h in modes.items():
        for mm, hh in ((m, h), (-m, (-1) ** l * np.conj(h))):
            y = sylm(iota, l, mm)
            if abs(y) > 1e-12:
                G[mm] = G.get(mm, 0) + hh * y
    return G


def setup(tc, h22_nr, m_tot, fmin="rink"):
    """Sampling, taper and band from the NR (2,2) (identical for every model at this point and mass)."""
    ph = np.unwrap(np.angle(h22_nr)); w22 = np.abs(np.gradient(ph, tc))                 # (2,2) angular freq [1/M]
    i_on = int(np.searchsorted(np.abs(ph - ph[0]), 1.5 * 2 * np.pi))                    # 1.5 cycles of (2,2)
    t_on = tc[i_on] - tc[0]
    f_on = w22[i_on] / (2 * np.pi) / (m_tot * FC.MTSUN)                                 # = 2 x orbital freq [Hz]
    f_max = 8 * (w22[int(np.argmax(np.abs(h22_nr)))] / 2) / (2 * np.pi) / (m_tot * FC.MTSUN)
    fs = 4096.0
    while f_max > 0.45 * fs:
        fs *= 2
    f_min = f_on if fmin == "rink" else max(float(fmin), f_on)
    return dict(t_on=t_on, f_on=f_on, f_min=f_min, f_max=f_max, fs=fs)


def match2(tc, nr, mod, m_tot, iota=0.0, phi_s=0.0, fmin="rink", n_phi=24, psi_s=0.0):
    """nr, mod: {(l, m): complex array on tc}, m > 0.  psi_s rotates the SIGNAL's polarization (a gate: the two-detector
    match must not depend on it).  Returns dict(mm, f_min, f_max, fs, ...)."""
    st = setup(tc, nr[(2, 2)], m_tot, fmin)
    fs = st["fs"]
    tsec = (tc - tc[0]) * m_tot * FC.MTSUN
    tu = np.arange(0.0, tsec[-1], 1.0 / fs)
    ip = lambda h: np.interp(tu, tsec, h.real) + 1j * np.interp(tu, tsec, h.imag)
    dt_m = 1.0 / fs / (m_tot * FC.MTSUN)
    w = FC.planck(len(tu), int(round(st["t_on"] / dt_m)), int(round(T_OFF_M / dt_m)))
    n = (1 << int(np.ceil(np.log2(len(tu))))) * PAD
    f = np.fft.rfftfreq(n, 1.0 / fs); df = f[1] - f[0]
    W = np.where((f >= st["f_min"]) & (f <= st["f_max"]), 1.0 / psd(f), 0.0)
    ft = lambda x: np.fft.rfft(x * w, n) / fs
    inner = lambda a, b: 4 * df * float(np.real(np.sum(np.conj(a) * b * W)))
    corr = lambda a, b: 2 * df * n * np.fft.irfft(np.conj(a) * b * W, n)                 # <a|b shifted by tau>, all tau

    Hs = sum(np.exp(1j * m * phi_s) * a for m, a in mgroups(nr, iota).items()) * np.exp(2j * psi_s)
    Hs = ip(Hs); Sp, Sx = ft(Hs.real), ft(-Hs.imag)
    ss = inner(Sp, Sp) + inner(Sx, Sx)
    Gt = {m: ip(a) for m, a in mgroups(mod, iota).items()}
    FR = {m: ft(a.real) for m, a in Gt.items()}; FI = {m: ft(a.imag) for m, a in Gt.items()}

    def mism(phi):
        # H_t(phi) = sum_m e^{i m phi} A_m;  t_+ = Re H_t, t_x = -Im H_t  (FFT is linear)
        Tp = sum(np.cos(m * phi) * FR[m] - np.sin(m * phi) * FI[m] for m in FR)
        Tx = -sum(np.sin(m * phi) * FR[m] + np.cos(m * phi) * FI[m] for m in FR)
        tt = inner(Tp, Tp) + inner(Tx, Tx)
        A = corr(Sp, Tp) + corr(Sx, Tx); B = corr(Sp, Tx) - corr(Sx, Tp)
        m2 = (A ** 2 + B ** 2) / (ss * tt)
        k = int(np.argmax(m2)); y0, y1, y2 = m2[k - 1], m2[k], m2[(k + 1) % n]
        den = y0 - 2 * y1 + y2
        pk = y1 - 0.125 * (y2 - y0) ** 2 / den if den < 0 else y1
        return 1.0 - np.sqrt(min(pk, 1.0))

    ms = set(FR)
    if len({abs(m) for m in ms}) == 1 and len(ms) == 1:                                  # one m only: phi_t == psi
        best = mism(0.0)
    else:
        grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        v = [mism(p) for p in grid]; j = int(np.argmin(v)); h = grid[1] - grid[0]
        r = minimize_scalar(mism, bounds=(grid[j] - h, grid[j] + h), method="bounded", options={"xatol": 1e-4})
        best = min(float(r.fun), v[j])
    return dict(mm=float(best), f_min=st["f_min"], f_on=st["f_on"], f_max=st["f_max"], fs=fs,
                n_m=len(ms), raised=bool(st["f_min"] > st["f_on"]), rho=float(np.sqrt(ss)))


def brute_psi(tc, nr, mod, m_tot, iota=0.0, fmin="rink", n_psi=72):
    """Gate: explicit template polarization rotation on a psi grid (phi_t = 0), max over time by FFT; compare with the
    analytic sqrt(A^2 + B^2) of match2 at the same phi_t (only meaningful when phi_t is degenerate, e.g. face-on)."""
    st = setup(tc, nr[(2, 2)], m_tot, fmin); fs = st["fs"]
    tsec = (tc - tc[0]) * m_tot * FC.MTSUN; tu = np.arange(0.0, tsec[-1], 1.0 / fs)
    ip = lambda h: np.interp(tu, tsec, h.real) + 1j * np.interp(tu, tsec, h.imag)
    dt_m = 1.0 / fs / (m_tot * FC.MTSUN)
    w = FC.planck(len(tu), int(round(st["t_on"] / dt_m)), int(round(T_OFF_M / dt_m)))
    n = (1 << int(np.ceil(np.log2(len(tu))))) * PAD
    f = np.fft.rfftfreq(n, 1.0 / fs); df = f[1] - f[0]
    W = np.where((f >= st["f_min"]) & (f <= st["f_max"]), 1.0 / psd(f), 0.0)
    ft = lambda x: np.fft.rfft(x * w, n) / fs
    inner = lambda a, b: 4 * df * float(np.real(np.sum(np.conj(a) * b * W)))
    corr = lambda a, b: 2 * df * n * np.fft.irfft(np.conj(a) * b * W, n)
    Hs = ip(sum(mgroups(nr, iota).values())); Ht = ip(sum(mgroups(mod, iota).values()))
    Sp, Sx = ft(Hs.real), ft(-Hs.imag); ss = inner(Sp, Sp) + inner(Sx, Sx)
    def ov(psi):                                       # explicit rotation, max over the FFT time grid (no refinement)
        Hr = Ht * np.exp(2j * psi)
        Tp, Tx = ft(Hr.real), ft(-Hr.imag); tt = inner(Tp, Tp) + inner(Tx, Tx)
        return float(np.max(corr(Sp, Tp) + corr(Sx, Tx))) / np.sqrt(ss * tt)
    g = np.linspace(0, np.pi, n_psi, endpoint=False); v = [ov(p) for p in g]; j = int(np.argmax(v)); h = g[1] - g[0]
    r = minimize_scalar(lambda p: -ov(p), bounds=(g[j] - h, g[j] + h), method="bounded", options={"xatol": 1e-7})
    brute = 1.0 - max(-float(r.fun), v[j])
    Tp, Tx = ft(Ht.real), ft(-Ht.imag); tt = inner(Tp, Tp) + inner(Tx, Tx)       # analytic psi, same time grid
    A = corr(Sp, Tp) + corr(Sx, Tx); B = corr(Sp, Tx) - corr(Sx, Tp)
    analytic = 1.0 - float(np.max(np.sqrt((A ** 2 + B ** 2) / (ss * tt))))
    return brute, analytic
