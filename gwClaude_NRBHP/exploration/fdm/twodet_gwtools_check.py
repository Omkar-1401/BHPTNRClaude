"""Cross-check twodet_core.match2 against gwtools.mismatch.mismatch (Blackman & Field; the routine behind the procedure
Rink et al. 2024 cite), face-on, on identical tapered and padded data.
gwtools takes the two-sided FFT of the complex strain H = h_+ - i h_x; Re of its complex inner product over +-f equals
<h_+|h_+> + <h_x|h_x> (two detectors), and |d(tau)| is the polarization maximisation.  The band is imposed by setting
the PSD to infinity outside [f_min, f_max] in |f|.  Expected agreement ~1e-6 relative (gwtools fits the time peak on |d|,
match2 on |d|^2).
    python fdm/twodet_gwtools_check.py MODEL
"""
from __future__ import annotations
import numpy as np

import hm_run as HR
import fd_core as FC
import twodet_core as TD
from gwtools.mismatch import mismatch as gwt_mismatch

PTS = [(5.0, 0.0), (3.0, -0.6), (8.0, 0.6)]


def gwt_mm(tc, nr, mod, m_tot, fmin="rink"):
    st = TD.setup(tc, nr[(2, 2)], m_tot, fmin); fs = st["fs"]
    tsec = (tc - tc[0]) * m_tot * FC.MTSUN; tu = np.arange(0.0, tsec[-1], 1.0 / fs)
    ip = lambda h: np.interp(tu, tsec, h.real) + 1j * np.interp(tu, tsec, h.imag)
    dt_m = 1.0 / fs / (m_tot * FC.MTSUN)
    w = FC.planck(len(tu), int(round(st["t_on"] / dt_m)), int(round(TD.T_OFF_M / dt_m)))
    n = (1 << int(np.ceil(np.log2(len(tu))))) * TD.PAD
    f = np.fft.fftfreq(n, 1.0 / fs); fa = np.abs(f)
    S = np.full(n, np.inf); band = (fa >= st["f_min"]) & (fa <= st["f_max"])
    import lalsimulation as ls
    S[band] = [ls.SimNoisePSDaLIGOZeroDetHighPower(float(x)) for x in fa[band]]
    Hs = ip(sum(TD.mgroups(nr, 0.0).values())); Ht = ip(sum(TD.mgroups(mod, 0.0).values()))
    hs, ht = np.fft.fft(Hs * w, n) / fs, np.fft.fft(Ht * w, n) / fs
    return gwt_mismatch(hs, ht, psd=S, df=f[1] - f[0])[0]


def main():
    for k in PTS:
        tc, nr, mod = HR.maps(k)
        for M in (20.0, 50.0, 200.0):
            a = TD.match2(tc, nr, mod, M)["mm"]; b = gwt_mm(tc, nr, mod, M)
            print(f"{HR.MODEL} {k} M={M:.0f}: match2 {a:.8e}  gwtools {b:.8e}  rel diff {abs(a / b - 1):.1e}", flush=True)


if __name__ == "__main__":
    main()
