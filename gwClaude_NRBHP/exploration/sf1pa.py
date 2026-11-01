"""1PAT1 chirp ingredients, ported line by line from WaSABI InspiralModels/1PAT1.m
(Wardell et al. 2023, arXiv:2112.12265, Eq. 4), Schwarzschild, quasicircular, M = 1.

    dOmega/dt = nu F0(r0) + nu^2 F1(r0),   Omega = r0^(-3/2),  x = 1/r0

    F0(r0) = 3 (1-3/r0)^(3/2) r0^(-1/2) / (1-6/r0) * [F_inf(r0) + F_hor(r0)]
    F1(r0) = 3 (1-3/r0)^(3/2) r0^(-1/2) / (1-6/r0) *
             [ F2_inf(r0) + 2 (1-3/r0)^(3/2)/(1-6/r0) * (F_inf + F_hor)(r0) * EFLx'(1/r0) ]
    EFLx(x) = z(x)/2 - (x/3) z'(x) - 1 + sqrt(1-3x) + (x/6)(7-24x)/(1-3x)^(3/2)   (first law; z = 1SF redshift)
    F2_inf(r0) = Re exp(stored(ln r0))

Data: /work2/.../BHPTNRClaude/wasabi/InspiralModels/sf_data (WaSABI, Zenodo 10.5281/zenodo.17405583).
Interpolation: cubic splines (WaSABI uses Mathematica InterpolationOrder -> 8); 2SF data cover
r0 in [6.25, 30] (x in [1/30, 0.16]).

`python sf1pa.py` runs self-checks: Newtonian limit F0 -> (96/5) Omega^(11/3), and F1/F0 at small x
against the 1PN coefficient of the binary's chirp.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline

D = Path("/work2/11702/omkarsridhar1401/stampede3/BHPTNRClaude/wasabi/InspiralModels/sf_data")


def _read(path):
    s = path.read_text().split("*)", 1)[1]
    s = s.replace("*^", "e").replace(" ", "").replace("\n", "")
    pairs = re.findall(r"\{([^{}]+)\}", s)
    out = []
    for p in pairs:
        a, b = p.split(",", 1)
        b = b.replace("*I", "j").replace("I", "1j")
        out.append((float(a), complex(b)))
    return np.array(out)


_fi = _read(D / "1SF_Flux/Schwarz_Circ/1SFCircShwarzDotEInf.m")
_fh = _read(D / "1SF_Flux/Schwarz_Circ/1SFCircShwarzDotEHorizon.m")
_f2 = _read(D / "2SF_Flux/Schwarz_Circ/2SFCircShwarzDotEInf.m")
_z = _read(D / "1SF_Local_Invariants/Schwarz_Circ/1SFCircShwarzRedshfit.m")

FI = CubicSpline(_fi[:, 0].real, _fi[:, 1].real)
FH = CubicSpline(_fh[:, 0].real, _fh[:, 1].real)
_F2ln_re = CubicSpline(_f2[:, 0].real, _f2[:, 1].real)
_F2ln_im = CubicSpline(_f2[:, 0].real, _f2[:, 1].imag)
Z = CubicSpline(_z[:, 0].real, _z[:, 1].real)
R0_MIN = float(np.exp(_f2[0, 0].real))      # 6.25
R0_MAX = float(min(_fi[-1, 0].real, np.exp(_f2[-1, 0].real)))   # 30


def F2inf(r0):
    lr = np.log(r0)
    return np.real(np.exp(_F2ln_re(lr) + 1j * _F2ln_im(lr)))


def EFLx(x):
    return 0.5 * Z(x) - x / 3 * Z(x, 1) - 1 + np.sqrt(1 - 3 * x) + x / 6 * (7 - 24 * x) / (1 - 3 * x) ** 1.5


def dEFLx(x):
    return (0.5 * Z(x, 1) - Z(x, 1) / 3 - x / 3 * Z(x, 2) - 1.5 / np.sqrt(1 - 3 * x)
            + (7 - 48 * x) / 6 / (1 - 3 * x) ** 1.5 + x / 6 * (7 - 24 * x) * 4.5 / (1 - 3 * x) ** 2.5)


def _pref(r0):
    return 3 * (1 - 3 / r0) ** 1.5 * np.sqrt(1 / r0) / (1 - 6 / r0)


def F0(r0):
    return _pref(r0) * (FI(r0) + FH(r0))


def F1(r0):
    return _pref(r0) * (F2inf(r0) + 2 * (1 - 3 / r0) ** 1.5 / (1 - 6 / r0) * (FI(r0) + FH(r0)) * dEFLx(1 / r0))


def chirp_1pa(Om, nu):
    """dOmega/dt (M = 1) for a binary of symmetric mass ratio nu at orbital frequency Om (1PAT1)."""
    r0 = Om ** (-2 / 3)
    return nu * F0(r0) + nu ** 2 * F1(r0)


def _selftest():
    x = np.array([1 / 30, 1 / 20, 1 / 12, 0.12, 0.15])
    r0 = 1 / x
    Om = x ** 1.5
    print("F0 / (96/5 Om^(11/3)):", np.round(F0(r0) / (96 / 5 * Om ** (11 / 3)), 4))
    # 1PN chirp of the binary: dOmega/dt ∝ nu [1 + (-743/336 - 11/4 nu) x + ...]  ->  F1/F0 -> -11/4 x
    print("F1/F0 / (-11/4 x):     ", np.round(F1(r0) / F0(r0) / (-11 / 4 * x), 3))
    print("r0 range", R0_MIN, R0_MAX)


if __name__ == "__main__":
    _selftest()
