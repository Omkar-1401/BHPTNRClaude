"""1PA theory clock for the inspiral beta (pass 43; theory sub-model, user rule 2026-10-18).

Inputs: WaSABI (Zenodo 10.5281/zenodo.17405583, ../wasabi, read-only) Schwarzschild circular data:
    1SF energy flux at infinity + horizon  (r0 in [6, ...]),  2SF flux at infinity  {ln r0, ln F2 (complex: + i pi = negative)},
    1SF redshift z(x).
WaSABI 1PAT1 (InspiralModels/1PAT1.m), M = 1:
    dOmega/dt = nu F0(r0) + nu^2 F1(r0),  Omega = r0^(-3/2)
    F0 = 3 (1-3/r0)^(3/2) r0^(-1/2) / (1-6/r0) * Fdot1,           Fdot1 = 1SF flux (inf + horizon)
    F1 = 3 (1-3/r0)^(3/2) r0^(-1/2) / (1-6/r0) * [ Fdot2_inf + 2 (1-3/r0)^(3/2)/(1-6/r0) Fdot1 EFLx'(1/r0) ]
    EFLx(x) = z/2 - (x/3) z' - 1 + sqrt(1-3x) + (x/6)(7-24x)/(1-3x)^(3/2)
The ppBHPT inspiral is 0PA in eps = m2/m1 on the m1 clock: dOmega_pp/dt = eps F0.  The binary (NR) is 1PA in nu on the M
clock.  Equal accumulated orbital phase pairs the two:  Phi_pp(Omega_pp) = Phi_NR(Omega_NR) + c, and the phase-aligned
clock rate is beta = dtau/dt = Omega_pp / Omega_NR (m1 and M units).  With Phi = int Omega / (dOmega/dt) dOmega this gives
beta -> X1^(6/5) at early times (Newtonian chirp), and the 1PA drift; c (the alignment phase) is NOT fixed by theory.
"""
from __future__ import annotations
import re
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline

WD = Path(__file__).resolve().parents[2] / "wasabi" / "InspiralModels" / "sf_data"


def _load(path, cplx=False):
    s = Path(path).read_text(); s = s[s.index("{"):].replace("*^", "e")
    pairs = re.findall(r"\{\s*([^,{}]+?)\s*,\s*([^{}]+?)\s*\}", s)
    if cplx:
        out = []
        for a, b in pairs:
            m = re.match(r"([-+0-9.eE]+)\s*([-+]\s*[0-9.eE]+)\*I", b.replace(" ", ""))
            re_, im_ = (float(m.group(1)), float(m.group(2))) if m else (float(b), 0.0)
            out.append((float(a), (np.exp(complex(re_, im_))).real))
        return np.array(out)
    return np.array([(float(a), float(b)) for a, b in pairs])


@lru_cache(maxsize=1)
def data():
    inf = _load(WD / "1SF_Flux/Schwarz_Circ/1SFCircShwarzDotEInf.m")
    hor = _load(WD / "1SF_Flux/Schwarz_Circ/1SFCircShwarzDotEHorizon.m")
    f2 = _load(WD / "2SF_Flux/Schwarz_Circ/2SFCircShwarzDotEInf.m", cplx=True)
    z = _load(WD / "1SF_Local_Invariants/Schwarz_Circ/1SFCircShwarzRedshfit.m")
    assert np.allclose(inf[:, 0], hor[:, 0])
    F1sf = CubicSpline(inf[:, 0], inf[:, 1] + hor[:, 1])                  # in r0
    F2sf = CubicSpline(np.exp(f2[:, 0]), f2[:, 1])                        # in r0 (file stores ln r0)
    zs = CubicSpline(z[:, 0], z[:, 1])                                    # in x
    return dict(F1sf=F1sf, F2sf=F2sf, z=zs, r_lo=max(inf[0, 0], np.exp(f2[0, 0])), r_hi=min(inf[-1, 0], np.exp(f2[-1, 0])),
                x_lo_z=z[0, 0], x_hi_z=z[-1, 0])


def EFLx_prime(x):
    d = data(); z, zp, zpp = d["z"](x), d["z"](x, 1), d["z"](x, 2)
    # EFLx = z/2 - (x/3) z' - 1 + sqrt(1-3x) + (x/6)(7-24x)(1-3x)^(-3/2)
    t1 = zp / 2 - (zp / 3 + x * zpp / 3)
    t2 = -1.5 / np.sqrt(1 - 3 * x)
    g = (7 - 24 * x) * x / 6
    gp = (7 - 48 * x) / 6
    t3 = gp * (1 - 3 * x) ** -1.5 + g * 4.5 * (1 - 3 * x) ** -2.5
    return t1 + t2 + t3


def F0F1(r0):
    d = data(); pref = 3 * (1 - 3 / r0) ** 1.5 * r0 ** -0.5 / (1 - 6 / r0)
    f1 = d["F1sf"](r0)
    F0 = pref * f1
    F1 = pref * (d["F2sf"](r0) + 2 * (1 - 3 / r0) ** 1.5 / (1 - 6 / r0) * f1 * EFLx_prime(1 / r0))
    return F0, F1


XA = 0.035                                       # reference x for the phase origin (inside the data, below every window start)


@lru_cache(maxsize=64)
def phase_tables(nu, n=4000):
    """x grid, I0(x) = int_XA^x Omega/F0 dOmega (pp, per unit 1/eps) and J(x) = int Omega/(nu F0 + nu^2 F1) dOmega (NR)."""
    d = data(); x = np.linspace(XA, 1 / d["r_lo"], n)
    F0, F1 = F0F1(1 / x); Om = x ** 1.5; dOm = 1.5 * x ** 0.5
    den = nu * F0 + nu ** 2 * F1
    ok = den > 0.3 * nu * F0                     # 1PA breakdown guard near the ISCO (nu F1/F0 < -0.7)
    from scipy.integrate import cumulative_trapezoid as ctz
    I0 = ctz(Om / F0 * dOm, x, initial=0)
    J = ctz(np.where(ok, Om / np.where(ok, den, 1) * dOm, 0), x, initial=0)
    xJmax = x[np.argmin(ok)] if not ok.all() else x[-1]
    return x, I0, J, float(xJmax)


def beta_th(x_pp, q, c):
    """Phase-aligned 1PA clock beta(x_pp) = (x_pp / x_NR)^(3/2), pairing Phi_pp(x_pp) - Phi_pp(XA) = Phi_NR(x_NR) - Phi_NR(XB) + c,
    XB = XA X1^(-4/5) (the Newtonian pairing).  x_pp and x_NR are clipped to the data / 1PA range; beta held constant beyond."""
    eps, nu, X1 = 1 / q, q / (1 + q) ** 2, q / (1 + q)
    x, I0, J, xJmax = phase_tables(round(nu, 12))
    xp = np.clip(x_pp, XA, x[-1])
    phi_pp = np.interp(xp, x, I0) / eps
    XB = XA * X1 ** -0.8
    target = phi_pp + np.interp(XB, x, J) + c
    m = x <= xJmax
    xn = np.interp(target, J[m], x[m])           # J monotone increasing on the valid range
    return (xp / xn) ** 1.5, xn
