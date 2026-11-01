"""Stand-alone pass-40 calibration map alpha(t), beta(t) for the spinning ppBHPT (2,2) mode (exploration leader, 18c).

One module, no exploration import chain.  Model: exploration/scaling_p40.md (pass 39 L4 coefficients + closed-form R2, O2).
Inputs per case: q, chi1 and the ppBHPT (2,2) waveform on its own clock t (m1), with two derived drives:
    x_pp(t)  = (abs(dphi/dt)/2)^(2/3), phi = unwrapped phase smoothed by a cubic Savitzky–Golay filter of 80 m1 span, clip
               [1e-8, 0.6]                                                        (= fit_pn_anchored_spin.get_x)
    E(t)     = radiated energy of the (2,2) + (2,-2) modes per unit TOTAL mass, 0 at the first sample, despiked in the
               surrogate join window                                              (= physics_common.energy_drive, "M")
The caller may pass the arrays it already has (the repo's FS.build_case gives case["x"], case["losses"]["e_oft"]).
Outputs: alpha(t), beta(t) on the same t grid.  The waveform map is then
    tau(t) = t0 + int_{-100}^{t} beta dt',     h_NR(tau(t)) = exp(i phi0) alpha(t) h_pp(t)     (t0, phi0 per case alignment)
Dependencies: numpy, scipy, qnm (Kerr QNMs), surfinBH (NRSur3dq8Remnant).  verify() checks it against the exploration chain.
"""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------------------------------------- coefficients
# pass 39 L4 (exploration/p39/p39_L4.json), the 18 fitted coefficients; everything else in the form is 0 or derived
COEF = dict(b=-0.023601, b2=0.214896, P0=-4.200635, P1=30.676596, s2=-18.472052, s3=2.552821,
            t1a=-102.52678, t1b=79.122034, r1=0.081691, r3=-0.111607, l0=-0.114598,
            pa0=-0.095969, pa2=-0.367771, eb1=0.097222, pb0=0.167711, pb1=-0.275277, pb2=0.385677, ea0=-0.065958)
NU5, NU_C = 5.0 / 36.0, 0.04          # nu at q = 5 (reference of the nu weights); test-mass switch-off scale (pass 39)
XCLIP, S_RING, GAMMA_WIN = 0.26, 15.0, (15.0, 40.0)
R2_A, R2_B = 4.768, 21.68             # R2 closed form O2 (pass 40)


def load_exact_coefficients():
    """The full-precision 18 from p39_L4.json (COEF above is rounded to 6 significant digits)."""
    th = json.loads((HERE / "p39" / "p39_L4.json").read_text())["theta"]
    names = json.loads((HERE / "p32" / "p32_G2.json").read_text())["names"]
    d = dict(zip(names, th)); m = {"b": "flux_b", "b2": "flux_b2", "P0": "flux_P0", "P1": "flux_P1", "s2": "flux_s2", "s3": "flux_s3"}
    return {k: float(d[m.get(k, k)]) for k in COEF}


# ---------------------------------------------------------------------------------------------------------- fixed shapes
@lru_cache(maxsize=1)
def shapes():
    """phi_B, phi_A (pass 12) and psi_B, psi_A (pass 17), natural cubic splines on s, clipped to their knot ranges."""
    s12 = json.loads((HERE / "p12" / "p12_shapes.json").read_text())
    p17 = json.loads((HERE / "p17" / "p17_p15.json").read_text())
    full = np.array(p17["knots"], float)

    def psi(field):
        kn = np.array(p17["pca"][field]["knots"], float); v = np.array(p17["pca"][field]["comps"][0], float)
        pad = full[full > kn[-1]]; kn, v = np.r_[kn, pad], np.r_[v, np.zeros(len(pad))]
        return CubicSpline(kn, v, bc_type="natural"), (kn[0], kn[-1])
    PB = CubicSpline(s12["knots_beta"], s12["phi_beta"], bc_type="natural"), (s12["knots_beta"][0], s12["knots_beta"][-1])
    PA = CubicSpline(s12["knots_alpha"], s12["phi_alpha"], bc_type="natural"), (s12["knots_alpha"][0], s12["knots_alpha"][-1])
    return dict(phiB=PB, phiA=PA, psiB=psi("dlnB"), psiA=psi("dlnA"))


def _ev(spl_rng, s):
    spl, (lo, hi) = spl_rng; return spl(np.clip(s, lo, hi))


# ---------------------------------------------------------------------------------------------------------- physics inputs
def smooth5(u):
    u = np.clip(u, 0.0, 1.0); return u ** 3 * (10 - 15 * u + 6 * u * u)


def kappa_so(q):
    X1 = q / (1 + q); return 1.5 * (113 / 12 * X1 ** 1.2 - 19 / 6 * X1 ** 2 - 25 / 4 * X1)


def R2(x):
    """2SF / 1SF (2,2) amplitude ratio, closed form O2; x clipped to [0.005, 1/6] (0 at and beyond the Schwarzschild ISCO)."""
    x = np.clip(x, 0.005, 1.0 / 6.0)
    return 55.0 / 21.0 * x * (1 - 6 * x) * (1 + R2_A * x + R2_B * x * x)


def g_nu(nu):
    return 1.0 - np.exp(-(nu / NU_C) ** 2)


@lru_cache(maxsize=4096)
def _qnm(chi):
    import qnm
    a, m = (chi, 2) if chi >= 0 else (-chi, -2)                       # counter-rotating (2,2) branch for chi < 0
    w = complex(qnm.modes_cache(s=-2, l=2, m=m, n=0)(a=a)[0])
    return abs(w.real), abs(w.imag)                                      # frequency, decay rate (units of the hole's mass)


@lru_cache(maxsize=1)
def _remnant():
    import surfinBH
    return surfinBH.LoadFits("NRSur3dq8Remnant")


def anchors(q, chi1):
    """beta_r (QNM clock ratio, primary vs remnant), gamma_K (primary's Kerr decay rate per m1), gamma_NR (remnant, per M)."""
    F = _remnant(); s1 = [0, 0, float(chi1)]
    mf = float(F.mf(q, s1, [0, 0, 0])[0]); chif = float(F.chif(q, s1, [0, 0, 0])[0][2])
    w_pp, g_K = _qnm(round(float(chi1), 12)); w_f, g_f = _qnm(round(chif, 12))
    return w_pp * mf / w_f, g_K, g_f / mf


def gamma_meas(t, h):
    """Measured ppBHPT ringdown decay rate per m1: minus the slope of ln|h| over s in [15, 40] m1."""
    s = t - t[int(np.argmax(np.abs(h)))]; m = (s >= GAMMA_WIN[0]) & (s <= GAMMA_WIN[1])
    return -float(np.polyfit(s[m], np.log(np.abs(h[m])), 1)[0])


# ---------------------------------------------------------------------------------------------------------- the map
def alpha_beta(q, chi1, t, h, x_pp, E, coef=None, tpk=None, gm=None, t_full=None, h_full=None):
    """alpha(t), beta(t) on the grid t (m1).  h, x_pp, E on the same grid.  tpk and gm (gamma_meas) are taken from the
    full-resolution waveform (t_full, h_full; default: the given arrays); pass them when evaluating on a sub-sampled grid.
    Returns (None, None) where the form is invalid (blend onset outside [-150, -3] m1 is only possible for chi1 < -0.6)."""
    c = COEF if coef is None else coef
    tf, hf = (t, h) if t_full is None else (t_full, h_full)
    tpk = float(tf[int(np.argmax(np.abs(hf)))]) if tpk is None else tpk
    gm = gamma_meas(tf, hf) if gm is None else gm
    nu, X1 = q / (1 + q) ** 2, q / (1 + q); w = nu / NU5; G = g_nu(nu); F = G / w      # F = g(nu) nu5/nu
    br, gK, gNR = anchors(q, chi1)
    kap = (gK + G * (gm - gK)) / br - gNR
    s = np.asarray(t) - tpk
    # inspiral branch
    xs = X1 ** -0.8 * np.minimum(x_pp, XCLIP)
    P = c["P0"] + c["P1"] * nu + F * (nu * c["s2"] * chi1 ** 2 + c["s3"] * chi1)
    bI = (X1 ** 1.2 * (1 + c["b"] * nu + c["b2"] * nu ** 2) + P * np.asarray(E)) * (1 + kappa_so(q) * chi1 * xs ** 1.5)
    if np.any(bI <= 0):
        return None, None
    laI = 1.2 * np.log(X1) + np.log1p(nu * R2(xs) / 2)
    # blend
    s1 = c["t1a"] + c["t1b"] * chi1
    if not (-150 <= s1 <= 30 and s1 + 3 <= 0):
        return None, None
    W = smooth5((s - s1) / (0.0 - s1))
    # merger-ringdown branch
    lbM = np.log(br) + G * (c["r1"] * chi1 + c["r3"] * chi1 ** 2)
    laM = 1.2 * np.log(X1) + w * c["l0"] + kap * br * np.clip(s - S_RING, 0, None)
    lb = (1 - W) * np.log(bI) + W * lbM
    la = (1 - W) * laI + W * laM
    # shape terms
    S = shapes()
    lb = lb + w * (c["pa0"] + c["pa2"] * chi1 ** 2) * _ev(S["phiB"], s) + G * c["eb1"] * chi1 * _ev(S["psiB"], s)
    la = la + G * (c["pb0"] + c["pb1"] * chi1 + c["pb2"] * chi1 ** 2) * _ev(S["phiA"], s) + w * c["ea0"] * _ev(S["psiA"], s)
    return np.exp(la), np.exp(lb)


def verify(cases=((3.0, -0.6), (5.0, 0.0), (8.0, 0.8), (4.2, 0.4), (6.589744, -0.2)), keys=("fit", "full")):
    """Compare with the exploration chain (pass 40 = pass 39 L4 + O2): max relative difference of alpha and beta."""
    import sys
    sys.path.insert(0, str(HERE))
    sys.argv = [sys.argv[0], "L4"]
    import p40_r2closed as C
    P6 = C.P6; P6.get = C.L.get; P6.ab = C.L.ab; C.use("O2n")
    th = np.array(json.loads((HERE / "p39" / "p39_L4.json").read_text())["theta"]); cf = load_exact_coefficients()
    worst = 0.0
    for k in cases:
        cc = P6.get(k); case = cc["case"]
        for key in keys:
            pre = "fit_" if key == "fit" else ""
            a0, b0 = P6.ab(th, k, key)
            a1, b1 = alpha_beta(k[0], k[1], case[pre + "t_bhpt"], case[pre + "h_bhpt"], case[pre + "x"],
                                case[pre + "losses"]["e_oft"], coef=cf, t_full=case["t_bhpt"], h_full=case["h_bhpt"])
            d = max(np.max(np.abs(a1 / a0 - 1)), np.max(np.abs(b1 / b0 - 1))); worst = max(worst, d)
            print(f"  q={k[0]:g} chi1={k[1]:+.1f} {key:4s}: max rel diff alpha/beta {d:.2e}", flush=True)
    print(f"verify: worst {worst:.2e}", flush=True)
    return worst


if __name__ == "__main__":
    verify()
