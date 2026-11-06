"""Ori–Thorne join for the spin_dep_split models (decisions in README.md).

Everything here is on the ppBHPT side, in m1 units: a test particle (m2) on a Kerr hole (m1, chi1),
Ori–Thorne small parameter eta = m2/m1 = 1/q.

  kerr_isco(a)        r_isco, Omega_isco                         (OT Eqs. rms, Omegams; BPT)
  ot_constants(a)     alpha, beta, kappa, R_o, tau_o, (dtau/dt)_isco (OT Eqs. alpha1, beta1, kappa, RotauoDef,
                      dtaudt); kappa needs Finn's flux factor Edot_isco, interpolated (cubic) in a over
                      OT Table I -- it enters tau_o only as Edot^(-1/5)
  T_STAR              T at which OT's universal transition solution crosses X = 0 (the ISCO radius):
                      X'' = -X^2 - T matched to X = sqrt(-T) + 1/(8T^2) at T -> -inf
  blend(t, h, q, chi) S(t): 0 -> 1 over OT time T in [-1, T(peak of |h22|)], quintic smoothstep (C2);
                      T(t) = T* + (dtau/dt)_isco (t - t_isco) / (eta^(-1/5) tau_o)

The window end is the ppBHPT peak, not OT's T_plunge = 3.412 (user, 2026-10-03): at our mass ratios
T_plunge falls 32-81 m1 after the peak.  The blend is a modelling device, not part of OT.
`python ot_join.py` runs the self-tests.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline

# OT Table I: a, r_isco, Omega_isco, Edot, alpha, beta, kappa, R_o, tau_o  (gr-qc/0003032)
OT_TABLE = np.array([
    [-0.99, 8.972, 0.03863, 1.240, 0.0001543, 0.006626, 0.005013, 3.129, 45.50],
    [-0.9, 8.717, 0.04026, 1.233, 0.0001732, 0.007070, 0.005527, 3.117, 43.04],
    [-0.5, 7.555, 0.04935, 1.197, 0.0003070, 0.009730, 0.008966, 3.048, 32.69],
    [0.0, 6.000, 0.06804, 1.143, 0.0007716, 0.01604, 0.01955, 2.925, 21.05],
    [0.2, 5.329, 0.07998, 1.114, 0.001240, 0.02057, 0.02914, 2.852, 16.82],
    [0.5, 4.233, 0.1086, 1.053, 0.003115, 0.03270, 0.06291, 2.687, 10.93],
    [0.8, 2.907, 0.1737, 0.9144, 0.01401, 0.06446, 0.2123, 2.326, 5.539],
    [0.9, 2.321, 0.2254, 0.7895, 0.03447, 0.09039, 0.4214, 2.041, 3.770],
    [0.99, 1.454, 0.3644, 0.4148, 0.2234, 0.1289, 1.531, 1.284, 1.867],
    [0.999, 1.182, 0.4379, 0.2022, 0.5127, 0.09568, 2.594, 0.8551, 1.510]])
_EDOT = CubicSpline(OT_TABLE[:, 0], OT_TABLE[:, 3])
T_STAR = 0.7154          # recomputed and asserted in _selftest()
ETA = "1/q"              # Ori–Thorne small parameter: "1/q" (user's choice, default) or "nu" (comparison only)
WINDOW_START = -1.0      # OT: the solution leaves adiabatic inspiral at T ~ -1


def kerr_isco(a):
    A = abs(a)
    z1 = 1 + (1 - A * A) ** (1 / 3) * ((1 + A) ** (1 / 3) + (1 - A) ** (1 / 3))
    z2 = np.sqrt(3 * A * A + z1 * z1)
    r = 3 + z2 - np.sign(a) * np.sqrt((3 - z1) * (3 + z1 + 2 * z2)) if a else 6.0
    return float(r), float(1.0 / (r ** 1.5 + a))


def ot_constants(a, edot=None):
    r, om = kerr_isco(a)
    s = np.sqrt(1 - 3 / r + 2 * a / r ** 1.5)
    E = (1 - 2 / r + a / r ** 1.5) / s
    L = 2 / np.sqrt(3 * r) * (3 * np.sqrt(r) - 2 * a)
    al = 3 / r ** 6 * (r * r + 2 * (a * a * (E * E - 1) - L * L) * r + 10 * (L - a * E) ** 2)
    be = 2 / r ** 4 * ((L - a * a * E * om) * r - 3 * (L - a * E) * (1 - a * om))
    ed = float(_EDOT(a)) if edot is None else edot
    ka = 32 / 5 * om ** (7 / 3) * (1 + a / r ** 1.5) / s * ed
    return dict(r_isco=r, Omega_isco=om, alpha=al, beta=be, kappa=ka,
                R_o=(be * ka) ** 0.4 * al ** -0.6, tau_o=(al * be * ka) ** -0.2,
                dtaudt=s / (1 + a / r ** 1.5))


def ot_universal(T0=-60.0):
    """OT's universal solution X(T); returns (T at X = 0, T at X = -1e4)."""
    from scipy.integrate import solve_ivp
    zero = lambda T, y: y[0]
    blow = lambda T, y: y[0] + 1e4; blow.terminal = True
    s = solve_ivp(lambda T, y: [y[1], -y[0] ** 2 - T], (T0, 10.0),
                  [np.sqrt(-T0) + 1 / (8 * T0 ** 2), -0.5 / np.sqrt(-T0) - 0.25 / T0 ** 3],
                  rtol=1e-11, atol=1e-12, events=[zero, blow])
    return float(s.t_events[0][0]), float(s.t_events[1][0])


def t_isco_pp(t, h, chi):
    """ppBHPT time where the (2,2) frequency first reaches 2 Omega_isco(chi) before the |h| peak."""
    w = np.abs(np.gradient(np.unwrap(np.angle(h)), t))
    _, om = kerr_isco(chi)
    ipk = int(np.argmax(np.abs(h)))
    i = np.flatnonzero(w[:ipk] >= 2 * om)
    if not len(i):
        raise RuntimeError(f"ppBHPT never reaches its ISCO frequency (chi={chi})")
    i = i[0]
    return float(t[i - 1] + (2 * om - w[i - 1]) * (t[i] - t[i - 1]) / (w[i] - w[i - 1]))


def ot_time(t, t_isco, q, chi):
    c = ot_constants(chi)
    eta = 1.0 / q if ETA == "1/q" else q / (1.0 + q) ** 2
    unit = eta ** -0.2 * c["tau_o"] / c["dtaudt"]                  # m1 coordinate time per unit of T
    return T_STAR + (t - t_isco) / unit, unit


def smoothstep5(u):
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u * u)


def blend(t, h, q, chi):
    """S(t) on the ppBHPT time grid t, and a dict of the landmarks (all m1 units)."""
    ti = t_isco_pp(t, h, chi)
    T, unit = ot_time(t, ti, q, chi)
    ipk = int(np.argmax(np.abs(h)))
    T_end = T[ipk]
    if not T_end > WINDOW_START + 0.5:
        raise RuntimeError(f"peak too close to the window start: T_peak={T_end:.2f} (q={q}, chi={chi})")
    S = smoothstep5((T - WINDOW_START) / (T_end - WINDOW_START))
    return S, dict(t_isco_pp=ti, T_unit=unit, T_peak=float(T_end), t_peak_pp=float(t[ipk]),
                   t_window_start=float(ti + (WINDOW_START - T_STAR) * unit))


def _selftest():
    worst = 0.0
    for row in OT_TABLE[:7]:                     # the rows inside or bracketing our chi1 range
        a = row[0]; c = ot_constants(a, edot=row[3])
        got = [c["r_isco"], c["Omega_isco"], c["alpha"], c["beta"], c["kappa"], c["R_o"], c["tau_o"]]
        worst = max(worst, max(abs(g / x - 1) for g, x in zip(got, row[[1, 2, 4, 5, 6, 7, 8]])))
    assert worst < 5e-4, worst
    ts, tb = ot_universal()
    assert abs(ts - T_STAR) < 5e-4, ts
    assert abs(tb + np.sqrt(6e-4) - 3.412) < 2e-3, tb      # X ~ -6/(Tp - T)^2 near the divergence
    assert abs(kerr_isco(0.0)[0] - 6) < 1e-12 and abs(kerr_isco(1.0)[0] - 1) < 1e-9 \
        and abs(kerr_isco(-1.0)[0] - 9) < 1e-9
    u = np.linspace(0, 1, 11); s = smoothstep5(u)
    assert s[0] == 0 and s[-1] == 1 and np.all(np.diff(s) >= 0)
    print(f"ot_join self-test OK: Table I to {worst:.1e}, T* = {ts:.4f}, T_plunge = {tb + np.sqrt(6e-4):.4f}")


if __name__ == "__main__":
    _selftest()
