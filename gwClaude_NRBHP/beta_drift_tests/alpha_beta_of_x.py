"""Measure alpha(x) and beta(x) PARAMETER-FREE and PN-FREE, from chirp rates.

Kinematics only (no PN):  phase matching gives Omega_NR = Omega_pp/beta, hence
    x_NR = beta^(-2/3) x_pp            with x = (M Omega_orb)^(2/3)
and the chirp rate transforms as
    dx_NR/dtau = beta^(-5/3) (dx_pp/dt).
Matching to the MEASURED NR chirp rate at the same x_NR gives an implicit equation

    beta^(5/3)(x_NR) = xdot_pp(beta^(2/3) x_NR) / xdot_NR(x_NR)

solved by fixed-point iteration from beta = X1^(6/5).  Then

    alpha(x_NR) = A_NR(x_NR) / A_pp(beta^(2/3) x_NR).

Because the matching is at fixed x rather than fixed time, there is NO integration
constant: no phi0, no t0, no dphi gauge freedom.  Nothing is fitted.

Equivalently, via dx/dt = -Edot/(dE_bind/dx),
    beta^(5/3) = (Edot_pp/Edot_NR) * (dE_bind,NR/dx)/(dE_bind,pp/dx)
i.e. a measured dissipative factor times a measured conservative factor -- the same
split as the 1PA F_1/F_0 of Wardell et al. Eq (5), but measured, not imported.

Checks printed:
  * beta at the low-x end vs the derived anchor X1^(6/5)  (note Eq 12)
  * whether beta RISES or FALLS with x through the inspiral, at every q
  * alpha at the low-x end vs X1^(6/5) (note Eq 15)
"""
import sys, json, warnings
import numpy as np
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")
HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fit_scaling_PN_opt_creative_q_dep as qdep

QS = [3.0, 4.0, 5.0, 6.0, 8.0, 2.0]
X_ISCO = 1.0 / 6.0
X_MAX = 0.15                 # stay below ISCO: the adiabatic picture dies at 1/6


def inst_x_and_rate(t, h):
    """x = (omega_gw/2)^{2/3} and dx/dt, both smoothed; omega from Im(conj h * hdot)."""
    t = np.asarray(t, float); h = np.asarray(h)
    dh = np.gradient(h, t)
    w = np.abs(np.imag(np.conj(h) * dh) / np.maximum(np.abs(h) ** 2, 1e-300))
    # smooth omega over ~an orbit before forming x and its derivative
    n = len(t); win = min(401, n - (1 - n % 2)); win = win - 1 if win % 2 == 0 else win
    if win >= 11:
        w = savgol_filter(w, win, 3, mode="interp")
    x = (0.5 * w) ** (2.0 / 3.0)
    if win >= 11:
        x = savgol_filter(x, win, 3, mode="interp")
        xdot = savgol_filter(x, win, 3, mode="interp", deriv=1,
                             delta=float(np.median(np.diff(t))))
    else:
        xdot = np.gradient(x, t)
    return x, xdot, np.abs(h)


def monotone_inspiral(x, xdot, A, x_max):
    """Restrict to the rising, pre-merger, sub-ISCO stretch."""
    ipk = int(np.argmax(x))                 # x peaks at/near merger
    s = slice(0, ipk)
    x, xdot, A = x[s], xdot[s], A[s]
    m = (xdot > 0) & (x > 0) & (x < x_max)
    x, xdot, A = x[m], xdot[m], A[m]
    # enforce strict monotonicity for interpolation
    keep = np.r_[True, np.diff(x) > 0]
    return x[keep], xdot[keep], A[keep]


rows = {}
print(f"{'q':>5s} {'X1^(6/5)':>9s} | {'beta(low x)':>11s} {'ratio':>7s} "
      f"| {'beta(hi x)':>10s} {'drift':>8s} trend | {'alpha(low x)':>12s} {'ratio':>7s}")
for q in QS:
    d = np.load(qdep.waveform_cache_path(q))
    t_pp = d["t_bhpt"]; h_pp = d["h_bhpt_re"] + 1j * d["h_bhpt_im"]
    t_nr = d["t_nr"];   h_nr = d["h_nr_re"]   + 1j * d["h_nr_im"]

    xp, xdp, Ap = monotone_inspiral(*inst_x_and_rate(t_pp, h_pp), X_MAX)
    xn, xdn, An = monotone_inspiral(*inst_x_and_rate(t_nr, h_nr), X_MAX)

    X1 = q / (1.0 + q); anchor = X1 ** 1.2

    # interpolants in x
    f_xdp = lambda xx: np.interp(xx, xp, xdp, left=np.nan, right=np.nan)
    f_Ap = lambda xx: np.interp(xx, xp, Ap, left=np.nan, right=np.nan)

    # x_NR grid inside the region where x_pp = beta^{2/3} x_NR stays in the pp range
    lo = max(xn[0], xp[0] / anchor ** (2.0 / 3.0)) * 1.05
    hi = min(xn[-1], xp[-1] / anchor ** (2.0 / 3.0)) * 0.95
    if not (hi > lo):
        print(f"{q:5.2f}   no overlap"); continue
    xg = np.linspace(lo, hi, 400)

    # fixed-point iteration on beta(x)
    beta = np.full_like(xg, anchor)
    for _ in range(80):
        xpp = beta ** (2.0 / 3.0) * xg
        r = f_xdp(xpp) / np.interp(xg, xn, xdn)
        new = np.where(np.isfinite(r) & (r > 0), r ** 0.6, beta)   # beta = r^{3/5}
        if np.nanmax(np.abs(new - beta)) < 1e-12:
            beta = new; break
        beta = 0.5 * beta + 0.5 * new                              # damped
    alpha = np.interp(xg, xn, An) / f_Ap(beta ** (2.0 / 3.0) * xg)

    ok = np.isfinite(beta) & np.isfinite(alpha)
    xg, beta, alpha = xg[ok], beta[ok], alpha[ok]
    if len(xg) < 20:
        print(f"{q:5.2f}   too few valid points"); continue
    b_lo, b_hi = float(np.median(beta[:20])), float(np.median(beta[-20:]))
    a_lo = float(np.median(alpha[:20]))
    drift = (b_hi - b_lo) / b_lo * 100
    trend = "RISING" if drift > 0.05 else ("falling" if drift < -0.05 else "flat")
    print(f"{q:5.2f} {anchor:9.5f} | {b_lo:11.5f} {b_lo/anchor:7.4f} "
          f"| {b_hi:10.5f} {drift:+7.2f}% {trend:>7s} | {a_lo:12.5f} {a_lo/anchor:7.4f}")
    rows[q] = dict(x=xg.tolist(), beta=beta.tolist(), alpha=alpha.tolist(),
                   anchor=anchor, beta_lo=b_lo, beta_hi=b_hi, drift_pct=drift,
                   alpha_lo=a_lo, x_range=[float(xg[0]), float(xg[-1])])

print(f"\nx range used per q (capped at {X_MAX}, ISCO = {X_ISCO:.4f}):")
for q, r in rows.items():
    print(f"   q={q:g}: x in [{r['x_range'][0]:.4f}, {r['x_range'][1]:.4f}]")
json.dump({str(k): v for k, v in rows.items()},
          open(str(HERE / "alpha_beta_of_x.json"), "w"))
print("\nwrote alpha_beta_of_x.json")
