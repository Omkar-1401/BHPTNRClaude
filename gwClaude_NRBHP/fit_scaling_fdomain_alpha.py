"""
fit_scaling_fdomain_alpha.py

Frequency-domain alpha calibration — full regression pipeline.

Model
-----
    alpha(t, q) = a_PP(q) * g(z(t), q)

    x(t)    = (omega_gw(t)/2)^{2/3} = (pi*f_gw(t))^{2/3}
              instantaneous SPA coordinate, from -d(arg h_pp)/dt

    x_clip  = clip(x(t), X_LO, X_HI)   where [X_LO, X_HI] is the inspiral band
    z(t)    = (x_clip - X_MID) / X_SCALE   in [-1, 1]

    g(z, q) = c4(q)*z^4 + c3(q)*z^3 + c2(q)*z^2 + c1(q)*z + c0(q)   deg-4 in z

Using z instead of x:
    x is only fit over [X_LO, X_HI] ~ [0.079, 0.168].  Raw polynomial in x has
    coefficients O(10^4) that partially cancel; outside the band they blow up.
    Normalising to z in [-1, 1] gives O(1) coefficients that regress stably across q
    and are clipped to evaluate only within the trained range.

Nu-regression (nu = q/(1+q)^2)
    a_PP(nu) = 1 + p1*nu + p2*nu^2 + p3*nu^3          3 free  [PP-anchored at 1]
    ck(nu)   = dk0 + dk1*nu + dk2*nu^2   k=0..4        3 free each = 15
                                               Total:  18 regression coefficients

Beta is unchanged from the per-q gwr_energy_mult model (b_PP, b_E, t0_nr from cache).
For held-out q < 3, per-q mult params are re-fitted with G.optimize_case.

Training: all 64 q in per_q_cache_mult.json  (q in [3, 8])
Held-out: 2.75, 2.5, 2.25, 2.0

Usage:
    conda run -n ut_claude python fit_scaling_fdomain_alpha.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal.windows import tukey

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import fit_scaling_PN_opt_creative as creative
import fit_scaling_PN_opt_creative_q_dep as qdep
import fit_scaling_gw_remnant_energy as G

T_ANCHOR    = creative.T_ANCHOR
RESULTS_DIR = ROOT / "fdomain_alpha_results"
RESULTS_DIR.mkdir(exist_ok=True)
COEFFS_PATH = RESULTS_DIR / "coeffs.json"
MULT_CACHE  = G.RESULTS_DIR / "per_q_cache_mult.json"

LOW_Q    = (2.75, 2.5, 2.25, 2.0)
IN_Q     = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
BAND     = (0.004, 0.025)    # requested inspiral band [cycles/M]
N_SMOOTH = 30
G_DEG    = 4                 # degree of g(z) polynomial
A_DEG    = 3                 # a_PP(nu): 1 + p1*nu + ... + p_{A_DEG}*nu^{A_DEG}
C_DEG    = 2                 # ck(nu):   dk0 + dk1*nu + dk2*nu^{C_DEG}

# Fixed normalisation constants for z = (x - X_MID) / X_SCALE.
# These correspond to the empirical band edges after edge-trimming for a ~5000 M
# window (edge = N_SMOOTH//2 = 15 bins, df ~ 2e-4 M^{-1}):
#   f_eff_lo ~ 0.004 + 15*2e-4 = 0.007,  x_lo = (pi*0.007)^{2/3} ~ 0.079
#   f_eff_hi ~ 0.025 - 15*2e-4 = 0.022,  x_hi = (pi*0.022)^{2/3} ~ 0.168
X_LO   = (np.pi * 0.007) ** (2.0 / 3.0)    # ~0.079
X_HI   = (np.pi * 0.022) ** (2.0 / 3.0)    # ~0.168
X_MID  = 0.5 * (X_LO + X_HI)
X_SCALE = 0.5 * (X_HI - X_LO)

nu_of = lambda q: float(q) / (1.0 + float(q))**2


def x_to_z(x):
    return (np.clip(x, X_LO, X_HI) - X_MID) / X_SCALE


# ---------------------------------------------------------------------------
# FFT helpers
# ---------------------------------------------------------------------------

def windowed_fft(t, h, tukey_alpha=0.05):
    """h_22 = A exp(-iΦ): signal at negative f.  Return |f|, H."""
    dt  = float(np.median(np.diff(t)))
    win = tukey(len(h), alpha=tukey_alpha)
    H   = fft(h * win) * dt
    f   = fftfreq(len(h), d=dt)
    neg = f < 0
    return -f[neg][::-1], H[neg][::-1]


def smooth_band(H_nr, H_pp, n, f_all):
    """Smooth power spectra inside BAND, trim n//2 edge bins."""
    lo, hi = BAND
    mask   = (f_all > lo) & (f_all < hi)
    f_raw  = f_all[mask]
    P_nr   = np.convolve(np.abs(H_nr[mask])**2, np.ones(n)/n, mode="same")
    P_pp   = np.convolve(np.abs(H_pp[mask])**2, np.ones(n)/n, mode="same")
    edge   = n // 2
    return f_raw[edge:-edge], P_nr[edge:-edge], P_pp[edge:-edge]


def instantaneous_x(t_pp, h_pp):
    """x(t) = (pi f_gw)^{2/3} from phase derivative of h_ppBHPT."""
    phase = np.unwrap(np.angle(h_pp))
    omega = -np.gradient(phase, t_pp)   # h = A exp(-iΦ) => d(arg)/dt = -omega
    omega = np.maximum(omega, 1e-8)
    return (0.5 * omega) ** (2.0 / 3.0)


# ---------------------------------------------------------------------------
# Time map
# ---------------------------------------------------------------------------

def build_tau(t_pp, h_pp, q, params):
    b_PP, b_E, t0_nr = params[2], params[3], params[4]
    E      = G.gwr_energy_coordinates(t_pp, h_pp, q)["e_oft"]
    beta_t = b_PP * (1.0 + b_E * E)
    bc     = np.concatenate([[0.], np.cumsum(np.diff(t_pp)*0.5*(beta_t[:-1]+beta_t[1:]))])
    anchor = float(np.interp(T_ANCHOR, t_pp, bc))
    return t0_nr + bc - anchor


def map_pp_to_nr(t_pp, h_pp, t_nr, tau):
    valid   = (tau >= t_nr[0]) & (tau <= t_nr[-1])
    tv, tp  = tau[valid], t_pp[valid]
    lo, hi  = tv[0], tv[-1]
    mask    = (t_nr >= lo) & (t_nr <= hi)
    t_pp_on = np.interp(t_nr[mask], tv, tp)
    h_pp_on = (np.interp(t_pp_on, t_pp, h_pp.real)
             + 1j * np.interp(t_pp_on, t_pp, h_pp.imag))
    return t_nr[mask], h_pp_on, mask


# ---------------------------------------------------------------------------
# Per-q shape polynomial (in z)
# ---------------------------------------------------------------------------

def compute_per_q_shape(q, params):
    """
    Compute per-q (a_PP, nu, c_poly_z) where c_poly_z is the deg-G_DEG polynomial
    in z = (x - X_MID) / X_SCALE that fits alpha_f / a_PP in the inspiral band.
    Returns (a_PP, nu, c_poly_z, rms).
    """
    a_PP = float(params[0])
    nu   = nu_of(q)

    d    = np.load(qdep.waveform_cache_path(q))
    t_pp = d["t_bhpt"];  h_pp = d["h_bhpt_re"] + 1j*d["h_bhpt_im"]
    t_nr = d["t_nr"];    h_nr = d["h_nr_re"]   + 1j*d["h_nr_im"]

    tau              = build_tau(t_pp, h_pp, q, params)
    t_grid, h_pp_on, mask = map_pp_to_nr(t_pp, h_pp, t_nr, tau)
    h_nr_cut         = h_nr[mask]

    f_all, H_pp = windowed_fft(t_grid, h_pp_on)
    _,     H_nr = windowed_fft(t_grid, h_nr_cut)

    f_band, P_nr, P_pp = smooth_band(H_nr, H_pp, N_SMOOTH, f_all)
    x_band  = (np.pi * f_band) ** (2.0 / 3.0)
    z_band  = x_to_z(x_band)                      # z in [-1, 1], clipping not needed here
    alpha_f = np.sqrt(P_nr / P_pp) / a_PP

    c_poly_z = np.polyfit(z_band, alpha_f, G_DEG)
    rms      = float(np.std(alpha_f - np.polyval(c_poly_z, z_band)))
    return a_PP, nu, c_poly_z, rms


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def regress(nus, a_PPs, c_polys_z):
    """
    Regress a_PP(nu) and each z-polynomial coefficient ck(nu) across q.

    a_PP(nu) = 1 + p1*nu + ... + p_{A_DEG}*nu^{A_DEG}   [PP-anchored]
    ck(nu)   = dk0 + dk1*nu + ... + dk_{C_DEG}*nu^{C_DEG}  k=0..G_DEG

    c_polys_z has shape (n_q, G_DEG+1); highest degree first (np.polyfit convention).

    Returns
        a_PP_coeffs : shape (A_DEG,)   [p1, p2, p3]
        g_coeffs    : shape (C_DEG+1, G_DEG+1)
                      g_coeffs[:, j] = [dk0, dk1, dk2] for poly-coefficient cj
                      (j=0 -> highest degree c4, j=G_DEG -> c0)
    """
    nus       = np.array(nus)
    a_PPs     = np.array(a_PPs)
    c_mat     = np.array(c_polys_z)          # (n_q, G_DEG+1)

    # a_PP: anchored at 1
    X_a = np.column_stack([nus**(k+1) for k in range(A_DEG)])
    a_PP_coeffs, _, _, _ = np.linalg.lstsq(X_a, a_PPs - 1.0, rcond=None)

    # ck: free polynomial in nu
    X_c = np.column_stack([nus**k for k in range(C_DEG+1)])
    g_coeffs, _, _, _ = np.linalg.lstsq(X_c, c_mat, rcond=None)   # (C_DEG+1, G_DEG+1)

    return a_PP_coeffs, g_coeffs


def eval_a_PP(nu, a_PP_coeffs):
    return 1.0 + sum(a_PP_coeffs[k] * nu**(k+1) for k in range(A_DEG))


def eval_c_poly_z(nu, g_coeffs):
    """Return [c4_hat, c3_hat, ..., c0_hat] (highest degree first) for given nu."""
    nu_vec = np.array([nu**k for k in range(C_DEG+1)])
    return g_coeffs.T @ nu_vec     # (G_DEG+1,)


# ---------------------------------------------------------------------------
# Mismatch
# ---------------------------------------------------------------------------

def mismatch_fdomain(q, nu, a_PP_coeffs, g_coeffs, beta_params):
    """
    Time-domain mismatch with
        alpha(t) = eval_a_PP(nu) * polyval(eval_c_poly_z(nu), z(t))
        z(t)     = (clip(x(t), X_LO, X_HI) - X_MID) / X_SCALE
        beta     from beta_params = [b_PP, b_E, t0_nr]
    phi0 solved analytically.
    """
    b_PP, b_E, t0_nr = float(beta_params[0]), float(beta_params[1]), float(beta_params[2])
    a_PP_hat = eval_a_PP(nu, a_PP_coeffs)
    c_hat    = eval_c_poly_z(nu, g_coeffs)

    d     = np.load(qdep.waveform_cache_path(q))
    t_pp  = d["t_bhpt"];  h_pp = d["h_bhpt_re"] + 1j*d["h_bhpt_im"]
    t_nr  = d["t_nr"];    h_nr = d["h_nr_re"]   + 1j*d["h_nr_im"]

    E      = G.gwr_energy_coordinates(t_pp, h_pp, q)["e_oft"]
    beta_t = b_PP * (1.0 + b_E * E)
    if np.any(beta_t <= 0):
        return 1.0
    bc     = np.concatenate([[0.], np.cumsum(np.diff(t_pp)*0.5*(beta_t[:-1]+beta_t[1:]))])
    anchor = float(np.interp(T_ANCHOR, t_pp, bc))
    tau    = t0_nr + bc - anchor

    if not np.all(np.diff(tau) > 0):
        return 1.0

    # alpha(t_pp): x clipped to band, then z, then polynomial
    x_t     = instantaneous_x(t_pp, h_pp)
    z_t     = x_to_z(x_t)                    # clips x to [X_LO, X_HI], normalises
    alpha_t = a_PP_hat * np.polyval(c_hat, z_t)

    t_min = max(t_nr[0], tau[0])
    t_max = min(t_nr[-1], tau[-1])
    mask  = (t_nr >= t_min) & (t_nr <= t_max)
    if mask.sum() < 100:
        return 1.0

    t_c   = t_nr[mask]
    h_ref = h_nr[mask]

    h_pp_r  = np.interp(t_c, tau, h_pp.real)
    h_pp_i  = np.interp(t_c, tau, h_pp.imag)
    alpha_c = np.interp(t_c, tau, alpha_t)
    h0      = alpha_c * (h_pp_r + 1j * h_pp_i)

    z_val = np.sum(h_ref * h0.conjugate())
    sd    = float(np.abs(z_val))
    n1    = float(np.sum(np.abs(h_ref)**2))
    n2    = float(np.sum(np.abs(h0)**2))
    return float(((n1 + n2) - 2.0*sd) / (2.0*n1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cache      = json.loads(MULT_CACHE.read_text())
    training_q = sorted(float(k) for k in cache.keys())
    print(f"Training on {len(training_q)} q values in [{training_q[0]:.3f}, {training_q[-1]:.3f}]")
    print(f"Band z-normalisation: X_LO={X_LO:.4f}  X_HI={X_HI:.4f}  "
          f"X_MID={X_MID:.4f}  X_SCALE={X_SCALE:.4f}")

    # ------------------------------------------------------------------
    # Step 1: per-q FFT shape polynomials (in z)
    # ------------------------------------------------------------------
    print("\nStep 1: per-q alpha(f)/a_PP polynomial in z ...")
    nus, a_PPs, c_polys_z = [], [], []
    for i, q in enumerate(training_q):
        key    = min(cache.keys(), key=lambda k: abs(float(k)-q))
        params = cache[key]["params"]
        a_PP, nu, c_poly_z, rms = compute_per_q_shape(q, params)
        nus.append(nu); a_PPs.append(a_PP); c_polys_z.append(c_poly_z)
        print(f"  [{i+1:2d}/{len(training_q)}]  q={q:.4f}  a_PP={a_PP:.4f}  "
              f"c0(z=0)={c_poly_z[-1]:.4f}  rms={rms:.3e}", flush=True)

    nus       = np.array(nus)
    a_PPs     = np.array(a_PPs)
    c_polys_z = np.array(c_polys_z)    # (64, 5)

    # ------------------------------------------------------------------
    # Step 2: regression
    # ------------------------------------------------------------------
    print("\nStep 2: regression across nu ...")
    a_PP_coeffs, g_coeffs = regress(nus, a_PPs, c_polys_z)

    print(f"  a_PP(nu) = 1 + {a_PP_coeffs[0]:.4f}*nu "
          f"+ {a_PP_coeffs[1]:.4f}*nu^2 + {a_PP_coeffs[2]:.4f}*nu^3")
    labels = [f"c{G_DEG-j}" for j in range(G_DEG+1)]
    for j, lab in enumerate(labels):
        row = g_coeffs[:, j]
        print(f"  {lab}(nu) = {row[0]:.5f} + {row[1]:.5f}*nu + {row[2]:.5f}*nu^2   "
              f"rms={np.std(c_polys_z[:,j] - (np.column_stack([nus**k for k in range(C_DEG+1)]) @ g_coeffs[:,j])):.4e}")

    # ------------------------------------------------------------------
    # Step 3: training-q mismatch
    # ------------------------------------------------------------------
    print("\nStep 3: mismatch at training q ...")
    all_train = {}
    for q in training_q:
        nu     = nu_of(q)
        key    = min(cache.keys(), key=lambda k: abs(float(k)-q))
        params = cache[key]["params"]
        beta   = (params[2], params[3], params[4])
        E      = mismatch_fdomain(q, nu, a_PP_coeffs, g_coeffs, beta)
        all_train[q] = E

    in_range_errors = {q: all_train[q] for q in IN_Q}
    errs = list(all_train.values())
    print(f"  In-range  median={np.median(errs):.4e}  max={np.max(errs):.4e}")
    for q in IN_Q:
        print(f"  q={q:.2f}  E={in_range_errors[q]:.4e}")

    # ------------------------------------------------------------------
    # Step 4: held-out q < 3
    # ------------------------------------------------------------------
    print("\nStep 4: held-out q < 3 ...")
    low_q_errors = {}
    for q in LOW_Q:
        nu = nu_of(q)
        print(f"  q={q}  fitting per-q mult beta ...", end=" ", flush=True)
        result = G.optimize_case(q, form="mult")
        params = result["params"]
        beta   = (params[2], params[3], params[4])
        E      = mismatch_fdomain(q, nu, a_PP_coeffs, g_coeffs, beta)
        low_q_errors[q] = E
        print(f"E={E:.4e}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "="*55)
    print("SUMMARY")
    print("="*55)
    print(f"  In-range [3,8]  median  {np.median(errs):.4e}")
    print(f"  In-range [3,8]  max     {np.max(errs):.4e}")
    for q in IN_Q:
        print(f"  q={q:.1f}  {in_range_errors[q]:.4e}")
    print()
    for q in (2.75, 2.5, 2.25, 2.0):
        print(f"  q={q}  {low_q_errors[q]:.4e}")
    print("="*55)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out = {
        "description":  "fdomain alpha: a_PP(nu)*g(z(t),nu), z=(clip(x,X_LO,X_HI)-X_MID)/X_SCALE",
        "X_LO": X_LO, "X_HI": X_HI, "X_MID": X_MID, "X_SCALE": X_SCALE,
        "G_DEG": G_DEG, "A_DEG": A_DEG, "C_DEG": C_DEG,
        "n_regression_coeffs": A_DEG + (C_DEG+1)*(G_DEG+1),
        "a_PP_coeffs": a_PP_coeffs.tolist(),
        "g_coeffs":    g_coeffs.tolist(),
        "in_range": {"median": float(np.median(errs)), "max": float(np.max(errs)),
                     "by_q": {f"{q:.4f}": v for q, v in all_train.items()}},
        "low_q": {f"{q:.2f}": v for q, v in low_q_errors.items()},
    }
    COEFFS_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {COEFFS_PATH}")


if __name__ == "__main__":
    main()
