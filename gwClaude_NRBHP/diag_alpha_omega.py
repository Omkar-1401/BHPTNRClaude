"""
Diagnostic: is alpha smoother in omega space than in t space?

For q=5, compute the empirical alpha two ways:
  1. alpha_emp(t)   — amplitude ratio at each ppBHPT time t, using the optimal
                      time map tau(t) to evaluate NR.
  2. alpha_emp(omega) — same ratio, re-parameterised by the ppBHPT GW frequency
                        omega_GW(t) = d(arg h)/dt  (or x = (M omega_orb)^{2/3}).

Also show the direct frequency-domain measurement:
  alpha_direct(omega_NR) = A_NR(omega_NR) / A_pp(omega_NR * beta_PP)

using only a constant beta_PP — no time map, model-free.

Usage: conda run -n ut_claude python diag_alpha_omega.py [--q 5.0]
"""
from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import fit_scaling_PN_opt_creative_q_dep as qdep
import fit_scaling_gw_remnant_energy as G
import fit_scaling_PN_opt_creative as creative

T_ANCHOR = creative.T_ANCHOR   # typically -2000 M


def load_waveforms(q: float) -> dict:
    d = np.load(qdep.waveform_cache_path(q))
    return {
        "t_pp": d["t_bhpt"],
        "h_pp": d["h_bhpt_re"] + 1j * d["h_bhpt_im"],
        "t_nr": d["t_nr"],
        "h_nr": d["h_nr_re"]   + 1j * d["h_nr_im"],
        "t_pp_merger": float(d["t_bhpt_merger"]),
        "t_nr_merger": float(d["t_nr_merger"]),
    }


def inst_gw_freq(t: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Instantaneous GW frequency  omega_GW = d(arg h)/dt  [rad/M]."""
    # Use Im(conj(h) * dh/dt) / |h|^2; gradient for dh/dt.
    dh = np.gradient(h, t)
    amp2 = np.abs(h) ** 2
    amp2 = np.where(amp2 < 1e-30, 1e-30, amp2)   # avoid /0 near t=0 end
    return np.imag(np.conj(h) * dh) / amp2


def make_time_map(t_pp, beta_arr):
    """tau(t) = t0_nr + integral beta dt, anchored at T_ANCHOR."""
    beta_cum = np.concatenate([[0.0], np.cumsum(np.diff(t_pp) * 0.5 * (beta_arr[:-1] + beta_arr[1:]))])
    anchor_val = float(np.interp(T_ANCHOR, t_pp, beta_cum))
    return beta_cum - anchor_val   # relative to anchor; caller adds t0_nr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=5.0)
    args = ap.parse_args()
    q = args.q

    # ------------------------------------------------------------------ load
    wf = load_waveforms(q)
    t_pp, h_pp = wf["t_pp"], wf["h_pp"]
    t_nr, h_nr = wf["t_nr"], wf["h_nr"]

    # gw_remnant coordinates (E(t) on ppBHPT grid)
    coords = G.gwr_energy_coordinates(t_pp, h_pp, q)
    E = coords["e_oft"]   # raw cumulative radiated energy

    # Per-q optimal params from mult cache (form = "mult"):
    #   params = [alpha_PP, alpha_E, beta_PP, beta_E, t0_nr, phi0]
    import json
    cache = json.loads((ROOT / "gw_remnant_energy_results" / "per_q_cache_mult.json").read_text())
    key = min(cache.keys(), key=lambda k: abs(float(k) - q))
    p = cache[key]["params"]
    a_PP, a_E, b_PP, b_E, t0_nr, phi0 = p
    print(f"q={q}  key={key}  a_PP={a_PP:.4f} a_E={a_E:.4f}  b_PP={b_PP:.4f} b_E={b_E:.4f}")

    # ------------------------------------------------- model alpha and beta
    alpha_model = a_PP * (1.0 + a_E * E)
    beta_model  = b_PP * (1.0 + b_E * E)

    # Time map tau(t) = t0_nr + integrated beta, anchored
    rel_cum = make_time_map(t_pp, beta_model)
    tau = t0_nr + rel_cum

    # ----------------------------------------- empirical alpha via time map
    # alpha_emp(t) = |h_NR(tau(t))| / |h_pp(t)|
    A_nr_at_tau = np.interp(tau, t_nr, np.abs(h_nr))
    A_pp        = np.abs(h_pp)
    A_pp_safe   = np.where(A_pp < 1e-30 * A_pp.max(), np.nan, A_pp)
    alpha_emp   = A_nr_at_tau / A_pp_safe

    # Inspiral window: tau inside NR domain AND before ppBHPT amplitude peak
    i_merger_pp = int(np.argmax(np.abs(h_pp)))
    t_pp_merger = t_pp[i_merger_pp]
    valid = (tau >= t_nr[0]) & (tau <= t_nr[-1]) & (t_pp <= t_pp_merger)
    t_v      = t_pp[valid]
    alpha_v  = alpha_emp[valid]
    alpha_mv = alpha_model[valid]
    print(f"Valid ppBHPT window: {t_v[0]:.1f} to {t_v[-1]:.1f} M  ({valid.sum()} pts)")
    print(f"ppBHPT merger at t = {t_pp_merger:.1f} M")

    # ----------------------------------------- instantaneous GW frequencies
    omega_pp = inst_gw_freq(t_pp, h_pp)    # rad/M  (= 2 * omega_orb for (2,2) mode)
    omega_nr = inst_gw_freq(t_nr, h_nr)

    # x = (M * omega_orb)^{2/3} = (omega_GW / 2)^{2/3}
    x_pp = (0.5 * np.abs(omega_pp)) ** (2.0 / 3.0)
    x_nr = (0.5 * np.abs(omega_nr)) ** (2.0 / 3.0)

    x_v = x_pp[valid]

    # -------------------------------- direct frequency-domain alpha (model-free)
    # The paper (Eq. 17): h_NR(omega_NR) ≈ alpha * h_pp(omega_NR * beta_PP)
    # => alpha(omega_NR) = A_NR(omega_NR) / A_pp(omega_NR * beta_PP)
    # Map via x: omega_NR = omega_pp / beta_PP => x_NR = x_pp / beta_PP^{2/3}
    #
    # Use the inspiral-only portion of h_NR where x_NR is monotone increasing.
    # The NR merger is at the amplitude peak; use only data up to that.
    A_nr       = np.abs(h_nr)
    i_merger_nr = int(np.argmax(A_nr))
    x_nr_ins   = x_nr[:i_merger_nr]
    A_nr_ins   = A_nr[:i_merger_nr]

    # Sort by x (should already be monotone, but sort to be safe)
    sort_nr    = np.argsort(x_nr_ins)
    x_nr_s     = x_nr_ins[sort_nr]
    A_nr_s     = A_nr_ins[sort_nr]

    # For each ppBHPT point, evaluate A_NR at the corresponding x_NR
    x_nr_eval    = x_pp / b_PP ** (2.0 / 3.0)
    A_nr_interp  = np.interp(x_nr_eval, x_nr_s, A_nr_s, left=np.nan, right=np.nan)
    A_pp_safe    = np.where(A_pp < 1e-30 * A_pp.max(), np.nan, A_pp)
    alpha_direct = A_nr_interp / A_pp_safe

    alpha_dv = alpha_direct[valid]

    # ------------------------------------------------ polynomial smoothness test
    # Fit degree-2 and degree-3 polynomials to alpha_emp in two coordinates
    # and measure the residual. Lower residual = smoother/simpler curve.
    good = np.isfinite(alpha_v) & np.isfinite(alpha_dv)
    t_g   = t_v[good]
    x_g   = x_v[good]
    ae_g  = alpha_v[good]
    ad_g  = alpha_dv[good]

    def poly_rms(coord, vals, deg):
        c = np.polyfit(coord, vals, deg)
        resid = vals - np.polyval(c, coord)
        return float(np.std(resid)), c

    print("\n--- Polynomial fit residuals (std of alpha - poly fit) ---")
    print(f"{'deg':<5} {'alpha_emp(t)':>16} {'alpha_emp(x)':>16} {'alpha_direct(x)':>18}")
    for deg in (1, 2, 3, 4):
        r_t,  _ = poly_rms(t_g,  ae_g, deg)
        r_x,  _ = poly_rms(x_g,  ae_g, deg)
        r_dx, _ = poly_rms(x_g,  ad_g, deg)
        print(f"  {deg}   {r_t:16.4e}   {r_x:16.4e}   {r_dx:18.4e}")

    print(f"\nalpha_emp    range: {np.nanmin(ae_g):.4f}  to  {np.nanmax(ae_g):.4f}")
    print(f"alpha_direct range: {np.nanmin(ad_g):.4f}  to  {np.nanmax(ad_g):.4f}")

    # Best-fit lines for plot overlays
    _, c_t  = poly_rms(t_g, ae_g, 3)
    _, c_xd = poly_rms(x_g, ad_g, 3)

    # ---------------------------------------------------------------- plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"Alpha smoothness: time domain vs frequency domain  (q = {q}, inspiral only)", fontsize=11)

    # Panel 1: empirical alpha vs t
    ax = axes[0]
    ax.plot(t_g, ae_g, lw=0.6, color="C0", alpha=0.6, label="empirical (tau map)")
    ax.plot(t_g, np.polyval(c_t, t_g), lw=1.5, color="C1", ls="--", label="deg-3 poly in t")
    ax.set_xlabel("t_ppBHPT  [M]")
    ax.set_ylabel("alpha")
    ax.set_title("alpha vs t")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: same alpha_emp re-parameterised by x
    ax = axes[1]
    ax.plot(x_g, ae_g, lw=0.6, color="C0", alpha=0.6, label="empirical (tau map)")
    _, c_xe = poly_rms(x_g, ae_g, 3)
    ax.plot(x_g, np.polyval(c_xe, x_g), lw=1.5, color="C1", ls="--", label="deg-3 poly in x")
    ax.set_xlabel(r"$x = (M\,\omega_\mathrm{orb})^{2/3}$")
    ax.set_ylabel("alpha")
    ax.set_title(r"alpha vs $x$ (same values, different coord)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: direct frequency-domain alpha (model-free, no tau)
    ax = axes[2]
    ax.plot(x_g, ad_g, lw=0.6, color="C2", alpha=0.6,
            label=r"$A_\mathrm{NR}(\omega_\mathrm{NR})\,/\,A_\mathrm{pp}(\omega_\mathrm{NR}\cdot\beta)$")
    ax.plot(x_g, np.polyval(c_xd, x_g), lw=1.5, color="C3", ls="--", label="deg-3 poly in x")
    ax.set_xlabel(r"$x = (M\,\omega_\mathrm{orb})^{2/3}$")
    ax.set_ylabel("alpha")
    ax.set_title(r"$\alpha(\omega)$ direct (no time map)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = ROOT / f"diag_alpha_omega_q{q}.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
