"""
Frequency-domain alpha diagnostic.

Computes the Fourier transforms of h_NR and h_ppBHPT, then extracts alpha(f)
directly as the amplitude ratio in the Fourier domain.

The time-domain relation  h_NR(t_NR) = alpha * h_pp(t_pp),  t_NR = beta * t_pp
implies in the Fourier domain (constant alpha, beta):

    H_NR(f) = alpha * beta * H_pp(beta * f)
 =>  alpha(f) = |H_NR(f)| / (beta * |H_pp(beta * f)|)

For time-varying alpha(t), this ratio is frequency-dependent and we want to know
whether it is a smooth function of f (or x = (pi*M*f_GW)^{2/3}).

Usage:  conda run -n ut_claude python diag_alpha_fdomain.py [--q 5.0]
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal.windows import tukey
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import fit_scaling_PN_opt_creative_q_dep as qdep
import fit_scaling_PN_opt_creative as creative

T_ANCHOR = creative.T_ANCHOR


def load_waveforms(q):
    d = np.load(qdep.waveform_cache_path(q))
    return {
        "t_pp":  d["t_bhpt"],
        "h_pp":  d["h_bhpt_re"] + 1j * d["h_bhpt_im"],
        "t_nr":  d["t_nr"],
        "h_nr":  d["h_nr_re"]   + 1j * d["h_nr_im"],
        "t_pp_merger": float(d["t_bhpt_merger"]),
        "t_nr_merger": float(d["t_nr_merger"]),
    }


def windowed_fft(t, h, tukey_alpha=0.05):
    """Apply Tukey window and return the physical frequency slice (f [cycles/M], H_tilde).

    h_22 uses the PN convention  h = A exp(-iΦ)  with Φ̇ > 0, so the FFT signal
    lives at NEGATIVE frequencies (f = -f_GW < 0).  We take the negative-f half
    and return |f| (so callers see positive apparent frequencies, just as with a
    real-valued FFT).  The positive-f half contains only window-leakage and is
    discarded.
    """
    dt  = float(np.median(np.diff(t)))
    N   = len(h)
    win = tukey(N, alpha=tukey_alpha)
    H_full = fft(h * win) * dt      # complex, two-sided spectrum
    f_full = fftfreq(N, d=dt)       # signed frequencies, cycles/M
    # Signal at f < 0.  Flip to positive apparent frequency.
    neg = f_full < 0
    return -f_full[neg][::-1], H_full[neg][::-1]


def poly_rms(coord, vals, deg):
    c = np.polyfit(coord, vals, deg)
    return float(np.std(vals - np.polyval(c, coord))), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=5.0)
    args = ap.parse_args()
    q = args.q

    # ------------------------------------------------------------------ load
    wf = load_waveforms(q)
    t_pp, h_pp = wf["t_pp"], wf["h_pp"]
    t_nr, h_nr = wf["t_nr"], wf["h_nr"]

    # Per-q optimal beta_PP from mult cache
    cache = json.loads((ROOT / "gw_remnant_energy_results" / "per_q_cache_mult.json").read_text())
    key   = min(cache.keys(), key=lambda k: abs(float(k) - q))
    p     = cache[key]["params"]
    a_PP, _, b_PP, _, _, _ = p
    print(f"q={q}  a_PP={a_PP:.4f}  b_PP={b_PP:.4f}")

    # ------------------------------------------------------------------ FFTs
    # Strategy: compute the full varying-β time map tau(t_pp) using the optimal
    # per-q parameters, then invert it to get t_pp(t_NR).  Interpolate h_pp at
    # those ppBHPT times so both waveforms sit on the NR time grid and their
    # FFTs have the same frequency axis.
    import fit_scaling_gw_remnant_energy as G

    t0_nr  = cache[key]["params"][4]
    a_E_p  = cache[key]["params"][1]
    b_E_p  = cache[key]["params"][3]

    coords = G.gwr_energy_coordinates(t_pp, h_pp, q)
    E      = coords["e_oft"]
    beta_t = b_PP * (1.0 + b_E_p * E)    # full varying beta
    alpha_t = a_PP * (1.0 + a_E_p * E)   # full varying alpha (for reference)

    # tau(t_pp) via cumulative trapezoid
    beta_cum   = np.concatenate([[0.0], np.cumsum(
        np.diff(t_pp) * 0.5 * (beta_t[:-1] + beta_t[1:]))])
    anchor_val = float(np.interp(T_ANCHOR, t_pp, beta_cum))
    tau        = t0_nr + beta_cum - anchor_val   # NR time as function of t_pp

    # Invert: for each NR time, find corresponding ppBHPT time.
    # tau(t_pp) is monotone; valid_tau selects ppBHPT times whose mapped NR time
    # is inside the NR window.  The NR times ACTUALLY covered are tau[valid_tau][0]
    # through tau[valid_tau][-1].  We must restrict to that range — np.interp
    # otherwise clamps out-of-range NR times to the boundary t_pp value, creating
    # a spurious constant segment in h_pp_on_nr.
    valid_tau  = (tau >= t_nr[0]) & (tau <= t_nr[-1])
    t_pp_valid = t_pp[valid_tau]
    tau_valid  = tau[valid_tau]

    tau_lo, tau_hi = tau_valid[0], tau_valid[-1]   # NR-time range with valid ppBHPT
    common = (t_nr >= tau_lo) & (t_nr <= tau_hi)   # strict: no clamped boundary

    t_pp_of_nr = np.interp(t_nr[common], tau_valid, t_pp_valid)   # now all in-range

    t_grid     = t_nr[common]
    h_nr_cut   = h_nr[common]
    h_pp_on_nr = (np.interp(t_pp_of_nr, t_pp, h_pp.real)
                + 1j * np.interp(t_pp_of_nr, t_pp, h_pp.imag))

    # Also grab the time-domain alpha at those ppBHPT times (for comparison)
    alpha_t_on_nr = np.interp(t_pp_of_nr, t_pp, alpha_t)

    print(f"Common grid: t_NR ∈ [{t_grid[0]:.1f}, {t_grid[-1]:.1f}]  "
          f"({common.sum()} pts, dt={t_grid[1]-t_grid[0]:.3f} M)")

    # Both waveforms now on the same time grid → FFT frequency axes match.
    f_all, H_pp = windowed_fft(t_grid, h_pp_on_nr)
    _,     H_nr = windowed_fft(t_grid, h_nr_cut)

    # Spectral smoothing: the finite NR window (~5000 M) creates Fresnel oscillations
    # of period 1/T ~ 2e-4 cycles/M, roughly 1 FFT bin (df = 1.96e-4 M^{-1}).
    # Smooth the POWER spectra within the inspiral band to kill these fringes.
    # N_smooth ~ 30 bins spans ~30 oscillation periods (>> 1) while leaving enough
    # band to fit polynomials.  Edge-trim within the band, not the full FFT array.
    N_smooth = 30

    def smooth(x, n):
        return np.convolve(x, np.ones(n) / n, mode="same")

    # Cut to inspiral band FIRST (avoid DC-boundary contamination of the smooth)
    f_lo, f_hi = 0.004, 0.025
    band_full  = (f_all > f_lo) & (f_all < f_hi)
    f_raw      = f_all[band_full]
    P_nr_raw   = np.abs(H_nr[band_full]) ** 2
    P_pp_raw   = np.abs(H_pp[band_full]) ** 2

    P_nr_s = smooth(P_nr_raw, N_smooth)
    P_pp_s = smooth(P_pp_raw, N_smooth)

    # Trim convolution edge artefacts within the band
    edge   = N_smooth // 2
    f_band = f_raw[edge:-edge]
    alpha_f = np.sqrt(P_nr_s[edge:-edge] / P_pp_s[edge:-edge]) / a_PP

    # x coordinate: x = (pi * f_GW)^{2/3}
    x_band = (np.pi * f_band) ** (2.0 / 3.0)

    # ------------------------------------------ polynomial smoothness summary
    print(f"\n--- Polynomial fit residuals (N_smooth={N_smooth} bins) ---")
    print(f"{'deg':<5} {'poly in f':>14} {'poly in x':>14}")
    for deg in (1, 2, 3, 4):
        r_f, _ = poly_rms(f_band, alpha_f, deg)
        r_x, _ = poly_rms(x_band, alpha_f, deg)
        print(f"  {deg}   {r_f:14.4e}   {r_x:14.4e}")

    print(f"\nalpha_f range: {alpha_f.min():.4f}  to  {alpha_f.max():.4f}  "
          f"(mean {alpha_f.mean():.4f}, a_PP={a_PP:.4f})")
    print(f"x     range:   {x_band.min():.4f}  to  {x_band.max():.4f}")

    # ------------------------------------------------------------------ plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"Frequency-domain alpha   (q = {q})", fontsize=11)

    # Panel 1: smoothed power spectra within the inspiral band
    ax = axes[0]
    ax.plot(f_raw, np.sqrt(P_nr_raw), lw=0.5, color="C0", alpha=0.3)
    ax.plot(f_raw, np.sqrt(P_pp_raw) * a_PP, lw=0.5, color="C1", alpha=0.3)
    ax.plot(f_raw[edge:-edge], np.sqrt(P_nr_s[edge:-edge]),
            lw=1.2, color="C0", label=r"$\sqrt{\langle P_{NR}\rangle}$")
    ax.plot(f_raw[edge:-edge], a_PP * np.sqrt(P_pp_s[edge:-edge]),
            lw=1.2, color="C1", ls="--",
            label=r"$\alpha_{PP}\sqrt{\langle P_{pp}\rangle}$ (mapped)")
    ax.set_xlabel("f  [cycles/M]")
    ax.set_ylabel(r"smoothed $|\tilde{h}|$")
    ax.set_title(f"Smoothed FFT amplitudes  (N_smooth={N_smooth} bins)")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    # Panel 2: alpha(f) vs f
    _, c_f = poly_rms(f_band, alpha_f, 3)
    ax = axes[1]
    ax.plot(f_band, alpha_f, lw=1.0, color="C0", label=r"$\alpha(f)$  [smoothed ratio]")
    ax.plot(f_band, np.polyval(c_f, f_band), lw=1.5, color="C1", ls="--",
            label="deg-3 poly in f")
    ax.axhline(1.0, color="C3", lw=1.0, ls=":", label="1.0  (constant α = a_PP)")
    ax.set_xlabel("f  [cycles/M]")
    ax.set_ylabel(r"$\alpha(f)\,/\,\alpha_{PP}$")
    ax.set_title(r"$\alpha(f) = \sqrt{\langle P_{NR}\rangle / \langle P_{pp}\rangle}\,/\,\alpha_{PP}$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: alpha vs x = (pi*f)^{2/3}
    _, c_x = poly_rms(x_band, alpha_f, 3)
    ax = axes[2]
    ax.plot(x_band, alpha_f, lw=1.0, color="C0", label=r"$\alpha(x)$")
    ax.plot(x_band, np.polyval(c_x, x_band), lw=1.5, color="C1", ls="--",
            label="deg-3 poly in x")
    ax.axhline(1.0, color="C3", lw=1.0, ls=":", label="1.0")
    ax.set_xlabel(r"$x = (\pi M f_\mathrm{GW})^{2/3}$")
    ax.set_ylabel(r"$\alpha(x)\,/\,\alpha_{PP}$")
    ax.set_title(r"$\alpha$ vs $x$  (SPA frequency coordinate)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = ROOT / f"diag_alpha_fdomain_q{q}.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
