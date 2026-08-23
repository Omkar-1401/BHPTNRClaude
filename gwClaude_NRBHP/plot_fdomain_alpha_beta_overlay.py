"""
Overlay alpha(f) and cross-phase(f) for multiple q values.

Panel 1: alpha(f)/a_PP = sqrt(P_NR/P_pp)/a_PP  -- the S-curve
Panel 2: cross-spectrum phase = angle(H_NR * conj(H_pp_on_nr)) - per-q mean
         (constant phi0 subtracted; residual slope/shape encodes beta accuracy)

Per-q mult params from gw_remnant_energy_results/per_q_cache_mult.json.
Q values come from --q arg (default: 3 4 5 6 7 8).

Usage:
  conda run -n ut_claude python plot_fdomain_alpha_beta_overlay.py
  conda run -n ut_claude python plot_fdomain_alpha_beta_overlay.py --q 2 3 4 5 6 7 8
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal.windows import tukey
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import fit_scaling_PN_opt_creative_q_dep as qdep
import fit_scaling_PN_opt_creative       as creative
import fit_scaling_gw_remnant_energy     as G

T_ANCHOR = creative.T_ANCHOR
N_SMOOTH = 30
F_LO, F_HI = 0.004, 0.025


def load_waveforms(q):
    d = np.load(qdep.waveform_cache_path(q))
    return {
        "t_pp": d["t_bhpt"],
        "h_pp": d["h_bhpt_re"] + 1j * d["h_bhpt_im"],
        "t_nr": d["t_nr"],
        "h_nr": d["h_nr_re"]   + 1j * d["h_nr_im"],
    }


def windowed_fft(t, h, tukey_alpha=0.05):
    dt = float(np.median(np.diff(t)))
    win = tukey(len(h), alpha=tukey_alpha)
    H_full = fft(h * win) * dt
    f_full = fftfreq(len(h), d=dt)
    neg = f_full < 0
    return -f_full[neg][::-1], H_full[neg][::-1]


def smooth(x, n):
    return np.convolve(x, np.ones(n) / n, mode="same")


def build_time_map(t_pp, h_pp, b_PP, b_E, E, t0_nr):
    beta_t   = b_PP * (1.0 + b_E * E)
    beta_cum = np.concatenate([[0.0], np.cumsum(
        np.diff(t_pp) * 0.5 * (beta_t[:-1] + beta_t[1:]))])
    anchor_val = float(np.interp(T_ANCHOR, t_pp, beta_cum))
    return t0_nr + beta_cum - anchor_val   # tau(t_pp)


def compute_spectra(q, params):
    """
    Returns dict with:
      f_band    : frequency axis in the inspiral band [cycles/M]
      alpha_f   : alpha(f)/a_PP (smoothed amplitude ratio)
      x_band    : (pi*f)^{2/3} frequency coordinate
      cross_phase: angle(H_NR * conj(H_pp_on_nr)), per-q mean subtracted
    """
    a_PP, a_E, b_PP, b_E, t0_nr, _ = params

    wf = load_waveforms(q)
    t_pp, h_pp = wf["t_pp"], wf["h_pp"]
    t_nr, h_nr = wf["t_nr"], wf["h_nr"]

    coords = G.gwr_energy_coordinates(t_pp, h_pp, q)
    E      = coords["e_oft"]

    tau = build_time_map(t_pp, h_pp, b_PP, b_E, E, t0_nr)

    valid_tau  = (tau >= t_nr[0]) & (tau <= t_nr[-1])
    tau_valid  = tau[valid_tau]
    t_pp_valid = t_pp[valid_tau]
    tau_lo, tau_hi = tau_valid[0], tau_valid[-1]
    common = (t_nr >= tau_lo) & (t_nr <= tau_hi)

    t_pp_of_nr = np.interp(t_nr[common], tau_valid, t_pp_valid)
    t_grid     = t_nr[common]
    h_nr_cut   = h_nr[common]
    h_pp_on_nr = (np.interp(t_pp_of_nr, t_pp, h_pp.real)
                + 1j * np.interp(t_pp_of_nr, t_pp, h_pp.imag))

    f_all, H_pp = windowed_fft(t_grid, h_pp_on_nr)
    _,     H_nr = windowed_fft(t_grid, h_nr_cut)

    band_full = (f_all > F_LO) & (f_all < F_HI)
    f_raw     = f_all[band_full]
    P_nr_raw  = np.abs(H_nr[band_full]) ** 2
    P_pp_raw  = np.abs(H_pp[band_full]) ** 2

    P_nr_s = smooth(P_nr_raw, N_SMOOTH)
    P_pp_s = smooth(P_pp_raw, N_SMOOTH)

    edge    = N_SMOOTH // 2
    f_band  = f_raw[edge:-edge]
    x_band  = (np.pi * f_band) ** (2.0 / 3.0)
    alpha_f = np.sqrt(P_nr_s[edge:-edge] / P_pp_s[edge:-edge]) / a_PP

    # cross-spectrum phase: angle(H_NR * conj(H_pp_on_nr))
    # Smooth the COMPLEX cross-spectrum before taking angle() so Fresnel fringes
    # cancel in the complex plane rather than aliasing into the phase estimate.
    # Mean-subtract to remove phi0 (constant offset); residual shape = beta error.
    cross_full  = H_nr[band_full] * np.conj(H_pp[band_full])
    cross_re_s  = smooth(cross_full.real, N_SMOOTH)
    cross_im_s  = smooth(cross_full.imag, N_SMOOTH)
    cross_phase_r = np.angle(cross_re_s + 1j * cross_im_s)[edge:-edge]
    cross_phase_r = cross_phase_r - np.mean(cross_phase_r)   # subtract phi0

    return dict(f_band=f_band, x_band=x_band,
                alpha_f=alpha_f, cross_phase=cross_phase_r)


def get_params(q, cache):
    key = min(cache.keys(), key=lambda k: abs(float(k) - q))
    return cache[key]["params"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", nargs="+", type=float,
                    default=[3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    ap.add_argument("--coord", choices=["f", "x"], default="x",
                    help="x-axis: f [cycles/M] or x=(pi*f)^{2/3}")
    args = ap.parse_args()

    qs = sorted(args.q)
    cache = json.loads((ROOT / "gw_remnant_energy_results"
                              / "per_q_cache_mult.json").read_text())

    # For q < 3 not in cache: optimize fresh
    need_fresh = [q for q in qs if min(float(k) for k in cache) > q - 0.01
                  or float(min(cache.keys(), key=lambda k: abs(float(k)-q))) > q + 0.1]
    fresh_params = {}
    for q in qs:
        key = min(cache.keys(), key=lambda k: abs(float(k) - q))
        if abs(float(key) - q) > 0.05:
            print(f"  Optimizing per-q mult at q={q} (not in cache) ...", flush=True)
            case = G.load_case(q)
            err, p, t0 = G.optimize_case(q, form="mult")
            fresh_params[q] = list(p) + [0.0]
            print(f"    err={err:.4e}")

    # colours: viridis from low-q to high-q
    cmap   = cm.viridis
    colors = [cmap(i / max(len(qs) - 1, 1)) for i in range(len(qs))]

    use_x = (args.coord == "x")
    xlabel = (r"$x = (\pi M f_\mathrm{GW})^{2/3}$" if use_x
              else r"$f\;[\mathrm{cycles}/M]$")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(
        r"Frequency-domain alpha and beta diagnostics — multi-$q$ overlay"
        "\n(per-q mult model; viridis: low $q$ → high $q$)",
        fontsize=10)

    ax1, ax2 = axes

    for q, col in zip(qs, colors):
        if q in fresh_params:
            p = fresh_params[q]
        else:
            p = get_params(q, cache)

        print(f"q={q}  a_PP={p[0]:.4f}  b_PP={p[2]:.4f}", flush=True)
        res = compute_spectra(q, p)

        coord = res["x_band"] if use_x else res["f_band"]
        lbl   = f"$q={q:g}$"

        ax1.plot(coord, res["alpha_f"],    color=col, lw=1.3, label=lbl)
        ax2.plot(coord, res["cross_phase"], color=col, lw=1.3, label=lbl)

    # panel 1
    ax1.axhline(1.0, color="k", lw=0.8, ls=":", alpha=0.5)
    ax1.set_xlabel(xlabel, fontsize=10)
    ax1.set_ylabel(r"$\alpha(f)\,/\,\alpha_{PP}$", fontsize=10)
    ax1.set_title(r"Amplitude ratio $\alpha(f)/\alpha_{PP}$", fontsize=9.5)
    ax1.legend(fontsize=7.5, ncol=2)
    ax1.grid(alpha=0.25)

    # panel 2
    ax2.axhline(0.0, color="k", lw=0.8, ls=":", alpha=0.5)
    ax2.set_xlabel(xlabel, fontsize=10)
    ax2.set_ylabel(r"$\angle(H_\mathrm{NR}\,\tilde{h}^\ast_{pp}) - \langle\phi_0\rangle$  [rad]",
                   fontsize=9)
    ax2.set_title(r"Cross-spectrum phase (mean $\phi_0$ subtracted) $\propto$ $\beta$ residual",
                  fontsize=9.5)
    ax2.legend(fontsize=7.5, ncol=2)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    out = ROOT / "Agentic_plots" / "fdomain_alpha_beta_overlay.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
