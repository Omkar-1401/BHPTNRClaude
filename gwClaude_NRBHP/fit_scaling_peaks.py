"""
Stage 1 of the peaks-based BHPT->NR calibration model.

Implements the "using the peaks" method of Islam & Khanna (arXiv:2307.03155,
Sec. II.A.3): at each waveform peak the local amplitude- and time-rescaling
factors are read off directly as ratios between NR and ppBHPT, rather than
obtained from an L2 waveform fit.

For a peak k (counted back from merger, k=0 at merger):

    alpha_peak(k) = |h|_NR(t_NR,k) / |h|_ppBHPT(t_bhpt,k)
    beta_abs(k)   = (t_NR,k - t_NR,merger) / (t_bhpt,k - t_bhpt,merger)   # paper-literal
    beta_slope(k) = d t_NR / d t_bhpt  along the peak time-map            # local rate

Peaks are located from the GW phase (extrema spaced by pi in unwrapped phase),
which is more robust than amplitude peak-finding and immune to a constant phase
offset; amplitudes and frequencies are read off by interpolation at the peak
times (the cubic-spline-precision step the paper describes).

Each peak is tagged with a frequency coordinate x = (M*omega_orb)^(2/3) taken
from the ppBHPT side (omega_orb = omega_gw/2 for the (2,2) mode). x is a monotone
reparameterisation of the paper's orbital frequency omega_orb; PN amplitude/phase
corrections are low-order polynomials in x, which is why it is the fit coordinate.

This module only MEASURES alpha(t), beta(t) per q and caches the point clouds.
The (x, nu) regression + QNM ringdown branch live in a separate Stage-2 script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
UT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UT_ROOT / "BHPTNRSurrogate" / "surrogates"))

MODE = (2, 2)
WAVEFORM_CACHE_DIR = ROOT / ".cache" / "q_dep"
RESULTS_DIR = ROOT / "peaks_results"
PER_Q_DIR = RESULTS_DIR / "per_q"
DPHI = np.pi  # peak spacing in GW phase (extrema: crests + troughs)

# Calibration range: the peaks method is trustworthy through the inspiral up to
# the last few pre-merger cycles.  Beyond t_peaks_end (M) the peaks are sparse
# and the ringdown is handled by the QNM/remnant branch in Stage 2.
NR_T_START = -5000.1


def grid_q_values(q_lo: float = 3.0, q_hi: float = 8.0) -> list[float]:
    qs = []
    for path in WAVEFORM_CACHE_DIR.glob("waveforms_q*.npz"):
        q = float(path.stem.replace("waveforms_q", ""))
        if q_lo - 1e-9 <= q <= q_hi + 1e-9:
            qs.append(q)
    return sorted(qs)


def load_waveforms(q: float):
    d = np.load(WAVEFORM_CACHE_DIR / f"waveforms_q{q:.10f}.npz")
    t_bhpt = d["t_bhpt"]
    h_bhpt = d["h_bhpt_re"] + 1j * d["h_bhpt_im"]
    t_nr = d["t_nr"]
    h_nr = d["h_nr_re"] + 1j * d["h_nr_im"]
    return t_bhpt, h_bhpt, t_nr, h_nr


def phase_peaks(t: np.ndarray, h: np.ndarray, dphi: float = DPHI):
    """Times/amps/freqs of GW-phase extrema, counting back from merger (k=0)."""
    amp = np.abs(h)
    t_m = float(t[int(np.argmax(amp))])
    psi = np.unwrap(np.angle(h))
    sign = 1.0 if psi[-1] > psi[0] else -1.0
    psi_mono = sign * psi                       # strictly increasing in t (inspiral)
    psi_m = float(np.interp(t_m, t, psi_mono))
    n_max = int((psi_m - psi_mono[0]) / dphi)
    targets = psi_m - dphi * np.arange(0, n_max + 1)
    targets = targets[targets >= psi_mono[0]]
    t_pk = np.interp(targets, psi_mono, t)      # k=0 -> merger, k increasing -> past
    amp_pk = np.interp(t_pk, t, amp)
    omega_gw = np.abs(np.gradient(psi, t))
    om_pk = np.interp(t_pk, t, omega_gw)
    return t_m, t_pk, amp_pk, om_pk


def extract(q: float, nr_t_start: float = NR_T_START) -> dict:
    """Measure the alpha/beta peak point cloud for one mass ratio."""
    t_bhpt, h_bhpt, t_nr, h_nr = load_waveforms(q)

    tm_b, tpk_b, amp_b, om_b = phase_peaks(t_bhpt, h_bhpt)
    tm_n, tpk_n, amp_n, om_n = phase_peaks(t_nr, h_nr)

    # match by peak index from merger over the common count
    K = min(len(tpk_b), len(tpk_n))
    tpk_b, amp_b, om_b = tpk_b[:K], amp_b[:K], om_b[:K]
    tpk_n, amp_n, om_n = tpk_n[:K], amp_n[:K], om_n[:K]

    tb = tpk_b - tm_b        # merger-relative BHPT peak times (<= 0)
    tn = tpk_n - tm_n        # merger-relative NR peak times   (<= 0)

    alpha = amp_n / amp_b
    with np.errstate(invalid="ignore", divide="ignore"):
        beta_abs = np.where(tb != 0.0, tn / tb, np.nan)
    # local rate d t_NR / d t_bhpt along the (monotone) peak time-map
    beta_slope = np.gradient(tn, tb)
    x = (om_b / 2.0) ** (2.0 / 3.0)              # ppBHPT-side PN frequency coord

    # restrict to the calibration window (drop the merger point with beta_abs nan)
    keep = (tpk_n >= nr_t_start) & np.isfinite(beta_abs) & (tb < 0.0)
    nu = q / (1.0 + q) ** 2
    return {
        "q": q,
        "nu": nu,
        "chi": 0.0,
        "k": np.arange(K)[keep],
        "t_bhpt": tb[keep],
        "t_nr": tn[keep],
        "x": x[keep],
        "omega_gw_bhpt": om_b[keep],
        "alpha": alpha[keep],
        "beta_abs": beta_abs[keep],
        "beta_slope": beta_slope[keep],
        "t_bhpt_merger": tm_b,
        "t_nr_merger": tm_n,
        "n_peaks": int(np.count_nonzero(keep)),
    }


def save_case(case: dict) -> None:
    PER_Q_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(PER_Q_DIR / f"peaks_q{case['q']:.10f}.npz", **case)


def inspiral_roughness(case: dict, x_hi: float = 0.12) -> tuple[float, float]:
    """std of successive differences in the inspiral (noise proxy) for both betas."""
    m = case["x"] < x_hi
    if np.count_nonzero(m) < 5:
        return float("nan"), float("nan")
    return (
        float(np.std(np.diff(case["beta_abs"][m]))),
        float(np.std(np.diff(case["beta_slope"][m]))),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q-lo", type=float, default=3.0)
    ap.add_argument("--q-hi", type=float, default=8.0)
    ap.add_argument("--only", type=float, default=None, help="run a single q")
    args = ap.parse_args()

    qs = [args.only] if args.only else grid_q_values(args.q_lo, args.q_hi)
    print(f"extracting peaks for {len(qs)} mass ratios in [{args.q_lo},{args.q_hi}]")
    print(f"{'q':>6} {'nu':>7} {'chi_f?':>6} {'Npk':>4} {'x_lo':>6} {'x_hi':>6} "
          f"{'a_lo':>7} {'a_mrg':>7} {'b_lo':>7} {'b_mrg':>7} {'rgh_abs':>8} {'rgh_slp':>8}")
    rough_abs, rough_slp = [], []
    for q in qs:
        c = extract(q)
        save_case(c)
        ra, rs = inspiral_roughness(c)
        rough_abs.append(ra)
        rough_slp.append(rs)
        # order peaks by x for readable endpoints
        o = np.argsort(c["x"])
        a = c["alpha"][o]
        bs = c["beta_slope"][o]
        print(f"{q:6.3f} {c['nu']:7.4f} {'-':>6} {c['n_peaks']:4d} "
              f"{c['x'].min():6.3f} {c['x'].max():6.3f} "
              f"{a[0]:7.4f} {a[-1]:7.4f} {bs[0]:7.4f} {bs[-1]:7.4f} "
              f"{ra:8.2e} {rs:8.2e}")

    ra = np.nanmedian(rough_abs)
    rs = np.nanmedian(rough_slp)
    print(f"\nmedian inspiral roughness: beta_abs={ra:.3e}  beta_slope={rs:.3e}")
    print(f"beta recommendation: {'slope' if rs <= ra * 1.15 else 'abs'} "
          f"(slope preferred unless clearly worse; it is integrable and merger-safe)")
    print(f"saved per-q point clouds to {PER_Q_DIR}")


if __name__ == "__main__":
    main()
