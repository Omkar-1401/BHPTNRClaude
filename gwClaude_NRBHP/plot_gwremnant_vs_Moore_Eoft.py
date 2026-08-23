"""
Compare the radiated-energy history E(t) from `gw_remnant` against the 3PN
expressions of Moore, Favata, Arun & Mishra, arXiv:1605.00304 (circular limit).

Three curves, all in units of the total mass M and all zeroed at the first
sample so they are directly overlayable:

  1. gw_remnant          E(t) = cumtrapz( (1/16pi) sum_lm |dh_lm/dt|^2 )
                         i.e. RemnantMassCalculator.Eoft, the quantity behind
                         `calc.plot_mass_energy()`.  Modes (2,2) and (2,-2) of
                         the BHPT surrogate.

  2. Moore Eq. (6.3)     E(t) = int F(v(t')) dt'   -- integrate the 3PN GW
                         luminosity.  Structurally the same operation as (1),
                         with the PN flux replacing the waveform flux, so this
                         is the like-for-like comparison.

  3. Moore Eq. (6.2)     E(t) = E_orb(v_0) - E_orb(v(t))  -- the 3PN orbital
                         BINDING energy, converted to radiated energy through
                         energy balance.  Eq. (6.2) is not itself a radiated
                         energy, so it needs this step before it can be
                         compared with (1).

Both PN curves are evaluated at e_0 = 0 (circular), truncated at 3PN (v^6), and
driven by the PN velocity parameter read off the BHPT waveform itself:

    omega_22 = |d/dt arg h_22|,  omega_orb = omega_22 / 2,  v = (M omega_orb)^(1/3)

so all three curves share one time axis and one frequency evolution; the only
difference is the energy prescription.

Output: gwremnant_vs_Moore_Eoft.pdf
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import fit_scaling_PN_opt_creative_q_dep as qdep
from gw_remnant.remnant_calculators.remnant_mass_calculator import (
    RemnantMassCalculator,
)

Q_VALUES = (3.0, 5.0, 8.0)
T_WINDOW = (-1000.0, 50.0)      # relative to the (2,2) amplitude peak, in M
OUT_PDF  = ROOT / "gwremnant_vs_Moore_Eoft.pdf"

EULER_GAMMA = 0.5772156649015329

# categorical slots 1, 2, 7 of the reference palette (validated: worst adjacent
# CVD dE 24.7, normal-vision dE 33.6, all >= 3:1 on a light surface)
C_GWR   = "#2a78d6"   # blue
C_FLUX  = "#eb6834"   # orange
C_BIND  = "#4a3aa7"   # violet
INK     = "#1a1a19"
MUTED   = "#6b6a63"
GRID    = "#e3e2dd"


# ---------------------------------------------------------------------------
# Moore et al. 1605.00304, circular limit (e_0 = 0), through 3PN
# ---------------------------------------------------------------------------

def E_orb_3PN(v, eta, M=1.0):
    """
    Eq. (6.2), orbital binding energy, e_0 -> 0, truncated at v^6 (3PN).

        E(v) = -(1/2) eta M v^2 [ 1 - (3/4 + eta/12) v^2
                                    + (-27/8 + 19 eta/8 - eta^2/24) v^4
                                    + ( -675/64 + (34445/576 - 205 pi^2/96) eta
                                        - 155 eta^2/96 - 35 eta^3/5184 ) v^6 ]
    """
    c2 = -(3.0 / 4.0 + eta / 12.0)
    c4 = -27.0 / 8.0 + 19.0 * eta / 8.0 - eta ** 2 / 24.0
    c6 = (-675.0 / 64.0
          + (34445.0 / 576.0 - 205.0 * np.pi ** 2 / 96.0) * eta
          - 155.0 * eta ** 2 / 96.0
          - 35.0 * eta ** 3 / 5184.0)
    return -0.5 * eta * M * v ** 2 * (1.0 + c2 * v ** 2 + c4 * v ** 4 + c6 * v ** 6)


def flux_3PN(v, eta):
    """
    Eq. (6.3), GW energy flux F(v), e_0 -> 0, truncated at v^6 (3PN).

    NOTE the 32/5 prefactor.  The pre-existing helper `Moore_PN_exprs.dEdt` in
    this workspace uses 32 (no /5); that cancels in the models built on it,
    which only ever use peak-normalised flux coordinates, but it is a factor 5
    high as an absolute luminosity and must not be used here.
    """
    c2 = -(1247.0 / 336.0 + 35.0 * eta / 12.0)
    c3 = 4.0 * np.pi
    c4 = -44711.0 / 9072.0 + 9271.0 * eta / 504.0 + 65.0 * eta ** 2 / 18.0
    c5 = -(8191.0 / 672.0 + 583.0 * eta / 24.0) * np.pi
    c6 = (6643739519.0 / 69854400.0
          - 1712.0 / 105.0 * EULER_GAMMA
          + 16.0 * np.pi ** 2 / 3.0
          + (-134543.0 / 7776.0 + 41.0 * np.pi ** 2 / 48.0) * eta
          - 94403.0 * eta ** 2 / 3024.0
          - 775.0 * eta ** 3 / 324.0
          - 856.0 / 105.0 * np.log(16.0 * v ** 2))
    return (32.0 / 5.0) * eta ** 2 * v ** 10 * (
        1.0 + c2 * v ** 2 + c3 * v ** 3 + c4 * v ** 4 + c5 * v ** 5 + c6 * v ** 6)


# ---------------------------------------------------------------------------

def build_case(q):
    d = np.load(qdep.waveform_cache_path(q))
    t = d["t_bhpt"]
    h = d["h_bhpt_re"] + 1j * d["h_bhpt_im"]
    eta = float(d["nu"])

    # gw_remnant radiated energy (the plot_mass_energy quantity)
    calc = RemnantMassCalculator(
        time=t, h_dict={(2, 2): h, (2, -2): np.conj(h)}, q=q,
        E_initial=0.0, L_initial=0.0, M_initial=1.0, use_filter=False,
    )
    e_gwr = np.asarray(calc.Eoft, dtype=float)

    # PN velocity parameter from the waveform's own phasing.  The surrogate
    # phase carries isolated 1-2 sample glitches (e.g. t = -600.2 and -15.2 at
    # q=8, where |h| is perfectly smooth), which np.gradient turns into large
    # spikes; an 11-point median kills those exactly and leaves the smooth
    # inspiral trend untouched.
    phase = np.unwrap(np.angle(h))
    omega_22 = median_filter(np.abs(np.gradient(phase, t)), size=11, mode="nearest")
    v = (0.5 * omega_22) ** (1.0 / 3.0)          # M = 1

    e_flux = cumulative_trapezoid(flux_3PN(v, eta), t, initial=0.0)
    e_bind = E_orb_3PN(v[0], eta) - E_orb_3PN(v, eta)

    i_merger = int(np.argmax(np.abs(h)))
    return {
        "q": q, "eta": eta, "t": t, "v": v, "i_merger": i_merger,
        "t_merger": float(t[i_merger]),
        "e_gwr": e_gwr, "e_flux": e_flux, "e_bind": e_bind,
        "alpha_i": fitted_alpha_i(q),
    }


def fitted_alpha_i(q):
    """
    Inspiral amplitude calibration alpha_i(q) from the gw_remnant_energy per-q
    fits, if available.  The BHPT surrogate is evaluated with calibrated=False,
    so its amplitude is off by roughly alpha_i and its energy by alpha_i**2 --
    plotting alpha_i**2 * E_gwr tests whether the offset against PN is just
    that missing calibration.
    """
    path = ROOT / "gw_remnant_energy_results" / "per_q_cache.json"
    if not path.exists():
        return None
    cache = json.loads(path.read_text())
    row = cache.get(f"{q:.10f}")
    return float(row["params"][0]) if row else None


def main():
    cases = [build_case(q) for q in Q_VALUES]

    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.linewidth": 0.6, "lines.linewidth": 1.4,
        "figure.dpi": 140, "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(3, len(cases), figsize=(11.0, 8.2), sharex="col")

    for j, c in enumerate(cases):
        # time relative to the amplitude peak, linear axis
        rel = c["t"] - c["t_merger"]
        m = (rel >= T_WINDOW[0]) & (rel <= T_WINDOW[1])
        x = rel[m]

        # ---- row 0: radiated energy -------------------------------------
        ax = axes[0, j]
        ax.plot(x, c["e_gwr"][m],  color=C_GWR,  label="gw_remnant  $E_{\\rm rad}(t)$")
        ax.plot(x, c["e_flux"][m], color=C_FLUX, ls=(0, (5, 2)),
                label="Moore Eq. (6.3)  $\\int\\!\\mathcal{F}\\,dt$")
        ax.plot(x, c["e_bind"][m], color=C_BIND, ls=(0, (1.5, 1.5)),
                label="Moore Eq. (6.2)  $-\\Delta E_{\\rm orb}$")
        if c["alpha_i"] is not None:
            ax.plot(x, c["alpha_i"] ** 2 * c["e_gwr"][m], color=C_GWR, lw=1.0,
                    ls=(0, (1, 1.6)), alpha=0.85,
                    label="gw_remnant $\\times\\,\\alpha_i^2$  (calibrated)")
        ax.set_title(f"$q = {c['q']:.0f}$   ($\\eta = {c['eta']:.4f}$)", pad=6)
        if j == 0:
            ax.set_ylabel("radiated energy  $E(t)\\ [M]$")
            ax.legend(frameon=False, loc="upper left", handlelength=2.6,
                      borderaxespad=0.4)

        # ---- row 1: fractional difference vs gw_remnant ------------------
        ax = axes[1, j]
        eg = c["e_gwr"][m]
        with np.errstate(divide="ignore", invalid="ignore"):
            ax.plot(x, c["e_flux"][m] / eg - 1.0, color=C_FLUX, ls=(0, (5, 2)))
            ax.plot(x, c["e_bind"][m] / eg - 1.0, color=C_BIND, ls=(0, (1.5, 1.5)))
            if c["alpha_i"] is not None:
                a2 = c["alpha_i"] ** 2
                ax.plot(x, c["e_flux"][m] / (a2 * eg) - 1.0, color=C_FLUX,
                        lw=1.0, ls=(0, (1, 1.6)), alpha=0.85)
        ax.axhline(0.0, color=C_GWR, lw=1.0)
        ax.set_ylim(-0.8, 0.4)
        ax.axhspan(-0.1, 0.1, color=GRID, zorder=0)
        if j == 0:
            ax.set_ylabel("PN / gw_remnant $-$ 1")
            ax.text(0.03, 0.72, "$\\pm10\\%$ band", transform=ax.transAxes,
                    color=MUTED, fontsize=7)
            ax.text(0.03, 0.06, "dotted: same ratio after $\\alpha_i^2$ rescaling",
                    transform=ax.transAxes, color=MUTED, fontsize=7)

        # ---- row 2: PN expansion parameter -------------------------------
        ax = axes[2, j]
        ax.plot(x, c["v"][m], color=INK, lw=1.2)
        for lev, lab in ((0.3, "$v=0.3$"), (0.4, "$v=0.4$")):
            ax.axhline(lev, color=MUTED, lw=0.6, ls=":")
            if j == 0:
                ax.text(T_WINDOW[0] * 0.95, lev * 1.02, lab, color=MUTED, fontsize=7)
        ax.set_ylim(0.05, 0.65)
        ax.set_xlabel("$t - t_{\\rm peak}\\ [M]$")
        if j == 0:
            ax.set_ylabel("PN parameter  $v = (M\\omega_{\\rm orb})^{1/3}$")

        for ax in axes[:, j]:
            ax.set_xlim(*T_WINDOW)
            ax.axvline(0.0, color=MUTED, lw=0.6, ls=(0, (4, 3)))
            ax.grid(True, color=GRID, lw=0.5, alpha=0.9)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)

    fig.suptitle(
        "Radiated energy from gw_remnant vs the 3PN expressions of "
        "Moore et al. (arXiv:1605.00304), circular limit $e_0 = 0$",
        fontsize=10.5, y=0.985)
    fig.text(0.5, 0.005,
             "PN curves use $v(t)$ read off the BHPT (2,2) phasing, so all three "
             "share one time axis and one frequency evolution.  Truncated at "
             "$v^6$ (3PN).  BHPT surrogate, uncalibrated.  Dashed vertical line "
             "marks the (2,2) amplitude peak; PN is not valid to its right.",
             ha="center", color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PDF.with_suffix(".png"), dpi=140)
    print(f"Wrote {OUT_PDF}")

    # quick numeric summary at a few times before merger
    print("\nfractional difference vs gw_remnant  (PN/gwr - 1)")
    print(f"{'q':>4} {'t-t_peak':>10} {'Eq6.3 flux':>12} {'Eq6.2 bind':>12} {'v':>7}")
    for c in cases:
        rel = c["t"] - c["t_merger"]
        for target in (-1000.0, -500.0, -200.0, -50.0, 0.0):
            i = int(np.argmin(np.abs(rel - target)))
            print(f"{c['q']:>4.0f} {rel[i]:>10.0f} "
                  f"{c['e_flux'][i]/c['e_gwr'][i]-1:>12.4f} "
                  f"{c['e_bind'][i]/c['e_gwr'][i]-1:>12.4f} {c['v'][i]:>7.3f}")


if __name__ == "__main__":
    main()
