"""THE COMPARISON PLOT: calibration-implied second-order flux vs second-order self-force.

Our side (closed form, derivation in wardell_ideation.md 4g):

    R2_calib(x) = F2_22/F1_22 = 2 a(x) - 2 b(x) + (2/3) L(x) b(x) + (4/5)[5 - L(x)]
    a(x) = (55/42) x + a2 x^2        b(x) = b1 x + b2 x^2      (pn_anchored inspiral)
    L(x) = dlnF1_22/dlnx, from the test-mass PN (2,2) flux (see below)

Their side:
  * the analytic PN nu-part of the (2,2) flux at fixed waveform x (leading term 55x/21,
    the coefficient Warburton+ Fig. 7 names) -- constructed from the PN (2,2) amplitude
      Hhat_22 = 1 + x(-107/42 + 55 nu/42) + 2 pi x^(3/2)
                + x^2(-2173/1512 - 1069 nu/216 + 2047 nu^2/1512) + ...
    via F_22 ~ x^5 |Hhat_22|^2:  nu-part of Fhat_22 = (55/21) x + [2 A0 A1 + 2 B1] x^2,
    then divided by the test-mass Fhat1_22 = |Hhat_22(nu=0)|^2 to match our F2/F1
    normalisation.  (2PN amplitude coefficients per Blanchet LRR / Faye+ 1204.1043;
    flagged for independent verification.)
  * the WaSABI 2SF data (wardell_data/F2_EI_schwarz_circ.json), which is the TOTAL flux
    (all modes) at fixed ORBITAL x in the nu-scheme -- verified by its PN limit
    -(35/12)x to 5% at x=0.05.  A DIFFERENT observable from the (2,2) ratio; drawn for
    orientation only, clearly labelled.

Both pn_anchored coefficient sets are drawn (shipped + refit basin) as the ~10%
parameter-degeneracy band.

Cheap (JSON + closed forms, no waveforms); run via sbatch for the env.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcdefaults()
mpl.rcParams["text.usetex"] = False

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Agentic_plots" / "wardell_comparison"
OUT.mkdir(parents=True, exist_ok=True)

# ---- our side --------------------------------------------------------------
THETAS = {
    "shipped": dict(b1=0.352, b2=8.553, a2=-22.9193, ls="-", lw=2.0, c="tab:red"),
    "refit basin": dict(b1=0.2032, b2=5.8594, a2=-19.5788, ls="--", lw=1.4, c="tab:red"),
}

# test-mass PN (2,2) amplitude coefficients (nu = 0 parts)
A0 = -107.0 / 42.0          # 1PN
C15 = 2.0 * np.pi           # 1.5PN tail
B0 = -2173.0 / 1512.0       # 2PN
# nu-parts
A1 = 55.0 / 42.0
B1 = -1069.0 / 216.0


def Fhat1_22(x):
    """Test-mass Newtonian-normalised (2,2) flux |Hhat(nu=0)|^2 through 2PN+tail."""
    return (1.0 + 2 * A0 * x + 2 * C15 * x ** 1.5
            + (A0 ** 2 + 2 * B0) * x ** 2)


def L_22(x):
    """dln F1_22 / dln x from the same series."""
    g = 2 * A0 * x + 3 * C15 * x ** 1.5 + 2 * (A0 ** 2 + 2 * B0) * x ** 2
    return 5.0 + g / Fhat1_22(x)


def R2_calib(x, b1, b2, a2):
    a = (55.0 / 42.0) * x + a2 * x ** 2
    b = b1 * x + b2 * x ** 2
    L = L_22(x)
    return 2 * a - 2 * b + (2.0 / 3.0) * L * b + (4.0 / 5.0) * (5.0 - L)


def R2_pn22(x):
    """Analytic PN nu-part of F2_22/F1_22 at fixed waveform x, through 2PN."""
    nu_part_newt_norm = (55.0 / 21.0) * x + (2 * A0 * A1 + 2 * B1) * x ** 2
    return nu_part_newt_norm / Fhat1_22(x)


# ---- their data: the TRUE (2,2) second-order ratio -------------------------
# built from the WaSABI mode-resolved Teukolsky amplitudes (1SF + 2SF) via the
# assembly in AmplitudeModels/1PAT1.m (Wardell Eq. 7 re-expansion included):
#   R2_22 = 2 Re[h2_eff/h0],  h2_eff = h0 + h2 + (2/3) r0 (1-(1/r0)/sqrt(1-3/r0)) h0'
# Validated: R2_22/x -> 55/21 at the low-x end (2.33 at x=0.05, bending as PN predicts).
dw = json.loads((ROOT / "wardell_data" / "R2_22_from_amplitudes.json").read_text())
xw = np.array(dw["x"]); R2w = np.array(dw["R2_22"])

# ---- plot ------------------------------------------------------------------
x = np.linspace(0.02, 0.16, 300)
fig, ax = plt.subplots(figsize=(7.5, 5))

for name, t in THETAS.items():
    ax.plot(x, R2_calib(x, t["b1"], t["b2"], t["a2"]), color=t["c"], ls=t["ls"],
            lw=t["lw"], zorder=4,
            label=f"calibration-implied  $F^{{(2)}}_{{22}}/F^{{(1)}}_{{22}}$  ({name})")

ax.plot(x, R2_pn22(x), color="tab:blue", lw=1.8, zorder=3,
        label=r"PN prediction, $(2,2)$ mode (through 2PN)")
ax.plot(x, (55.0 / 21.0) * x, color="gray", ls=":", lw=1.0, zorder=2,
        label=r"leading term $55x/21$ (both sides, exact)")
ax.plot(xw, R2w, color="tab:green", lw=2.0, marker="o", ms=3.5, zorder=5,
        label=r"2SF data, $(2,2)$ mode (WaSABI amplitudes"
              "\n"
              r"via Wardell Eq. 7) — the comparison target")

ax.axvline(0.13, color="0.6", ls="--", lw=0.8)
ax.text(0.131, 0.9, "adiabatic band edge", rotation=90, fontsize=7,
        color="0.4", va="top", transform=ax.get_xaxis_transform())
ax.set_xlabel(r"$x = (M\varpi)^{2/3}$   ($\varpi = \dot\Phi_{22}/2$, waveform frequency)")
ax.set_ylabel(r"$F^{(2)}/F^{(1)}$")
ax.set_title("Second-order flux: calibration-implied vs self-force / PN", fontsize=11)
ax.grid(True, alpha=0.4)
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
for ext in ("pdf", "png"):
    p = OUT / f"F2_comparison.{ext}"
    fig.savefig(p, dpi=130)
    print("wrote", p)

# numbers at reference x
print(f"\n{'x':>6} {'calib(ship)':>12} {'calib(refit)':>13} {'PN22':>9} {'2SF (2,2) data':>14}")
for xq in (0.03, 0.05, 0.08, 0.11, 0.14):
    cs = R2_calib(xq, **{k: THETAS['shipped'][k] for k in ('b1', 'b2', 'a2')})
    cr = R2_calib(xq, **{k: THETAS['refit basin'][k] for k in ('b1', 'b2', 'a2')})
    pn = R2_pn22(xq)
    tw = float(np.interp(xq, xw, R2w)) if xw.min() <= xq <= xw.max() else np.nan
    print(f"{xq:6.2f} {cs:12.4f} {cr:13.4f} {pn:9.4f} {tw:14.4f}")
