"""TOTAL-flux second-order comparison: hm-backbone-implied vs Warburton 2SF data.

Per mode, the backbone (note Eq. 43) gives alpha_lm = alpha_22 * C_lm(nu) *
beta^(-(l+eps-2)/3) with a SHARED time map, so the per-mode flux identity is
F_NR,lm = (alpha_lm^2/beta^2) F_pp,lm and the (2,2) derivation generalises exactly:

  R2_lm(x) = 2[ a(x) + c_lm + e_lm b(x) ] - 2 b(x) + (2/3) L_lm(x) b(x)
             + (4/5)[ p_lm - L_lm(x) ]        ( - 2 ct_lm  in the rho-measured variant)

  c_lm = nu-slope of C_lm  = -(l+eps_lm-1)  for k>=2,  0 for (2,2)   [exact]
  e_lm = -(l+eps_lm-2)/3                     [beta-power in the backbone]
  p_lm = l+eps_lm+3                          [Newtonian exponent of F1_lm]
  L_lm = dlnF1_lm/dlnx: ANALYTIC -- exact PN (through 2PN+tail) for (2,2), and
         L_lm = p_lm + (L_22 - 5) for subdominant modes (same relative bending; their
         weights are <2% so the approximation is second-order small).  The measured
         log-gradient was tried first and spikes at envelope notches (v1 of this plot).
  ct_lm = measured constant-rho nu-slope, scaling_hm_backbone.md:
          (1-k)/nu q-flat at ~{0, 0.46, 1.2} for l=3,4,5 and 0.55 for (2,1)

Newtonian consistency was checked analytically: the backbone's beta-power makes the
per-mode unit-conversion constants cancel iff p_lm = l+eps+3, which holds.

  R2_tot(x) = sum_lm w_lm(x) R2_lm(x),   w_lm = F1_lm / sum F1_lm

with the weights and L_lm measured from the q=8 ppBHPT multi-mode cache (mode RATIOS of
the first-order fluxes are epsilon-independent, so any q serves; q=3 cross-checked).
Fluxes via the envelope form F_lm ~ m^2 Omega^2 |h_lm|^2 (no orbital-timescale
oscillation problem).  9 modes = BHPT/NR intersection; missing modes carry <0.1% of the
band's flux.

VALIDATION BUILT IN: R2_tot(x)/x must -> -35/12 = -2.917 as x->0 (the PN nu-part of the
total flux) -- the sign flip from (2,2)'s +55/21 must emerge from the C_lm factors.

Comparison data: WaSABI total 2SF/1SF flux (wardell_data/F2_EI_schwarz_circ.json /
F1_EI...), nu-scheme at fixed ORBITAL x (ours is waveform x-bar; difference "very
small" per Warburton).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

mpl.rcdefaults()
mpl.rcParams["text.usetex"] = False

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Agentic_plots" / "wardell_comparison"
OUT.mkdir(parents=True, exist_ok=True)

# pn_anchored inspiral drift shapes (shipped theta)
B1T, B2T, A2T = 0.352, 8.553, -22.9193
a_of = lambda x: (55.0 / 42.0) * x + A2T * x ** 2
b_of = lambda x: B1T * x + B2T * x ** 2

MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (3, 1), (4, 4), (4, 3), (4, 2), (5, 5)]
CT_RHO = {(2, 1): 0.55, (3, 3): 0.0, (3, 2): 0.0, (3, 1): 0.0,
          (4, 4): 0.46, (4, 3): 0.46, (4, 2): 0.46, (5, 5): 1.2, (2, 2): 0.0}

eps_of = lambda l, m: (l + m) % 2
c_of = lambda l, m: 0.0 if (l, m) == (2, 2) else -float(l + eps_of(l, m) - 1)
e_of = lambda l, m: -(l + eps_of(l, m) - 2) / 3.0
p_of = lambda l, m: float(l + eps_of(l, m) + 3)


def measure_modes(q):
    """w_lm(x) and L_lm(x) from the ppBHPT multi-mode cache at this q."""
    d = np.load(ROOT / ".cache" / "hm" / f"waveforms_hm_q{q:.10f}.npz")
    t = np.asarray(d["t_bhpt"], float)
    dt = float(np.median(np.diff(t)))
    n = max(int(round(20.0 / dt)) | 1, 7)
    h22 = d["bhpt_22_re"] + 1j * d["bhpt_22_im"]
    psi = np.unwrap(np.angle(h22))
    om = np.abs(savgol_filter(psi, n, 3, deriv=1, delta=dt))
    x = (0.5 * om) ** (2.0 / 3.0)
    Om2 = (0.5 * om) ** 2
    msk = t < t[np.argmax(np.abs(h22))] - 50.0          # inspiral only
    F = {}
    for (l, m) in MODES:
        h = d[f"bhpt_{l}{m}_re"] + 1j * d[f"bhpt_{l}{m}_im"]
        amp = savgol_filter(np.abs(h), n, 3)
        F[(l, m)] = (m ** 2) * Om2 * amp ** 2
    xg = x[msk]
    o = np.argsort(xg)
    xg = xg[o]
    Fg = {lm: F[lm][msk][o] for lm in MODES}
    tot = sum(Fg.values())
    w = {lm: Fg[lm] / tot for lm in MODES}
    return xg, w


# analytic L: test-mass PN (2,2) flux through 2PN + tail (as wardell_comparison_plot.py)
A0PN, C15PN, B0PN = -107.0 / 42.0, 2.0 * np.pi, -2173.0 / 1512.0


def L22_analytic(x):
    f = 1.0 + 2 * A0PN * x + 2 * C15PN * x ** 1.5 + (A0PN ** 2 + 2 * B0PN) * x ** 2
    g = 2 * A0PN * x + 3 * C15PN * x ** 1.5 + 2 * (A0PN ** 2 + 2 * B0PN) * x ** 2
    return 5.0 + g / f


def R2_lm(x, l, m, use_rho):
    a, b = a_of(x), b_of(x)
    c, e, p = c_of(l, m), e_of(l, m), p_of(l, m)
    L = p + (L22_analytic(x) - 5.0)          # exact for (2,2); shifted bending otherwise
    r = (2.0 * (a + c * np.ones_like(x) + e * b) - 2.0 * b
         + (2.0 / 3.0) * L * b + (4.0 / 5.0) * (p - L))
    if use_rho:
        r = r - 2.0 * CT_RHO[(l, m)]
    return r


def build_total(q, use_rho):
    xg, w = measure_modes(q)
    band = (xg > 0.03) & (xg < 0.155)
    xg = xg[band]
    tot = np.zeros_like(xg)
    for (l, m) in MODES:
        tot += w[(l, m)][band] * R2_lm(xg, l, m, use_rho)
    return xg, tot


def main():
    # data side
    d2 = json.loads((ROOT / "wardell_data" / "F2_EI_schwarz_circ.json").read_text())
    x2, F2 = np.array(d2["x"]), np.array(d2["F2_EI"])
    d1 = json.loads((ROOT / "wardell_data" / "F1_EI_schwarz_circ.json").read_text())
    r1, F1 = np.array(d1["r0"]), np.array(d1["F1_EI"])
    x1 = 1.0 / r1
    o = np.argsort(x1)
    m2 = (x2 >= x1.min()) & (x2 <= x1.max())
    xd = x2[m2]
    Rd = F2[m2] / np.interp(xd, x1[o], F1[o])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for use_rho, ls, lab in ((False, "-", r"backbone-implied total ($\rho=1$)"),
                             (True, "--", r"backbone + measured $\rho$ slopes")):
        xg, tot = build_total(8.0, use_rho)
        ax.plot(xg, tot, color="tab:red", ls=ls, lw=1.8 if not use_rho else 1.3,
                label=lab)
        if not use_rho:
            print(f"\nVALIDATION rho=1: R2_tot/x -> -35/12 = {-35/12:.4f} ?")
            for xq in (0.035, 0.05, 0.08, 0.11, 0.14):
                v = float(np.interp(xq, xg, tot))
                print(f"  x={xq:5.3f}  R2_tot={v:+8.4f}  /x={v/xq:+8.3f}")
        # q=3 cross-check of the measured weights
    xg3, tot3 = build_total(3.0, False)
    ax.plot(xg3, tot3, color="tab:orange", ls=":", lw=1.0,
            label=r"same, weights from $q=3$ cache (consistency)")

    ax.plot(xd, Rd, color="tab:green", lw=2.0, marker="o", ms=3.5, zorder=5,
            label="Warburton 2SF data, TOTAL flux")
    xr = np.linspace(0.02, 0.16, 100)
    ax.plot(xr, -(35.0 / 12.0) * xr, color="gray", ls=":", lw=1.0,
            label=r"PN leading term $-35x/12$ (total)")
    ax.axvline(0.13, color="0.6", ls="--", lw=0.8)
    ax.text(0.131, 0.97, "adiabatic band edge", rotation=90, fontsize=7,
            color="0.4", va="top", transform=ax.get_xaxis_transform())
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$F^{(2)}_{\rm tot}/F^{(1)}_{\rm tot}$")
    ax.set_title("TOTAL second-order flux: hm-backbone-implied vs 2SF data", fontsize=11)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = OUT / f"F2_total_comparison.{ext}"
        fig.savefig(p, dpi=130)
        print("wrote", p)

    xg, tot = build_total(8.0, False)
    print(f"\n{'x':>6} {'backbone tot':>13} {'+rho':>9} {'2SF data':>10}")
    xgr, totr = build_total(8.0, True)
    for xq in (0.035, 0.05, 0.08, 0.11, 0.14):
        v = float(np.interp(xq, xg, tot))
        vr = float(np.interp(xq, xgr, totr))
        vd = float(np.interp(xq, xd, Rd)) if xd.min() <= xq <= xd.max() else np.nan
        print(f"{xq:6.3f} {v:13.4f} {vr:9.4f} {vd:10.4f}")


if __name__ == "__main__":
    main()
