"""MODEL-FREE flux-ratio measurement vs second-order self-force.  NO calibration.

For each waveform SEPARATELY (own phase, own clock, no time map, no alpha/beta):

    x(t)   = (0.5 * d/dt unwrap arg h_22)^(2/3)      savgol analytic deriv, 20 M window
    F_lm   = m^2 * Omega^2 * |h_lm|^2                envelope form (no orbital
                                                     oscillation), Omega from the 22
    (2,2): F = F_22.   total: F = sum over the 9 cached modes.

Then the ratio AT EQUAL x, with only the DERIVED unit bookkeeping (nu^2 = eps^2 X1^4,
so F_pp at its own x is eps^2 F1(x) while F_NR is nu^2 F1(x)(1 + nu R2)):

    R2_meas(x) = ( F_NR(x)/F_pp(x) / X1^4  -  1 ) / nu

Zero fitted parameters.  Pointwise in x, so it CANNOT have fit leakage — this is the
arbiter between the calibration-implied curves and the Warburton data (wardell_ideation
4h/4i): if R2_meas agrees with Warburton, the divergence is pn_anchored's fault
(form/window); if not, it is in the surrogates.

Built-in validations, printed per q:
  * F_NR/F_pp / X1^4 must sit near 1 (the derived conversion is right);
  * the R2_meas curves must COLLAPSE across q (the 4c nu-independence test).

Caveats: NR side is NRHybSur (PN-hybridised at early times, so the low-x end of "NR" is
really PN); the standard NR cache starts ~ -5000 M so the band begins at x ~ 0.055-0.06;
first 300 M of NR dropped (startup); q=2, 2.5 excluded (BHPT surrogate out of domain).
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

QCOLS = {3.0: "tab:orange", 4.0: "tab:purple", 5.0: "tab:cyan",
         6.0: "tab:olive", 8.0: "tab:blue"}
HM_MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (3, 1), (4, 4), (4, 3), (4, 2), (5, 5)]

nu_of = lambda q: q / (1.0 + q) ** 2
X1_of = lambda q: q / (1.0 + q)


def side_flux(t, modes, window_M=20.0, drop_head_M=300.0):
    """x(t) and envelope flux from ONE waveform's own data.  modes = {(l,m): h}."""
    t = np.asarray(t, float)
    dt = float(np.median(np.diff(t)))
    n = max(int(round(window_M / dt)) | 1, 7)
    h22 = modes[(2, 2)]
    psi = np.unwrap(np.angle(h22))
    om = np.abs(savgol_filter(psi, n, 3, deriv=1, delta=dt))
    x = (0.5 * om) ** (2.0 / 3.0)
    Om2 = (0.5 * om) ** 2
    F = np.zeros_like(t)
    for (l, m), h in modes.items():
        amp = savgol_filter(np.abs(h), n, 3)
        F += (m ** 2) * Om2 * amp ** 2
    t_pk = t[np.argmax(np.abs(h22))]
    msk = (t < t_pk - 50.0) & (t > t[0] + drop_head_M)
    xg, Fg = x[msk], F[msk]
    o = np.argsort(xg)
    return xg[o], Fg[o]


def measure_q22(q):
    d = np.load(ROOT / ".cache" / "q_dep" / f"waveforms_q{q:.10f}.npz")
    hp = {(2, 2): d["h_bhpt_re"] + 1j * d["h_bhpt_im"]}
    hn = {(2, 2): d["h_nr_re"] + 1j * d["h_nr_im"]}
    return (side_flux(d["t_bhpt"], hp, drop_head_M=100.0),
            side_flux(d["t_nr"], hn, drop_head_M=300.0))


def measure_qtot(q):
    d = np.load(ROOT / ".cache" / "hm" / f"waveforms_hm_q{q:.10f}.npz")
    hp = {lm: d[f"bhpt_{lm[0]}{lm[1]}_re"] + 1j * d[f"bhpt_{lm[0]}{lm[1]}_im"]
          for lm in HM_MODES}
    hn = {lm: d[f"nr_{lm[0]}{lm[1]}_re"] + 1j * d[f"nr_{lm[0]}{lm[1]}_im"]
          for lm in HM_MODES}
    return (side_flux(d["t_bhpt"], hp, drop_head_M=100.0),
            side_flux(d["t_nr"], hn, drop_head_M=300.0))


def r2_measured(q, pp, nr):
    (xp, Fp), (xn, Fn) = pp, nr
    lo = max(xp.min(), xn.min(), 0.04)
    hi = min(xp.max(), xn.max(), 0.155)
    xg = np.linspace(lo, hi, 250)
    R_raw = np.interp(xg, xn, Fn) / np.interp(xg, xp, Fp)
    X1, nu = X1_of(q), nu_of(q)
    conv = R_raw / X1 ** 4
    return xg, (conv - 1.0) / nu, conv


def one_plot(kind, qs, measure, theory_x, theory_R, theory_label, calib_x, calib_R,
             calib_label, pn_line, pn_label, fname, title):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    print(f"\n===== {kind} =====")
    print(f"{'q':>5} {'band':>16} {'conv@lo':>8} {'conv@hi':>8}   (F_NR/F_pp/X1^4; ~1 expected)")
    for q in qs:
        xg, R2, conv = r2_measured(q, *measure(q))
        print(f"{q:5g} [{xg.min():.3f},{xg.max():.3f}] {conv[0]:8.3f} {conv[-1]:8.3f}")
        ax.plot(xg, R2, color=QCOLS[q], lw=1.2,
                label=f"measured, $q={q:g}$  (no calibration)")
    ax.plot(theory_x, theory_R, color="tab:green", lw=2.2, marker="o", ms=3.5,
            zorder=5, label=theory_label)
    ax.plot(calib_x, calib_R, color="tab:red", lw=1.6, ls="--", zorder=4,
            label=calib_label)
    xr = np.linspace(0.02, 0.16, 100)
    ax.plot(xr, pn_line(xr), color="gray", ls=":", lw=1.0, label=pn_label)
    ax.axvline(0.13, color="0.6", ls="--", lw=0.8)
    ax.text(0.131, 0.97, "adiabatic band edge", rotation=90, fontsize=7,
            color="0.4", va="top", transform=ax.get_xaxis_transform())
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$F^{(2)}/F^{(1)}$")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = OUT / f"{fname}.{ext}"
        fig.savefig(p, dpi=130)
        print("wrote", p)


def main():
    # theory curves
    dw = json.loads((ROOT / "wardell_data" / "R2_22_from_amplitudes.json").read_text())
    d2 = json.loads((ROOT / "wardell_data" / "F2_EI_schwarz_circ.json").read_text())
    d1 = json.loads((ROOT / "wardell_data" / "F1_EI_schwarz_circ.json").read_text())
    x1 = 1.0 / np.array(d1["r0"]); F1 = np.array(d1["F1_EI"])
    o = np.argsort(x1)
    x2t = np.array(d2["x"]); F2t = np.array(d2["F2_EI"])
    m2 = (x2t >= x1.min()) & (x2t <= x1.max())
    xt, Rt = x2t[m2], F2t[m2] / np.interp(x2t[m2], x1[o], F1[o])

    # calibration-implied curves, for context
    import wardell_total_flux as WT
    xr = np.linspace(0.035, 0.155, 200)
    calib22 = WT.R2_lm(xr, 2, 2, False)
    xgt, calibtot = WT.build_total(8.0, False)

    one_plot("(2,2)", [3.0, 4.0, 5.0, 6.0, 8.0], measure_q22,
             np.array(dw["x"]), np.array(dw["R2_22"]),
             "2SF theory, $(2,2)$ (WaSABI amplitudes)",
             xr, calib22, "calibration-implied (pn_anchored)",
             lambda x: (55.0 / 21.0) * x, r"$+55x/21$",
             "F2_measured_22",
             "MEASURED $(2,2)$ second-order flux ratio — no calibration")

    one_plot("total", [3.0, 4.0, 5.0, 8.0], measure_qtot,
             xt, Rt, "2SF theory, TOTAL flux (Warburton)",
             xgt, calibtot, "backbone-implied total",
             lambda x: -(35.0 / 12.0) * x, r"$-35x/12$",
             "F2_measured_total",
             "MEASURED TOTAL second-order flux ratio — no calibration")


if __name__ == "__main__":
    main()
