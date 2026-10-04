"""Plots + band analysis for STEP 1 of the spin extension (spin_measure_alpha_beta.py).

Q1 time-independence: R(x) = A(x;q,chi)/A(x;q,0) flat in x?
Q2 separability:      R independent of q?  (Islam Eq 8/9 predict a specific mild
                      q-dependence, overlaid.)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
mpl.rcdefaults(); mpl.rcParams["text.usetex"] = False
RES = ROOT / "spin_measure_results"
OUT = ROOT / "Agentic_spin_plots"; OUT.mkdir(parents=True, exist_ok=True)

QS = [3.0, 4.0, 5.0, 6.0, 8.0]
CHIS = [-0.6, -0.3, 0.3, 0.6, 0.8]
CCOL = {-0.6: "tab:blue", -0.3: "tab:cyan", 0.3: "tab:olive",
        0.6: "tab:orange", 0.8: "tab:red"}
QLS = {3.0: "-", 4.0: "--", 5.0: "-.", 6.0: ":", 8.0: (0, (3, 1, 1, 1))}
X0, X1B = 0.05, 0.10          # clean band: away from the NR start and from merger


def ratio(data, q, chi, key, xlo=X0, xhi=X1B, n=160):
    b, d = data[f"{q:.1f}_+0.00"], data[f"{q:.1f}_{chi:+.2f}"]
    xb, yb = np.array(b["x"]), np.array(b[key]); ob = np.argsort(xb)
    xd, yd = np.array(d["x"]), np.array(d[key]); od = np.argsort(xd)
    lo = max(xb.min(), xd.min(), xlo); hi = min(xb.max(), xd.max(), xhi)
    xg = np.linspace(lo, hi, n)
    return xg, np.interp(xg, xd[od], yd[od]) / np.interp(xg, xb[ob], yb[ob])


def islam(coef, q, chi):
    A, B = coef[:4], coef[4:]
    P = sum(A[n] * q ** -(n + 1) for n in range(4))
    S = sum(B[n] * chi ** (n + 1) for n in range(2))
    return (1 + P * (1 + S)) / (1 + P)


def main():
    import h5py
    data = json.loads((RES / "curves_win20.json").read_text())
    with h5py.File("/work2/11702/omkarsridhar1401/stampede3/BHPTNRSurrogate/data/"
                   "BHPTNRSur2dq1e3.h5", "r") as f:
        g = f["nr_calib_params"]["(2,2)"]
        CO = {"alpha": np.array(g["alpha"]), "beta": np.array(g["beta"])}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, key in enumerate(("alpha", "beta")):
        axL, axR = axes[row]
        # --- Q1: R vs x
        for q in QS:
            for c in CHIS:
                xg, R = ratio(data, q, c, key, 0.045, 0.125)
                axL.plot(xg, R, color=CCOL[c], ls=QLS[q], lw=1.0,
                         label=f"$\\chi_1={c:+.1f}$" if q == 3.0 else None)
        axL.axhline(1.0, color="k", lw=0.8, ls=":")
        axL.axvspan(X0, X1B, color="0.9", zorder=0)
        axL.set_xlabel("$x$"); axL.set_ylabel(f"$R_\\{key}(x)$")
        axL.set_title(f"Q1 time-independence: $R_\\{key}$ vs $x$\n"
                      "(colour = $\\chi_1$, linestyle = $q$; grey = clean band)",
                      fontsize=9)
        axL.grid(alpha=0.4); axL.legend(fontsize=7, ncol=2, loc="lower left")
        # --- Q2: band-mean R vs chi
        for q in QS:
            m = [np.mean(ratio(data, q, c, key)[1]) for c in CHIS]
            axR.plot(CHIS, m, "o-", ms=4, lw=1.3, label=f"measured $q={q:g}$")
            axR.plot(CHIS, [islam(CO[key], q, c) for c in CHIS], "s--", ms=3, lw=0.9,
                     alpha=0.55, color=axR.lines[-1].get_color(),
                     label="Islam Eq.8/9" if q == 3.0 else None)
        axR.axhline(1.0, color="k", lw=0.8, ls=":")
        axR.set_xlabel("$\\chi_1$"); axR.set_ylabel(f"band-mean $R_\\{key}$")
        axR.set_title(f"Q2 separability: $R_\\{key}$ vs $\\chi_1$ per $q$\n"
                      f"(solid = measured, dashed = Islam Eq. 8/9)", fontsize=9)
        axR.grid(alpha=0.4); axR.legend(fontsize=7, ncol=2)
    fig.suptitle("Spin dependence of the calibration, measured parameter-free "
                 "(ppBHPT 2d vs NRHybSur3dq8)", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = OUT / f"spin_ratio_tests.{ext}"; fig.savefig(p, dpi=130); print("wrote", p)

    # --- numbers on the clean band
    print(f"\nCLEAN BAND x in [{X0}, {X1B}]:  band-mean R, x-spread, vs Islam")
    for key in ("alpha", "beta"):
        print(f"\n===== {key} =====")
        print(f"{'q':>4} {'chi1':>6} {'mean R':>9} {'x-spread':>9} {'Islam':>8} "
              f"{'meas-1 / Islam-1':>17}")
        for q in QS:
            for c in CHIS:
                xg, R = ratio(data, q, c, key)
                mR, sp = np.mean(R), (R.max() - R.min()) / abs(np.mean(R)) * 100
                iv = islam(CO[key], q, c)
                rr = (mR - 1) / (iv - 1) if abs(iv - 1) > 1e-6 else np.nan
                print(f"{q:>4g} {c:>+6.2f} {mR:9.5f} {sp:8.2f}% {iv:8.5f} {rr:17.2f}")


if __name__ == "__main__":
    main()
