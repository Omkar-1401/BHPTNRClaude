"""alpha and beta vs x, one curve per q -- the MEASURED (parameter-free) curves.

Parametric in t: for each q, alpha(t) and beta(t) are plotted against x(t), with t the
joint parameter, so each curve traces the inspiral from low x up to merger.

Source: beta_drift_tests/alpha_beta_of_omega_v2.json -- beta = omega_pp/omega_NR and
alpha = |h_NR|/|h_pp| at points paired by equal accumulated phase measured back from
merger (merger alignment).  Nothing fitted.  Matched savgol smoothing in M on both sides.

House format: 1 x 2, figsize (12,4), q=2/3/5/8 in tab:red/orange/green/blue with lw 2.2
for q=2; the intermediate q=2.5/4/6 are drawn thin grey so nothing is hidden.  Dashed
horizontals in the beta panel mark X1^(6/5), the derived early-inspiral anchor.

Usage:  python NRBHP_alpha_beta_vs_x.py [window_M]      (10 | 20 | 40, default 20)
"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np

mpl.rcdefaults(); mpl.rcParams["text.usetex"] = False
ROOT = Path(__file__).resolve().parent
WIN = sys.argv[1] if len(sys.argv) > 1 else "20"
d = json.loads((ROOT/"beta_drift_tests"/"alpha_beta_of_omega_v2.json").read_text())[WIN]
PLOT_DIR = ROOT/"Agentic_plots"/"alpha_beta_of_x"; PLOT_DIR.mkdir(parents=True, exist_ok=True)

MAIN = {"2.0": "tab:red", "3.0": "tab:orange", "5.0": "tab:green", "8.0": "tab:blue"}
SIDE = ["2.5", "4.0", "6.0"]
X_ISCO = 1.0/6.0

fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4))
for k in SIDE:                                   # intermediate q, thin grey
    r = d[k]; x = np.array(r["x"])
    axa.plot(x, r["alpha"], color="0.65", lw=0.8, zorder=1, label=f"q={float(k):g}")
    axb.plot(x, r["beta"],  color="0.65", lw=0.8, zorder=1, label=f"q={float(k):g}")
for k, c in MAIN.items():
    r = d[k]; x = np.array(r["x"]); lw = 2.2 if k == "2.0" else 1.3
    axa.plot(x, r["alpha"], color=c, lw=lw, zorder=3, label=f"q={float(k):g}")
    axb.plot(x, r["beta"],  color=c, lw=lw, zorder=3, label=f"q={float(k):g}")
    axb.axhline(r["anchor"], color=c, lw=0.7, ls=(0, (4, 3)), zorder=2)
for ax, yl, tl in ((axa, "alpha", "alpha vs x   (parametric in t)"),
                   (axb, "beta",  "beta vs x    (dashed: X1^(6/5) anchor)")):
    ax.axvline(X_ISCO, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("x = (M Omega_orb)^(2/3)"); ax.set_ylabel(yl)
    ax.set_title(tl); ax.grid(True); ax.legend(fontsize=8, ncol=2)
fig.suptitle(f"measured alpha, beta vs x  (merger-aligned, nothing fitted; "
             f"smoothing {WIN} M)")
fig.tight_layout()
for ext in ("pdf", "png"):
    p = PLOT_DIR/f"alpha_beta_vs_x_win{WIN}.{ext}"; fig.savefig(p); print("wrote", p)
plt.close(fig)
