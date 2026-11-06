"""slim split, Ori–Thorne window start T = -1 (results/explore/slim.json) vs T = -2 (slim__win-2.0.json).
Left: blend S, alpha, beta on the NR clock at q = 5 for chi1 = -0.6, 0, +0.6.  Right: H270 error ratio T-2 / T-1.
-> Agentic_spin_split_plots/explore/window_T-2_vs_T-1.{png,pdf}"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import explore_split as X
import matplotlib; matplotlib.use("Agg"); matplotlib.rcdefaults()
import matplotlib.pyplot as plt
R = {w: json.loads((X.OUT / f"{n}.json").read_text()) for w, n in ((-1.0, "slim"), (-2.0, "slim__win-2.0"))}
cases = [(5.0, -0.6), (5.0, 0.0), (5.0, 0.6)]
fig = plt.figure(figsize=(15, 9)); gs = fig.add_gridspec(3, 4)
for i, k in enumerate(cases):
    c = X.M.prepare(*k); X.M.CASES[k] = c
    tpk = c["t_bhpt"][int(np.argmax(np.abs(c["h_bhpt"])))]
    tt = X.SL.M.D.peak(c["t_nr"], c["h_nr"])[0] + (k[0] / (1 + k[0])) ** 1.2 * (c["t_bhpt"] - tpk)   # LO map to NR clock, peaks aligned
    axs, axa, axb = fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1]), fig.add_subplot(gs[i, 2])
    for w, ls in ((-1.0, "-"), (-2.0, "--")):
        X.set_window(w); d = R[w]
        al, be = X.ab("split", np.array(d["theta_insp"]), ("form", "slim", d["x_mr"]), *k, c, "full")
        lab = f"T = {w:+.0f}"
        axs.plot(tt, c["S"], ls, label=lab); axa.plot(tt, al, ls, label=lab); axb.plot(tt, be, ls, label=lab)
    for ax, nm in zip(fig.axes[-3:], ("S (MR weight)", r"$\alpha$", r"$\beta$")):
        ax.set_xlim(-400, 80); ax.grid(alpha=.3); ax.set_ylabel(nm); ax.set_title(f"q = {k[0]:g}, $\\chi_1$ = {k[1]:+.1f}", fontsize=9)
        if i == 2: ax.set_xlabel(r"$t_{\rm NR}$ / M (peak at 0)")
    fig.axes[-3].legend(fontsize=8)
ax = fig.add_subplot(gs[:, 3])
ea = {(r["q"], r["chi1"]): r["full_E_full"] for r in R[-1.0]["cases"]["H270"]}
eb = {(r["q"], r["chi1"]): r["full_E_full"] for r in R[-2.0]["cases"]["H270"]}
qs = sorted({k[0] for k in ea}); cs = sorted({k[1] for k in ea})
Z = np.array([[eb[(q, cc)] / ea[(q, cc)] for q in qs] for cc in cs])
im = ax.pcolormesh(qs, cs, Z, cmap="RdBu_r", vmin=0.5, vmax=1.5, shading="nearest")
fig.colorbar(im, ax=ax, label=r"$\mathcal{E}(T=-2)\,/\,\mathcal{E}(T=-1)$")
ax.set_xlabel("q"); ax.set_ylabel(r"$\chi_1$"); ax.set_title(f"H270: better at {int((Z < 1).sum())}/{Z.size}", fontsize=10)
fig.suptitle("slim split (14c): Ori–Thorne window from T = -1 (solid) vs T = -2 (dashed)")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(HERE / "Agentic_spin_split_plots" / "explore" / f"window_T-2_vs_T-1.{e}", dpi=130)
