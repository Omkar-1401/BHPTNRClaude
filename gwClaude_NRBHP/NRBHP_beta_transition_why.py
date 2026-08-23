"""Companion to beta_transition_q: WHY the crossing sits near q=4.

Left  -- the fitted total drift dbeta(q) against the DERIVED anchor gap gb(q).  gb stays
         positive at every q, so the crossing is the fit falling short of the QNM plateau,
         not the physics changing sign.  b_E = 0 is a level the objective never sees.
Right -- the inspiral's and merger's SEPARATE preferences for b_E (exact additive split of
         the mismatch numerator).  They have the same sign at every q and flip together,
         so the crossing is NOT a tug-of-war between the two regions.

Data: beta_drift_tests/{gb_vs_fitted_drift,transition_q_decompose_flux_-200}.json
"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
mpl.rcdefaults(); mpl.rcParams["text.usetex"] = False
D = ROOT / "beta_drift_tests"
g = json.loads((D / "gb_vs_fitted_drift.json").read_text())
d = json.loads((D / "transition_q_decompose_flux_-200.json").read_text())

q = np.array([r["q"] for r in g]); gb = np.array([r["gb_pct"] for r in g])
df = np.array([r["dbeta_flux_pct"] for r in g]); de = np.array([r["dbeta_E_pct"] for r in g])
qd = np.array([float(k) for k in d if k[0].isdigit()])
bi = np.array([d[k]["b_ins"] for k in d if k[0].isdigit()])
bm = np.array([d[k]["b_mrg"] for k in d if k[0].isdigit()])

def cross(x, y):
    s = np.where(np.diff(np.sign(y)))[0]
    if not len(s): return np.nan
    i = s[0]; return x[i] + (x[i+1]-x[i])*(-y[i])/(y[i+1]-y[i])

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
a = ax[0]
a.axhline(0, color="0.5", lw=1.0)
a.plot(q, gb, "o-", color="tab:purple", lw=2.0,
       label=r"derived anchor gap $g_b=(B_1-B_0)/B_0$")
a.plot(q, df, "s-", color="tab:blue", lw=1.8, label=r"fitted $\Delta\beta$ (Edot drive)")
a.plot(q, de, "^--", color="tab:red", lw=1.4, label=r"fitted $\Delta\beta$ (E drive)")
a.fill_between(q, df, gb, color="0.85", zorder=0, label="deficit (2.2-4.3 pp)")
qc = cross(q, df)
a.plot([qc], [0], "k*", ms=14, zorder=5)
a.annotate(f"q = {qc:.2f}\n$g_b$ = {np.interp(qc, q, gb):.1f}%", (qc, 0),
           textcoords="offset points", xytext=(10, 14), fontsize=9)
a.set_xlabel("q"); a.set_ylabel(r"total drift of $\beta$ over the window  [%]")
a.set_title(r"$\beta$'s drift vs its derived endpoints")
a.legend(fontsize=8.5); a.grid(alpha=0.35)

b = ax[1]
b.axhline(0, color="0.5", lw=1.0)
b.plot(qd, bi, "o-", color="tab:green", lw=1.8, label=r"inspiral only ($t_{NR}\leq-200$)")
b.plot(qd, bm, "s-", color="tab:orange", lw=1.8, label=r"merger+ringdown ($t_{NR}>-200$)")
for y, c, nm in ((bi, "tab:green", "inspiral"), (bm, "tab:orange", "merger")):
    x0 = cross(qd, y); b.plot([x0], [0], "*", color=c, ms=13, zorder=5)
    b.annotate(f"{x0:.2f}", (x0, 0), textcoords="offset points",
               xytext=(-4, -16 if nm == "merger" else 8), fontsize=9, color=c)
b.set_xlabel("q"); b.set_ylabel(r"preferred $b_E$  (other 3 params held)")
b.set_title("the two regions flip TOGETHER, not against each other")
b.legend(fontsize=8.5); b.grid(alpha=0.35)
fig.tight_layout()
for e in ("pdf", "png"):
    p = ROOT / "Agentic_plots" / f"beta_transition_why.{e}"; fig.savefig(p); print("wrote", p)
