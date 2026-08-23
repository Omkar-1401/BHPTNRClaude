"""Where does beta's drift change sign?  One point per model.

beta = beta_PP*(1 + b_E*E(t)) with E monotone, so sign(dbeta/dt) = sign(P), P = beta_PP*b_E.
The "transition q" is where P = 0, i.e. where beta is flat over the window.  For master
models P(nu) is a polynomial and the root is solved directly; for per-q grids the measured
P is interpolated to its zero crossing.

Cross-model summary, so it lives at the Agentic_plots root rather than in one model's folder.
"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
mpl.rcdefaults(); mpl.rcParams["text.usetex"] = False
NU = lambda q: q/(1+q)**2
OUT = ROOT/"Agentic_plots"
rows = []   # (label, q_star, kind)

def root_of(f, lo=2.0, hi=10.0):
    try:
        return float(brentq(f, lo, hi))
    except ValueError:
        return None

# --- master models with a LINEAR P(nu) -------------------------------------------
for lab, path, key in (
        ("anchored (6c, global)", "gwr_energy_anchored_results/coeffs.json", "P"),
        ("fluxanchored (7c, global)", "gwr_energy_fluxanchored_results/coeffs.json", "P"),
        ("stiff (9c, global)", "gwr_energy_stiff_results/coeffs.json", "P")):
    d = json.loads((ROOT/path).read_text())
    P = np.asarray(d[key], float)
    r = root_of(lambda q: P[0] + P[1]*NU(q))
    if r: rows.append((lab, r, "master"))
    if "seeded" in d and key in d["seeded"]:
        Ps = np.asarray(d["seeded"][key], float)
        r = root_of(lambda q: Ps[0] + Ps[1]*NU(q))
        if r: rows.append((lab.replace("global", "seeded"), r, "master"))

# --- master models with a deg-3 P(nu) -------------------------------------------
for lab, path, sub in (("E deg-3 (14c, global)", "gwr_energy_global_results/coeffs.json", None),
                       ("flux deg-3 (14c, global)", "gwr_energy_flux_results/coeffs.json", "coeffs")):
    d = json.loads((ROOT/path).read_text())
    c = d[sub] if sub else d
    bp = np.asarray(c["beta_PP"], float); be = np.asarray(c["beta_E"], float)
    ev = lambda co, q: sum(co[i]*NU(q)**i for i in range(len(co)))
    r = root_of(lambda q: ev(bp, q)*ev(be, q))
    if r: rows.append((lab, r, "master"))

# --- per-q measured grids -------------------------------------------------------
for lab, path in (("per-q grid, E drive (64 q)", "gw_remnant_energy_results/per_q_cache_mult.json"),
                  ("per-q grid, Edot drive (16 q)", "gwr_energy_flux_results/per_q_cache_flux.json")):
    c = json.loads((ROOT/path).read_text())
    v = sorted(((x["q"], x["params"][2]*x["params"][3]) for x in c.values()))
    q = np.array([a for a, _ in v]); P = np.array([b for _, b in v])
    s = np.where(np.diff(np.sign(P)))[0]
    if len(s):
        i = s[0]; r = q[i] + (q[i+1]-q[i])*(-P[i])/(P[i+1]-P[i])
        rows.append((lab, float(r), "per-q"))

# --- constrained / anchored constructions ---------------------------------------
d = json.loads((ROOT/"gwr_beta_monotone_results"/"coeffs.json").read_text())
r = root_of(lambda q: NU(q) - d["P_nu_c"], 2.0, 40.0)
if r: rows.append(("beta_monotone (P>=0, flat above)", r, "constrained"))

rows.sort(key=lambda z: z[1])
print(f"{'model':38s} {'q*':>7s}  kind")
for lab, r, k in rows: print(f"{lab:38s} {r:7.3f}  {k}")

COL = {"master": "tab:blue", "per-q": "tab:red", "constrained": "tab:green"}
MRK = {"master": "o", "per-q": "s", "constrained": "^"}
fig, ax = plt.subplots(figsize=(9.5, 4.8))
y = np.arange(len(rows))
for i, (lab, r, k) in enumerate(rows):
    ax.plot(r, i, MRK[k], color=COL[k], ms=9, zorder=3)
    ax.annotate(f"{r:.2f}", (r, i), textcoords="offset points", xytext=(9, -3),
                fontsize=8, color=COL[k])
ms = [r for _, r, k in rows if k != "constrained"]
ax.axvspan(min(ms), max(ms), color="0.9", zorder=0,
           label=f"fitted models: {min(ms):.2f}-{max(ms):.2f}")
ax.axvline(float(np.mean(ms)), color="0.4", ls="--", lw=1.0, zorder=1,
           label=f"mean {np.mean(ms):.2f}")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
ax.set_xlabel("transition q  (where beta is flat: P = 0)")
ax.set_title("Where beta's drift changes sign, by model")
ax.grid(True, axis="x", alpha=0.4)
for k in COL:
    if any(kk == k for _, _, kk in rows):
        ax.plot([], [], MRK[k], color=COL[k], ms=8, label=k)
ax.legend(fontsize=8, loc="center right")
ax.set_xlim(3.7, 6.35)
ax.set_ylim(-0.8, len(rows) - 0.2)
ax.text(0.24, 0.06, "measured beta (parameter-free, merger-aligned)\n"
        "RISES at every q in [2,8] -- no transition at all",
        transform=ax.transAxes, fontsize=8.5, color="tab:purple", va="bottom",
        bbox=dict(fc="white", ec="tab:purple", alpha=0.9, lw=0.8))
fig.tight_layout()
for e in ("pdf", "png"):
    p = OUT/f"beta_transition_q.{e}"; fig.savefig(p); print("wrote", p)
plt.close(fig)
