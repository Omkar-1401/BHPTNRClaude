"""
Overlay mismatch vs q for four models:
  - fdomain_alpha         (18-coef, frequency-domain alpha + time-domain beta)
  - gwr_energy_stiff      (global, 9-coef, E-coupled, no PN)
  - gwr_energy_anchored   (global, 6-coef, E-coupled, X1^6/5 anchored)
  - gwr_energy_fluxanchored seeded (7-coef, flux-coupled, best at q<3)

Vertical dashed line at q=3 (training boundary for all models).
In-range q plotted as solid curves/filled markers; held-out q<3 as open markers.

Usage:
  conda run -n ut_claude python plot_fdomain_alpha_overlay.py [--noeval]

  --noeval  skip re-evaluation, use only stored JSON values (faster, fewer points)
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import fit_scaling_gw_remnant_energy      as G
import fit_scaling_gwr_energy_stiff       as STIFF
import fit_scaling_gwr_energy_anchored    as ANC
import fit_scaling_gwr_energy_fluxanchored as FA
import fit_scaling_gwr_energy_global      as GG
import fit_scaling_gwr_energy_flux        as FL

# ------------------------------------------------------------------ constants
IN_Q   = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
LOW_Q  = [2.75, 2.5, 2.25, 2.0]

# ------------------------------------------------------------------ load coefficients
stiff_c  = json.loads((ROOT / "gwr_energy_stiff_results"       / "coeffs.json").read_text())
anc_c    = json.loads((ROOT / "gwr_energy_anchored_results"    / "coeffs.json").read_text())
fa_c     = json.loads((ROOT / "gwr_energy_fluxanchored_results"/ "coeffs.json").read_text())
fd_c     = json.loads((ROOT / "fdomain_alpha_results"           / "coeffs.json").read_text())
mult_cache = json.loads((ROOT / "gw_remnant_energy_results" / "per_q_cache_mult.json").read_text())


def t0_seed(q):
    """Nearest-q t0_nr from the mult per-q cache as a starting point."""
    key = min(mult_cache.keys(), key=lambda k: abs(float(k) - q))
    return float(mult_cache[key]["params"][4])


# ------------------------------------------------------------------ stiff (stored per-q + global low-q)
stiff_ref = stiff_c["_meta"]["reference_mult_14coef"]   # {q_str: error}
stiff_in  = {float(k): v for k, v in stiff_ref.items() if float(k) >= 3.0}
stiff_low = {float(k): v["stiff_global"]
             for k, v in stiff_c["_meta"]["low_q"].items()}


# ------------------------------------------------------------------ anchored (evaluate at IN_Q, stored at LOW_Q)
def anc_mismatch(q):
    aE_deg = int(anc_c["alphaE_degree"])
    c = np.array(anc_c["c"]); A = np.array(anc_c["A"])
    b = float(anc_c["b"]);    P = np.array(anc_c["P"])
    theta = np.concatenate([c, A, [b], P])
    p = ANC.params_at(q, theta, aE_deg)
    case = G.load_case(q)
    _, t0 = GG.fast_mismatch(p[:4], case, t0_seed(q), "fit")
    e, _  = GG.fast_mismatch(p[:4], case, t0,         "full")
    return e


# ------------------------------------------------------------------ fluxanchored seeded (evaluate at IN_Q, stored at LOW_Q)
def fa_mismatch(q):
    s      = fa_c["seeded"]
    aF_deg = len(s["A"])
    c = np.array(s["c"]); A = np.array(s["A"])
    b = float(s["b"]);    P = np.array(s["P"])
    theta = np.concatenate([c, A, [b], P])
    p = FA.params_at(q, theta, aF_deg)
    case = FL.add_flux(G.load_case(q))
    _, t0 = FL.fast_mismatch(p, case, t0_seed(q), "fit")
    e, _  = FL.fast_mismatch(p, case, t0,         "full")
    return e


# ------------------------------------------------------------------ fdomain (all stored)
fd_by_q = fd_c["in_range"]["by_q"]   # {"q": error}
fd_in   = {float(k): v for k, v in fd_by_q.items()}
fd_low  = {float(k): v for k, v in fd_c["low_q"].items()}


# ------------------------------------------------------------------ evaluate
ap = argparse.ArgumentParser()
ap.add_argument("--noeval", action="store_true")
args = ap.parse_args()

if not args.noeval:
    print("Loading waveforms and evaluating anchored + fluxanchored at IN_Q ...")
    anc_in  = {}
    fa_in   = {}
    for q in IN_Q:
        print(f"  q={q}", end="  ", flush=True)
        anc_in[q] = anc_mismatch(q)
        fa_in[q]  = fa_mismatch(q)
        print(f"anc={anc_in[q]:.3e}  fa={fa_in[q]:.3e}")
else:
    anc_in = {}; fa_in = {}

anc_low = {float(k): v for k, v in anc_c["low_q"].items()}
fa_low  = {float(k): v for k, v in fa_c["seeded"]["low_q"].items()}

# ------------------------------------------------------------------ plot
plt.rcdefaults()
fig, ax = plt.subplots(figsize=(9, 5.5))

COLOR = {"fd": "C0", "stiff": "C2", "anc": "C1", "fa": "C3"}

# -- fdomain: continuous curve over all 64 training q + open markers at low-q
fd_qs = sorted(fd_in.keys())
ax.plot(fd_qs, [fd_in[q] for q in fd_qs], "-", lw=1.6,
        color=COLOR["fd"], label="fdomain_alpha (18-coef)", zorder=5)
ax.plot(sorted(fd_low), [fd_low[q] for q in sorted(fd_low)],
        "o", mfc="none", mec=COLOR["fd"], ms=7, mew=1.4, zorder=5)

# -- stiff: 6 filled markers in-range + 4 open at low-q
ax.plot(sorted(stiff_in), [stiff_in[q] for q in sorted(stiff_in)],
        "s-", color=COLOR["stiff"], ms=5, lw=1.4,
        label="stiff_global (9-coef)", zorder=4)
ax.plot(sorted(stiff_low), [stiff_low[q] for q in sorted(stiff_low)],
        "s", mfc="none", mec=COLOR["stiff"], ms=7, mew=1.4, zorder=4)

# -- anchored
if anc_in:
    ax.plot(sorted(anc_in), [anc_in[q] for q in sorted(anc_in)],
            "^-", color=COLOR["anc"], ms=5, lw=1.4,
            label="anchored (6-coef)", zorder=3)
ax.plot(sorted(anc_low), [anc_low[q] for q in sorted(anc_low)],
        "^", mfc="none", mec=COLOR["anc"], ms=7, mew=1.4, zorder=3)

# -- fluxanchored seeded
if fa_in:
    ax.plot(sorted(fa_in), [fa_in[q] for q in sorted(fa_in)],
            "D-", color=COLOR["fa"], ms=5, lw=1.4,
            label="fluxanchored seeded (7-coef)", zorder=3)
ax.plot(sorted(fa_low), [fa_low[q] for q in sorted(fa_low)],
        "D", mfc="none", mec=COLOR["fa"], ms=7, mew=1.4, zorder=3)

# -- decorations
ax.axvline(3.0, color="k", lw=0.9, ls="--", alpha=0.45)
ax.text(3.08, 1.7e-4, "training\nboundary", fontsize=7, va="bottom", color="0.45")

for val, lab in [(1e-3, "$10^{-3}$"), (1e-4, "$10^{-4}$")]:
    ax.axhline(val, color="0.78", lw=0.7, ls=":", zorder=0)
    ax.text(8.12, val, lab, fontsize=7.5, va="center", color="0.5")

ax.set_yscale("log")
ax.set_xlabel("mass ratio  $q$", fontsize=11)
ax.set_ylabel(r"mismatch  $\mathcal{E}$", fontsize=11)
ax.set_title(
    "Frequency-domain alpha model vs time-domain benchmarks  (22 mode)\n"
    r"open markers = held-out $q<3$;  solid markers / curve = training range $q\in[3,8]$",
    fontsize=9.5)
ax.set_xlim(1.75, 8.5)
ax.set_ylim(1.2e-4, 1.5e-2)
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
ax.grid(True, which="both", alpha=0.18)

plt.tight_layout()
out = ROOT / "Agentic_plots" / "fdomain_alpha_overlay.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=150)
print(f"\nSaved: {out}")
