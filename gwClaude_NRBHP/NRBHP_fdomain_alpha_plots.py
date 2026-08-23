"""
Alpha(f) and beta(f) overlay for the fdomain_alpha model.

x-axis: x = (pi * f_GW)^{2/3}  (SPA orbital-frequency coordinate)

alpha(x, q) = a_PP(nu) * g(z(x), nu)  -- direct function of x
beta(x, q)  = b_PP * (1 + b_E * E(t)) -- parametric: x(t_pp) vs beta(t_pp)

Both panels restricted to the inspiral band x in [X_LO, X_HI].

Output: Agentic_plots/fdomain_alpha/fdomain_alpha_alpha_beta_overlay.pdf

Usage:
    conda run -n ut_claude python NRBHP_fdomain_alpha_plots.py
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import fit_scaling_PN_opt_creative_q_dep as qdep
import fit_scaling_gw_remnant_energy     as G
import fit_scaling_fdomain_alpha         as FD

PLOT_DIR = ROOT / "Agentic_plots" / "fdomain_alpha"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

Q_INPUTS = [2.0, 3.0, 5.0, 8.0]
COLS     = {2.0: "tab:red", 3.0: "tab:orange", 5.0: "tab:green", 8.0: "tab:blue"}

# ------------------------------------------------------------------ load
fd_c        = json.loads((ROOT / "fdomain_alpha_results" / "coeffs.json").read_text())
mult_cache  = json.loads((G.RESULTS_DIR / "per_q_cache_mult.json").read_text())
a_PP_coeffs = np.array(fd_c["a_PP_coeffs"])
g_coeffs    = np.array(fd_c["g_coeffs"])


def get_beta_params(q):
    key = min(mult_cache.keys(), key=lambda k: abs(float(k) - q))
    if abs(float(key) - q) < 0.05:
        p = mult_cache[key]["params"]
        return [p[2], p[3], p[4]]
    print(f"  q={q}: optimizing per-q mult for beta ...", flush=True)
    res = G.optimize_case(q, form="mult")
    p   = res["params"]
    return [float(p[2]), float(p[3]), float(p[4])]


def build_model(q):
    nu = FD.nu_of(q)
    beta_params = get_beta_params(q)
    b_PP, b_E, _ = beta_params

    d    = np.load(qdep.waveform_cache_path(q))
    t_pp = d["t_bhpt"];  h_pp = d["h_bhpt_re"] + 1j * d["h_bhpt_im"]

    E      = G.gwr_energy_coordinates(t_pp, h_pp, q)["e_oft"]
    x_t    = FD.instantaneous_x(t_pp, h_pp)
    beta_t = b_PP * (1.0 + b_E * E)

    # alpha: evaluate directly on a dense x grid (it IS a function of x)
    x_grid  = np.linspace(FD.X_LO, FD.X_HI, 500)
    z_grid  = FD.x_to_z(x_grid)
    c_hat   = FD.eval_c_poly_z(nu, g_coeffs)
    alpha_x = FD.eval_a_PP(nu, a_PP_coeffs) * np.polyval(c_hat, z_grid)

    # beta: parametric — restrict to inspiral band
    band   = (x_t >= FD.X_LO) & (x_t <= FD.X_HI)
    x_beta = x_t[band]
    b_beta = beta_t[band]
    # sort by x for clean line plot
    order  = np.argsort(x_beta)

    err = FD.mismatch_fdomain(q, nu, a_PP_coeffs, g_coeffs, beta_params)

    return dict(x_alpha=x_grid, alpha=alpha_x,
                x_beta=x_beta[order], beta=b_beta[order],
                error=err)


# ------------------------------------------------------------------ build
print("Building models ...")
results = {}
for q in Q_INPUTS:
    print(f"  q={q}", flush=True)
    results[q] = build_model(q)
    print(f"    E = {results[q]['error']:.4e}")


# ------------------------------------------------------------------ plot
plt.rcdefaults()
fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4))

for q in Q_INPUTS:
    r   = results[q]
    col = COLS[q]
    lw  = 2.2 if q == 2.0 else 1.3
    extrap_tag = " [extrap]" if q == 2.0 else ""
    axa.plot(r["x_alpha"], r["alpha"], color=col, lw=lw,
             label=f"q={q:.0f}{extrap_tag}  E={r['error']:.2e}")
    axb.plot(r["x_beta"],  r["beta"],  color=col, lw=lw,
             label=f"q={q:.0f}")

for ax, yl, tl in [(axa, r"$\alpha(x)$", r"$\alpha(x)$ — master"),
                   (axb, r"$\beta(x)$",  r"$\beta(x)$ — master")]:
    ax.set_xlabel(r"$x = (\pi M f_\mathrm{GW})^{2/3}$")
    ax.set_ylabel(yl)
    ax.set_title(tl)
    ax.axvline(FD.X_LO, color="gray", ls=":", lw=0.8)
    ax.axvline(FD.X_HI, color="gray", ls=":", lw=0.8)
    ax.legend(fontsize=8)
    ax.grid(True)

fig.suptitle("fdomain-alpha: master alpha/beta vs orbital frequency")
fig.tight_layout()

out = PLOT_DIR / "fdomain_alpha_alpha_beta_overlay.pdf"
fig.savefig(out); plt.close(fig)
print(f"\nSaved: {out}")
