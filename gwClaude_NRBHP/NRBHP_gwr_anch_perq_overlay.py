"""PER-Q alpha/beta overlay for the energy-anchored model, vs t.

Uses the PER-Q OPTIMAL parameters, not the master nu-fit -- so this isolates the FORM's
ability to describe each q, with no regression error folded in.  The gap between this and
`gwr_anch_alpha_beta_overlay.pdf` (master) is the regression error.

    alpha(t) = a_PP * (1 + a_E*E(t))        per-q [a_PP, a_E, b_PP, b_E, t0] from
    beta (t) = b_PP * (1 + b_E*E(t))        gw_remnant_energy_results/per_q_cache_mult.json
                                            (the `anchored` model shares the `mult` per-q model)

The cache covers q in [3,8] only, so q < 3 is fitted fresh here with G.optimize_case.

Style matches NRBHP_alpha_beta_vs_x.py: q=2/3/5/8 in house colours (lw 2.2 for q=2),
intermediates q=2.5/4/6 thin grey, dashed X1^(6/5) in the beta panel, xlim (-1000, 80).
rcdefaults() AFTER the model imports.

Usage:  python NRBHP_gwr_anch_perq_overlay.py
"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_global as GG
import fit_scaling_gwr_energy_flux as FL

mpl.rcdefaults(); mpl.rcParams["text.usetex"] = False

PLOT_DIR = ROOT/"Agentic_plots"/"gwr_energy_anchored"; PLOT_DIR.mkdir(parents=True, exist_ok=True)
PERQ_EXTRA = ROOT/"gwr_energy_anchored_results"/"per_q_lowq_mult.json"
MAIN = {2.0: "tab:red", 3.0: "tab:orange", 5.0: "tab:green", 8.0: "tab:blue"}
SIDE = [2.5, 4.0, 6.0]
QS = sorted(list(MAIN) + SIDE)

cache = json.loads((G.RESULTS_DIR/"per_q_cache_mult.json").read_text())
extra = json.loads(PERQ_EXTRA.read_text()) if PERQ_EXTRA.exists() else {}

def perq_params(q):
    hit = min(cache.values(), key=lambda v: abs(v["q"] - q))
    if abs(hit["q"] - q) < 1e-6:
        return np.asarray(hit["params"], float), hit["err"] if "err" in hit else hit["error"]
    k = f"{q:.4f}"
    if k not in extra:                                  # q < 3: fit it fresh
        print(f"   per-q fit at q={q:g} (not in the [3,8] cache) ...", flush=True)
        r = G.optimize_case(q, form="mult")
        extra[k] = {"params": r["params"], "error": r["error"]}
        PERQ_EXTRA.write_text(json.dumps(extra, indent=1))
    return np.asarray(extra[k]["params"], float), extra[k]["error"]

def build(q):
    case = FL.add_flux(G.load_case(q, 6, 10))
    p, err = perq_params(q)
    e = np.asarray(case["losses"]["e_oft"], float); t = np.asarray(case["t_bhpt"], float)
    alpha = p[0]*(1.0 + p[1]*e); beta = p[2]*(1.0 + p[3]*e)
    bc = G.creative.cumulative_trapezoid(beta, t)
    tau0 = bc - float(np.interp(G.T_ANCHOR, t, bc))
    r = minimize_scalar(lambda s: GG.err_at_t0(s, alpha, tau0, case["h_bhpt"],
                                               case["t_nr"], case["h_nr"]),
                        bounds=(-195., 45.), method="bounded",
                        options={"xatol": 1e-3, "maxiter": 80})
    return dict(tau=float(r.x)+tau0, alpha=alpha, beta=beta, err=float(r.fun),
                anchor=(q/(1+q))**1.2, params=p)

res = {q: build(q) for q in QS}
print(f"\n{'q':>6s} {'per-q E':>11s} {'a_PP':>9s} {'a_E':>10s} {'b_PP':>9s} {'b_E':>9s} "
      f"{'dbeta':>8s}")
for q in QS:
    p = res[q]["params"]; e = np.nan
    print(f"{q:6.2f} {res[q]['err']:11.4e} {p[0]:9.5f} {p[1]:+10.4f} {p[2]:9.5f} "
          f"{p[3]:+9.4f} {(res[q]['beta'][-1]/res[q]['beta'][0]-1)*100:+7.2f}%")

fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4))
for q in SIDE:
    r = res[q]; m = (r["tau"] >= -1000) & (r["tau"] <= 80)
    axa.plot(r["tau"][m], r["alpha"][m], color="0.65", lw=0.8, zorder=1, label=f"q={q:g}")
    axb.plot(r["tau"][m], r["beta"][m],  color="0.65", lw=0.8, zorder=1, label=f"q={q:g}")
for q, c in MAIN.items():
    r = res[q]; m = (r["tau"] >= -1000) & (r["tau"] <= 80); lw = 2.2 if q == 2 else 1.3
    axa.plot(r["tau"][m], r["alpha"][m], color=c, lw=lw, zorder=3,
             label=f"q={q:g}  E={r['err']:.2e}")
    axb.plot(r["tau"][m], r["beta"][m],  color=c, lw=lw, zorder=3, label=f"q={q:g}")
    axb.axhline(r["anchor"], color=c, lw=0.7, ls=(0, (4, 3)), zorder=2)
for ax, yl, tl in ((axa, "alpha(t)", "alpha(t) — PER-Q optimal"),
                   (axb, "beta(t)",  "beta(t) — PER-Q optimal  (dashed: X1^(6/5))")):
    ax.axvline(0, color="gray", ls=":", lw=0.8); ax.set_xlim(-1000, 80)
    ax.set_xlabel("t_NR / M"); ax.set_ylabel(yl); ax.set_title(tl)
    ax.grid(True); ax.legend(fontsize=8, ncol=2)
fig.suptitle("gwr-energy-anchored: PER-Q alpha/beta across q (no nu-regression)")
fig.tight_layout()
for ext in ("pdf", "png"):
    p = PLOT_DIR/f"gwr_anch_perq_alpha_beta_overlay.{ext}"; fig.savefig(p); print("wrote", p)
plt.close(fig)
