"""Decisive test for the user's idea: is the ringdown alpha plateau a smooth,
extrapolable function of chi_f (remnant coord) that hits the q=2 truth (~0.34)?
Compare chi_f vs nu as the regression coordinate for ringdown alpha.

For each cached per-q [3,8] fit: build model, measure ringdown alpha plateau
(mean alpha where S>0.98). Get (M_f, chi_f) from surfinBH. Overlay q=2 truth.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTutils"))
import fit_scaling_wf_nu_hybrid_q_dep as wfhyb
import fit_scaling_PN_opt_remnant_partial as remnant   # get_remnant(q)->(M_f,chi_f)
SP = "/tmp/claude-1000/-home-omkarnm1401-UT-Austin/f16ec578-7d70-4a62-81cc-3a1c1e5c39a6/scratchpad"

cache = json.loads(Path("wf_nu_hybrid_q_dep_results/per_q_cache.json").read_text())
key_of = {float(k): k for k in cache.keys()}
qs_all = sorted(key_of)
qs = qs_all[::5]            # ~13 points across [3,8]

def ringdown_alpha(q, params):
    case = wfhyb.load_case(q, source_stride=3, nr_stride=5)
    ev = wfhyb.evaluate_model_hybrid(params, case["t_bhpt"], case["h_bhpt"],
                                     case["t_nr"], case["h_nr"], case["losses"])
    p0, w = params[0], params[1]
    S = wfhyb.creative.sigmoid((case["losses"]["p_loss"] - p0)/w)
    tc = case["t_nr"][ev["common"]]
    a = np.interp(tc, ev["tau"], ev["alpha"]); Sc = np.interp(tc, ev["tau"], S)
    rd = Sc > 0.98
    return float(a[rd].mean()) if rd.any() else np.nan

rows = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for q in qs:
        p = np.array(cache[key_of[q]]["params"])
        ar = ringdown_alpha(q, p)
        mf, chif = remnant.get_remnant(float(q))
        rows.append((q, q/(1+q)**2, mf, chif, ar))
    # q=2 truth
    mf2, chif2 = remnant.get_remnant(2.0)
    truth_ar = 0.341   # measured in diag_hybrid_alpha_q2.py

qa = np.array(rows)
qv, nuv, mfv, chifv, arv = qa[:,0], qa[:,1], qa[:,2], qa[:,3], qa[:,4]
print(" q     nu     M_f    chi_f   ringdown_alpha")
for r in rows: print(f"{r[0]:5.2f} {r[1]:.4f} {r[2]:.4f} {r[3]:.4f}   {r[4]:.4f}")
print(f"\nq=2 truth: nu={2/9:.4f} M_f={mf2:.4f} chi_f={chif2:.4f} ringdown_alpha={truth_ar:.4f}")

# fit low-order polys in each coord over [3,8] and extrapolate to q=2
def fit_extrap(x, y, x2, deg=2):
    c = np.polyfit(x, y, deg); return np.polyval(c, x2), c

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, xv, x2, name in [(axes[0], chifv, chif2, "chi_f"),
                         (axes[1], nuv, 2/9, "nu"),
                         (axes[2], mfv, mf2, "M_f")]:
    order = np.argsort(xv)
    ax.plot(xv[order], arv[order], "o-", ms=4, label="per-q [3,8]")
    for deg in (2, 3):
        pred, _ = fit_extrap(xv, arv, x2, deg)
        xline = np.linspace(min(xv.min(), x2), max(xv.max(), x2), 100)
        ax.plot(xline, np.polyval(np.polyfit(xv, arv, deg), xline), "--", lw=1,
                label=f"deg{deg} extrap q2={pred:.3f}")
    ax.plot([x2], [truth_ar], "r*", ms=16, label=f"q=2 TRUTH={truth_ar:.3f}")
    ax.set_xlabel(name); ax.set_ylabel("ringdown alpha plateau")
    ax.set_title(f"ringdown alpha vs {name}"); ax.grid(True); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{SP}/alpha_r_coord.png", dpi=100)
print(f"\nsaved {SP}/alpha_r_coord.png")
