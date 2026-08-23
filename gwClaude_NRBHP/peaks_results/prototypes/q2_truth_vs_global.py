"""q=2: per-q TRUTH vs even-nu GLOBAL extrapolation, alpha(t) and beta(t)."""
import sys, warnings, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTutils"))
import fit_scaling_wf_nu_hybrid_q_dep as H
Q = 2.0
OUT = ROOT/"Agentic_plots"/"wf_nu_hybrid_global"

def ab_on_nr(params, case):
    ev = H.evaluate_model_hybrid(params, case["t_bhpt"], case["h_bhpt"],
                                 case["t_nr"], case["h_nr"], case["losses"])
    p0, w = params[0], params[1]
    S = H.creative.sigmoid((case["losses"]["p_loss"] - p0)/w)
    tau = ev["tau"]; common = ev["common"]; tc = case["t_nr"][common]
    a = np.interp(tc, tau, ev["alpha"]); b = np.interp(tc, tau, ev["beta"])
    Sc = np.interp(tc, tau, S)
    return tc, a, b, Sc, ev["error"]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    row = H.optimize_case_worker((Q, 3, 5, 4, 6000))       # per-q truth
    truth = np.array(row["params"], dtype=float)
    d = json.loads((ROOT/"wf_nu_hybrid_global_results"/"coeffs.json").read_text())
    coeffs = {n: np.array(d[n]) for n in H.PARAM_NAMES}     # even-nu global
    glob = H.constrained_master_params(Q, coeffs)
    case = H.load_case(Q, 3, 5)
    glob, _ = H.polish_nuisance(glob, case)
    tct, at, bt, St, et = ab_on_nr(truth, case)
    tcg, ag, bg, Sg, eg = ab_on_nr(glob, case)

print(f"q={Q}: per-q truth mathcalE={et:.4e}   even-nu global mathcalE={eg:.4e}")
print("\nparam        truth        global(even-nu)")
for i, n in enumerate(H.PARAM_NAMES):
    print(f"  {n:<9} {truth[i]:11.5g}  {glob[i]:11.5g}")
for lbl, a, b, S in [("truth", at, bt, St), ("global", ag, bg, Sg)]:
    rd = S > 0.98
    if rd.any(): print(f"{lbl} ringdown(S>0.98): alpha={a[rd].mean():.4f} beta={b[rd].mean():.4f}")

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
for col, (lo, hi) in enumerate([(-1500, 80), (-200, 80)]):
    ax[0, col].plot(tct, at, lw=1.9, label=f"per-q truth (E={et:.2e})")
    ax[0, col].plot(tcg, ag, "--", lw=1.5, label=f"even-nu global (E={eg:.2e})")
    ax[1, col].plot(tct, bt, lw=1.9, label="per-q truth")
    ax[1, col].plot(tcg, bg, "--", lw=1.5, label="even-nu global")
    for r in (0, 1):
        ax[r, col].set_xlim(lo, hi); ax[r, col].grid(alpha=0.3); ax[r, col].axvline(0, ls=":", color="gray", lw=1)
ax[0,0].set_ylabel("alpha(t)"); ax[1,0].set_ylabel("beta(t)")
ax[0,0].set_title("alpha  (full)"); ax[0,1].set_title("alpha  (merger/ringdown zoom)")
ax[1,0].set_title("beta  (full)"); ax[1,1].set_title("beta  (merger/ringdown zoom)")
for a in ax[1]: a.set_xlabel("t_NR / M")
ax[0,0].legend(fontsize=8); ax[1,0].legend(fontsize=8)
fig.suptitle("q=2 [EXTRAP]: even-nu global vs per-q truth  —  alpha & beta")
fig.tight_layout()
p = OUT/"hybrid_global_q2_vs_perq_truth.png"; fig.savefig(p, dpi=120)
print(f"\nsaved {p}")
