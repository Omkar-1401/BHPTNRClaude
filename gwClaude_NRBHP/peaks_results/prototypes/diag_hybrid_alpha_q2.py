"""Diagnose the alpha SHAPE error at q=2 for wf_nu_hybrid.
Compare the master-extrapolated alpha(t) against the PER-Q OPTIMAL alpha (truth),
and split by the sigmoid S so we see whether the error is inspiral or remnant segment.
Also report the alpha/beta values the two models put in the ringdown (S->1).
"""
import sys, warnings
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTutils"))
import fit_scaling_wf_nu_hybrid_q_dep as wfhyb
import NRBHP_time_dep_wf_nu_hybrid_q_dep as P
SP = "/tmp/claude-1000/-home-omkarnm1401-UT-Austin/f16ec578-7d70-4a62-81cc-3a1c1e5c39a6/scratchpad"
Q = 2.0

def alpha_beta_on_nr(params, case):
    ev = wfhyb.evaluate_model_hybrid(params, case["t_bhpt"], case["h_bhpt"],
                                     case["t_nr"], case["h_nr"], case["losses"])
    # map alpha,beta,S onto NR common time
    p0, w = params[0], params[1]
    S = wfhyb.creative.sigmoid((case["losses"]["p_loss"] - p0)/w)
    tau = ev["tau"]; common = ev["common"]; tc = case["t_nr"][common]
    a = np.interp(tc, tau, ev["alpha"]); b = np.interp(tc, tau, ev["beta"])
    Sc = np.interp(tc, tau, S)
    return tc, a, b, Sc, ev["error"]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # per-q OPTIMAL (truth) at q=2
    row = wfhyb.optimize_case_worker((Q, 3, 5, 4, 6000))
    truth = np.array(row["params"], dtype=float)
    # master extrapolation to q=2
    coeffs = P.read_markdown_coefficients(P.MD_PATH)
    master = wfhyb.constrained_master_params(Q, coeffs)
    case = wfhyb.load_case(Q, source_stride=3, nr_stride=5)
    # polish nuisance (t0,phi0) for master like the plotting path does
    master, _ = wfhyb.polish_nuisance(master, case)
    tc_t, a_t, b_t, S_t, e_t = alpha_beta_on_nr(truth, case)
    tc_m, a_m, b_m, S_m, e_m = alpha_beta_on_nr(master, case)

print(f"q={Q}  per-q optimal mathcalE={e_t:.4e}   master mathcalE={e_m:.4e}")
print("\nparam        truth        master")
for i, n in enumerate(wfhyb.PARAM_NAMES):
    print(f"  {n:<9} {truth[i]:11.5g}  {master[i]:11.5g}")

# ringdown-plateau values (S>0.98)
for lbl, tc, a, b, S in [("truth", tc_t, a_t, b_t, S_t), ("master", tc_m, a_m, b_m, S_m)]:
    rd = S > 0.98
    if rd.any():
        print(f"{lbl} ringdown (S>0.98): alpha={a[rd].mean():.4f}  beta={b[rd].mean():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(tc_t, a_t, label=f"per-q optimal (mathcalE={e_t:.2e})", lw=1.8)
axes[0].plot(tc_m, a_m, label=f"master extrap (mathcalE={e_m:.2e})", lw=1.4, ls="--")
ax2 = axes[0].twinx(); ax2.plot(tc_t, S_t, color="gray", lw=0.8, alpha=0.5)
ax2.set_ylabel("S (sigmoid switch)", color="gray"); ax2.set_ylim(-0.05, 1.05)
axes[0].set_xlim(-1500, 100); axes[0].set_xlabel("t_NR / M"); axes[0].set_ylabel("alpha")
axes[0].set_title(f"q={Q} alpha(t): master vs per-q truth"); axes[0].legend(fontsize=8); axes[0].grid(True)
axes[1].plot(tc_t, a_t, lw=1.8, label="truth"); axes[1].plot(tc_m, a_m, lw=1.4, ls="--", label="master")
axes[1].set_xlim(-200, 80); axes[1].set_xlabel("t_NR / M"); axes[1].set_ylabel("alpha")
axes[1].set_title("zoom near merger/ringdown"); axes[1].legend(fontsize=8); axes[1].grid(True)
plt.tight_layout(); plt.savefig(f"{SP}/hybrid_alpha_q2_diag.png", dpi=100)
print(f"\nsaved {SP}/hybrid_alpha_q2_diag.png")
