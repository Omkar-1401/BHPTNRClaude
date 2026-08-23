"""Validate a global-joint-fit coeff set: mismatch table (in-range + q=2 extrap) and
alpha(t)/beta(t) shape-continuity plot across q=2,3,5,8 vs the current master.
Usage: validate_global.py <global_coeffs.json>
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTutils"))
sys.path.insert(0, str(ROOT/"peaks_results"/"prototypes"))
import fit_scaling_wf_nu_hybrid_q_dep as H
import NRBHP_time_dep_wf_nu_hybrid_q_dep as P
import global_joint_fit as G
SP = "/tmp/claude-1000/-home-omkarnm1401-UT-Austin/f16ec578-7d70-4a62-81cc-3a1c1e5c39a6/scratchpad"
N = H.PARAM_NAMES

def load_global(fn):
    d = json.loads(Path(fn).read_text())
    return {n: np.array(d[n]) for n in N}

def alpha_beta_tau(q, coeffs, case):
    p = G.params_at(q, coeffs)
    p, _ = H.polish_nuisance(p, case)
    ev = H.evaluate_model_hybrid(p, case["t_bhpt"], case["h_bhpt"],
                                 case["t_nr"], case["h_nr"], case["losses"])
    return ev["tau"], ev["alpha"], ev["beta"], ev.get("error", 50.0)

def main():
    fn = sys.argv[1]
    tag = Path(fn).stem
    gco = load_global(fn)
    cache = json.loads((ROOT/"wf_nu_hybrid_q_dep_results"/"per_q_cache.json").read_text())
    rows = [{"q": float(k), "params": v["params"]} for k, v in cache.items()]
    master = H.fit_master(rows, 3)

    qs_plot = [8.0, 5.0, 3.0, 2.0]
    colors = {8.0: "#1f77b4", 5.0: "#2ca02c", 3.0: "#ff7f0e", 2.0: "#d62728"}
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    print(f"{'q':>5} {'global':>11} {'master':>11}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for q in [8.0, 5.0, 3.0, 2.0]:
            case = H.load_case(q, 3, 5)
            tg, ag, bg, eg = alpha_beta_tau(q, gco, case)
            tm, am, bm, em = alpha_beta_tau(q, master, case)
            ls = "--" if q == 2.0 else "-"
            lw = 2.2 if q == 2.0 else 1.3
            ax[0,0].plot(tg, ag, ls, color=colors[q], lw=lw, label=f"q={q}")
            ax[0,1].plot(tg, bg, ls, color=colors[q], lw=lw, label=f"q={q}")
            ax[1,0].plot(tm, am, ls, color=colors[q], lw=lw, label=f"q={q}")
            ax[1,1].plot(tm, bm, ls, color=colors[q], lw=lw, label=f"q={q}")
            print(f"{q:5.1f} {eg:11.3e} {em:11.3e}")
    for a in ax.ravel(): a.set_xlim(-800, 80); a.grid(True); a.set_xlabel("t_NR / M")
    ax[0,0].set_title(f"GLOBAL ({tag}): alpha(t)"); ax[0,0].set_ylabel("alpha"); ax[0,0].legend(fontsize=8)
    ax[0,1].set_title("GLOBAL: beta(t)"); ax[0,1].set_ylabel("beta")
    ax[1,0].set_title("MASTER: alpha(t)"); ax[1,0].set_ylabel("alpha")
    ax[1,1].set_title("MASTER: beta(t)"); ax[1,1].set_ylabel("beta")
    fig.suptitle(f"Shape continuity q=2(dashed) vs 3,5,8 — global vs master")
    fig.tight_layout(); out = f"{SP}/shape_{tag}.png"; fig.savefig(out, dpi=95)
    print(f"saved {out}")

if __name__ == "__main__":
    main()
