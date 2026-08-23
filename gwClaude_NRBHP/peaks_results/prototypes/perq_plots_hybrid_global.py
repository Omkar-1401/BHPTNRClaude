"""Per-q plot set for wf_nu_hybrid_global, matching the wf_nu_hybrid style:
  hybrid_global_q{N}[_extrap]_{waveform,zoomed,params,loss_coords}.pdf
into Agentic_plots/wf_nu_hybrid_global/.  q = 2, 3, 5, 8.
Uses the global coeffs; reuses P.build_model for the model evaluation.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTutils"))
import bhpt_utils
import fit_scaling_PN_opt_creative as creative
import fit_scaling_wf_nu_hybrid_q_dep as wfhyb
import NRBHP_time_dep_wf_nu_hybrid_q_dep as P

PLOT = ROOT/"Agentic_plots"/"wf_nu_hybrid_global"; PLOT.mkdir(parents=True, exist_ok=True)
Q_INPUTS = [8.0, 5.0, 3.0, 2.0]

def tag(q):
    s = f"q{int(q)}" if q == int(q) else f"q{q}"
    return s + ("_extrap" if (q < 3.0 or q > 8.0) else "")

def title(q, err=None):
    nu = wfhyb.get_nu(q)
    ex = " [extrapolation]" if (q < 3.0 or q > 8.0) else ""
    es = f";  mathcalE = {err:.3g}" if err is not None else ""
    return f"wf-nu-hybrid-global  q={q}{ex};  nu={nu:.4f}{es}"

def waveform_plot(q, t, h_nr, h_model, err):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios":[3,2]})
    for ax, xl in [(a1, (-1000, -200)), (a2, (-200, 100))]:
        ax.plot(t, np.real(h_nr), label="NR 22 mode")
        ax.plot(t, np.real(h_model), label="BHPT hybrid-global", alpha=0.8)
        ax.set_xlim(*xl); ax.grid(True); ax.set_xlabel("t/M")
    a1.legend(fontsize=8); fig.suptitle(title(q, err)); fig.tight_layout()
    fig.savefig(PLOT/f"hybrid_global_{tag(q)}_waveform.pdf"); plt.close(fig)

def zoomed_plot(q, t, h_nr, h_model):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, np.real(h_nr), label="NR 22 mode")
    ax.plot(t, np.real(h_model), label="BHPT hybrid-global", alpha=0.8)
    ax.set_xlim(-100, 100); ax.grid(True); ax.set_xlabel("t/M"); ax.legend(fontsize=8)
    ax.set_title(title(q)); fig.tight_layout()
    fig.savefig(PLOT/f"hybrid_global_{tag(q)}_zoomed.pdf"); plt.close(fig)

def params_plot(q, tau, alpha, beta, e_hat):
    ab = bhpt_utils.all_alpha_beta_1D_scaling_params(q, [creative.MODE])
    lbl = f"wf-nu-hybrid-global  nu={wfhyb.get_nu(q):.4f}"
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(tau, alpha, label=lbl); ax[0].axhline(ab["alpha_l2m2"], color="k", ls="--", label="BHPTNRSurrogate alpha")
    ax[0].set_xlim(-1000, 100); ax[0].grid(True); ax[0].set_xlabel("t_NR / M"); ax[0].set_ylabel("alpha(t)"); ax[0].legend(fontsize=7)
    ax[1].plot(tau, beta, label=lbl); ax[1].axhline(ab["beta"], color="k", ls="--", label="BHPTNRSurrogate beta")
    ax[1].set_xlim(-1000, 100); ax[1].grid(True); ax[1].set_xlabel("t_NR / M"); ax[1].set_ylabel("beta(t)"); ax[1].legend(fontsize=7)
    o = np.argsort(e_hat)
    ax[2].plot(e_hat[o], alpha[o], lw=1, label=lbl); ax[2].set_xlabel("Ehat (wf-flux)"); ax[2].set_ylabel("alpha")
    ax[2].set_title("alpha vs Ehat"); ax[2].grid(True)
    fig.suptitle(title(q)); fig.tight_layout()
    fig.savefig(PLOT/f"hybrid_global_{tag(q)}_params.pdf"); plt.close(fig)

def loss_plot(q, t_bhpt, losses):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    m = (t_bhpt >= -1500) & (t_bhpt <= 150)
    for key, a, lb in [("e_hat", ax[0], "Ehat"), ("j_hat", ax[1], "Jhat"), ("p_loss", ax[2], "p_loss")]:
        a.plot(t_bhpt[m], losses[key][m], lw=1); a.axvline(0, color="gray", ls="--", lw=0.8)
        a.axhline(0, color="k", ls=":", lw=0.6, alpha=0.5); a.set_xlabel("t / M"); a.set_ylabel(lb)
        a.set_title(f"{lb} (wf flux)"); a.grid(True)
    fig.suptitle(f"wf-nu-hybrid-global loss coordinates  q={q}  nu={wfhyb.get_nu(q):.4f}")
    fig.tight_layout(); fig.savefig(PLOT/f"hybrid_global_{tag(q)}_loss_coords.pdf"); plt.close(fig)

def main():
    d = json.loads((ROOT/"wf_nu_hybrid_global_results"/"coeffs.json").read_text())
    coeffs = {n: np.array(d[n]) for n in wfhyb.PARAM_NAMES}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for q in Q_INPUTS:
            res = P.build_model(q, coeffs)
            ev, case = res["eval"], res["case"]
            t = case["t_nr"][ev["common"]]
            waveform_plot(q, t, ev["h_ref"], ev["h_model"], ev["error"])
            zoomed_plot(q, t, ev["h_ref"], ev["h_model"])
            params_plot(q, ev["tau"], ev["alpha"], ev["beta"], ev["e_hat"])
            loss_plot(q, case["t_bhpt"], case["losses"])
            print(f"q={q}: mathcalE={ev['error']:.3e}  -> hybrid_global_{tag(q)}_*.pdf", flush=True)
    print(f"saved per-q plots to {PLOT}")

if __name__ == "__main__":
    main()
