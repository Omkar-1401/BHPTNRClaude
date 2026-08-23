"""One-off driver: generate wf_nu_hybrid plots at q=1.5 (deep extrapolation).

Reuses the exact plot functions in NRBHP_time_dep_wf_nu_hybrid_q_dep.py but bypasses
that script's q=2<1e-2 gate (we WANT to see how bad q=1.5 looks). Writes the standard
hybrid_q1.5_extrap_{waveform,zoomed,params,loss_coords}.pdf into Agentic_plots/wf_nu_hybrid/.
"""
import sys, warnings
from pathlib import Path
import numpy as np

ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "BHPTutils"))

import NRBHP_time_dep_wf_nu_hybrid_q_dep as P
import fit_scaling_wf_nu_hybrid_q_dep as wfhyb

Q = 1.5

def run():
    coeffs = P.read_markdown_coefficients(P.MD_PATH)
    print(f"building wf_nu_hybrid at q={Q} (nu={wfhyb.get_nu(Q):.4f}) ...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = P.build_model(Q, coeffs)
    ev, case = result["eval"], result["case"]
    err = ev["error"]
    cov = np.mean(ev["common"]) if "common" in ev else float("nan")
    print(f"  mathcalE = {err:.6g}   coverage = {cov:.3f}")

    P.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    common_t = case["t_nr"][ev["common"]]
    paths = [
        P.save_waveform_plot(Q, common_t, ev["h_ref"], ev["h_model"], err),
        P.save_zoomed_plot(Q, common_t, ev["h_ref"], ev["h_model"]),
        P.save_parameter_plot(Q, ev["tau"], ev["alpha"], ev["beta"],
                              ev["e_hat"], ev["j_hat"]),
        P.save_loss_coord_plot(Q, case["t_bhpt"], case["losses"]),
    ]
    print("wrote:")
    for p in paths:
        print(f"  {p}")

if __name__ == "__main__":
    run()
