"""What actually drives the q=2 master error (3.92e-3)? Override master param groups
to their per-q-truth values one at a time and measure mismatch. Isolates whether the
ringdown alpha (alpha_E,alpha_J) is the real lever, or beta / transition (p0,w) is.
"""
import sys, warnings
from pathlib import Path
import numpy as np
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTutils"))
import fit_scaling_wf_nu_hybrid_q_dep as wfhyb
import NRBHP_time_dep_wf_nu_hybrid_q_dep as P
N = wfhyb.PARAM_NAMES
idx = {n: i for i, n in enumerate(N)}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    row = wfhyb.optimize_case_worker((2.0, 3, 5, 4, 6000))
    truth = np.array(row["params"], float)
    coeffs = P.read_markdown_coefficients(P.MD_PATH)
    master0 = wfhyb.constrained_master_params(2.0, coeffs)
    case = wfhyb.load_case(2.0, source_stride=3, nr_stride=5)

def mism(params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, _ = wfhyb.polish_nuisance(params.copy(), case)   # re-polish t0,phi0
        ev = wfhyb.evaluate_model_hybrid(p, case["t_bhpt"], case["h_bhpt"],
                                         case["t_nr"], case["h_nr"], case["losses"])
    return ev.get("error", 50.0)

print(f"per-q truth mathcalE      = {mism(truth):.4e}")
print(f"master baseline mathcalE  = {mism(master0):.4e}\n")

groups = {
    "ringdown alpha (alpha_E,alpha_J)": ["alpha_E", "alpha_J"],
    "beta_r":                          ["beta_r"],
    "beta inspiral (beta_i,beta_L)":   ["beta_i", "beta_L"],
    "transition (p0,w)":               ["p0", "w"],
    "alpha_i":                         ["alpha_i"],
    "alpha_L":                         ["alpha_L"],
}
print("override master group -> truth:      mathcalE      (baseline 3.9e-3, truth 1.4e-3)")
for label, names in groups.items():
    p = master0.copy()
    for n in names: p[idx[n]] = truth[idx[n]]
    print(f"  {label:<34} {mism(p):.4e}")

# cumulative: everything remnant-related
p = master0.copy()
for n in ["alpha_E", "alpha_J", "beta_r"]: p[idx[n]] = truth[idx[n]]
print(f"\n  all remnant (alpha_E,alpha_J,beta_r)   {mism(p):.4e}")
p = master0.copy()
for n in ["p0","w","beta_i","beta_L"]: p[idx[n]] = truth[idx[n]]
print(f"  all inspiral/transition                {mism(p):.4e}")
