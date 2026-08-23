"""Test the degeneracy-breaking fix for wf_nu_hybrid q=2 (PURE [3,8] training).

Hypothesis (confirmed at single q): beta_r is degenerate/compensable, so the free
per-q fits pick inconsistent basins -> noisy param trends -> bad extrapolation.
Fix: PIN beta_r to a smooth curve (regressed from the free fits), then RE-FIT the
other 10 params per-q so they vary consistently, then regress THOSE in nu and
extrapolate to q=2.  No data below q=3 is used.

Compares q=2 mismatch: current master (free-fit regression) vs pinned-betar regression.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTutils"))
import fit_scaling_wf_nu_hybrid_q_dep as wfhyb
import NRBHP_time_dep_wf_nu_hybrid_q_dep as P
N = wfhyb.PARAM_NAMES; IB = N.index("beta_r")
FREE = [i for i in range(11) if i != IB]

cache = json.loads(Path("wf_nu_hybrid_q_dep_results/per_q_cache.json").read_text())
rows = sorted((float(k), np.array(v["params"], float)) for k, v in cache.items())
qs_all = np.array([r[0] for r in rows])
br_all = np.array([r[1][IB] for r in rows])
nu_all = qs_all/(1+qs_all)**2

# smooth beta_r curve from the FREE fits (nu deg-3); this is what we PIN to.
br_poly = np.polyfit(nu_all, br_all, 3)
def br_of(q): return float(np.polyval(br_poly, q/(1+q)**2))

TRAIN = [3.0, 3.4, 3.8, 4.2, 4.6, 5.0, 5.6, 6.2, 7.0, 8.0]   # subset of [3,8]
cases = {}
def get_case(q):
    if q not in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); cases[q] = wfhyb.load_case(q, 3, 5)
    return cases[q]

def mism(q, p):
    c = get_case(q)
    ev = wfhyb.evaluate_model_hybrid(p, c["t_bhpt"], c["h_bhpt"], c["t_nr"], c["h_nr"], c["losses"])
    return ev.get("error", 50.0)

def refit_pinned(q, seed):
    """re-fit the 10 free params with beta_r pinned to br_of(q)."""
    br = br_of(q)
    def obj(x):
        p = seed.copy()
        for j, i in enumerate(FREE): p[i] = x[j]
        p[IB] = br
        with warnings.catch_warnings(): warnings.simplefilter("ignore"); return mism(q, p)
    r = minimize(obj, seed[FREE], method="Nelder-Mead",
                 options={"maxiter": 6000, "fatol": 1e-9, "xatol": 1e-7})
    p = seed.copy()
    for j, i in enumerate(FREE): p[i] = r.x[j]
    p[IB] = br
    return p, r.fun

# 1) re-fit each training q with beta_r pinned, seeded from its free fit
seed_of = {q: rows[int(np.argmin(np.abs(qs_all-q)))][1] for q in TRAIN}
print("re-fitting [3,8] with beta_r pinned to smooth curve ...", flush=True)
pinned = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for q in TRAIN:
        p, e = refit_pinned(q, seed_of[q])
        pinned[q] = p
        print(f"  q={q:.2f}  beta_r_pin={br_of(q):.4f}  mismatch={e:.3e}", flush=True)

# 2) regress the 10 free params in nu, PP-anchored (reuse fit_poly_anchored)
nus = np.array([q/(1+q)**2 for q in TRAIN])
coeffs = {}
for i in FREE:
    name = N[i]
    vals = np.array([pinned[q][i] for q in TRAIN])
    coeffs[name] = wfhyb.fit_poly_anchored(nus, vals, 3, wfhyb.PP_ANCHORS[name])
coeffs["beta_r"] = br_poly[::-1]   # store beta_r nu-poly (low->high) for eval convention

def master_params_pinned(q):
    p = np.array([wfhyb.eval_poly(q, coeffs[n]) if n != "beta_r" else br_of(q) for n in N])
    # apply same clips as constrained_master_params
    p[2] = np.clip(p[2], 0.05, 2.5); p[IB] = np.clip(p[IB], 0.2, 1.6)
    return p

# 3) evaluate q=2 (and a couple in-range) for the pinned-regression master
print("\n=== q=2 comparison (PURE [3,8]) ===")
mc = P.read_markdown_coefficients(P.MD_PATH)
for q in [5.0, 3.0, 2.0]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = wfhyb.constrained_master_params(q, mc); pm, _ = wfhyb.polish_nuisance(pm, get_case(q))
        pp = master_params_pinned(q);              pp, _ = wfhyb.polish_nuisance(pp, get_case(q))
        e_master = mism(q, pm); e_pinned = mism(q, pp)
    tag = " <-- q=2" if q == 2.0 else ""
    print(f"  q={q}: current master={e_master:.3e}   pinned-betar={e_pinned:.3e}{tag}")
