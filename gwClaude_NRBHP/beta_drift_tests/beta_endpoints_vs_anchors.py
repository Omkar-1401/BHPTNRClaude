"""Does the fitted beta land on the two DERIVED anchors, and where is the miss?

The `mult`/`anchored`/`fluxanchored` models never see B1 -- only `betaqnm` imposes it.  So
the question is purely empirical: given a per-q fit that reaches 2.5e-4, what are
beta(window start) and beta(window end), and how far are they from X1^(6/5) and the QNM
ratio?  Also: what fraction of the mismatch numerator does the ringdown actually carry, i.e.
how much does a terminal-beta error cost?
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL
import gwModels
import transition_q_decompose as TD

W = 0.3737
QNM = json.loads((HERE.parent / "pn_anchored_results" / "config.json").read_text())["qnm"] \
    if (HERE.parent / "pn_anchored_results" / "config.json").exists() else None
if QNM is None:
    import BHPTNRPNAnchored as PNA; QNM = PNA._cfg()["qnm"]

mult = sorted(json.load(open(str(HERE.parent /
    "gw_remnant_energy_results/per_q_cache_mult.json"))).values(), key=lambda v: v["q"])
extra = json.loads((HERE.parent / "gwr_energy_anchored_results" /
                    "per_q_lowq_mult.json").read_text()) if (
    HERE.parent / "gwr_energy_anchored_results" / "per_q_lowq_mult.json").exists() else {}

def anchors(q):
    r = gwModels.remnants.gwModelRemS(q, 0.0, 0.0)
    Mf, chif = float(r[0]), float(r[1])
    B1 = W * Mf / (QNM["F1"] + QNM["F2"] * (1.0 - chif) ** QNM["F3"])
    return (q / (1 + q)) ** 1.2, B1

print("per-q `mult` fits (the gwr_energy_anchored per-q model), (2,2)\n")
print(f"{'q':>6s} {'per-q E':>10s} | {'b(start)':>9s} {'B0':>8s} {'miss %':>7s} | "
      f"{'b(merg)':>8s} {'b(end)':>8s} {'B1':>8s} {'miss %':>7s} | {'dbeta %':>8s} "
      f"{'g_b %':>7s} | {'ringdown':>9s}")
for qt in (3.0, 4.0, 5.0, 6.0, 8.0, 2.5, 2.0):
    rec = min(mult, key=lambda v: abs(v["q"] - qt))
    if abs(rec["q"] - qt) > 1e-6:
        k = f"{qt:.4f}"
        if k not in extra: continue
        rec = {"q": qt, "params": extra[k]["params"],
               "err": extra[k]["error"], "t0": None}
    q = float(rec["q"]); p = np.asarray(rec["params"], float)
    case = FL.add_flux(G.load_case(q, 6, 10))
    e = np.asarray(case["losses"]["e_oft"], float)
    t = np.asarray(case["t_bhpt"], float)
    b_pp, b_e = p[2], p[3]
    beta = b_pp * (1.0 + b_e * e)
    im = int(np.argmax(np.abs(case["h_bhpt"])))
    B0, B1 = anchors(q)
    ferr = float(rec["err"]) if "err" in rec else float(rec["error"])
    # how much of the mismatch numerator lives after the NR merger?
    def shape_mult(pp, tb, los):
        al, be = G.alpha_beta(pp, los, "mult")
        if np.any(be <= 0):
            return al, None, -1.0
        bc = G.creative.cumulative_trapezoid(be, tb)
        ts = bc - float(np.interp(G.T_ANCHOR, tb, bc))
        return al, ts, float(np.min(np.diff(ts)))
    t0 = float(p[4]) if len(p) > 4 else 0.0
    tot, pre, post, _ = TD.split_err(p[:4], case, t0, 0.0, shape_mult)
    print(f"{q:6.2f} {ferr:10.4e} | {beta[0]:9.5f} {B0:8.5f} "
          f"{(beta[0]/B0-1)*100:+7.3f} | {beta[im]:8.5f} {beta[-1]:8.5f} {B1:8.5f} "
          f"{(beta[-1]/B1-1)*100:+7.3f} | {(beta[-1]/beta[0]-1)*100:+8.3f} "
          f"{(B1/B0-1)*100:+7.3f} | {post/tot*100:8.2f}%", flush=True)
print("\ne_oft[0] (should be 0 -> beta(start) = beta_PP exactly):",
      float(e[0]))
print("last column = share of the full-window mismatch numerator from t_NR > 0")
