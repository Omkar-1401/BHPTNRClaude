"""Step 1: does the QNM-anchor beta (rising) fix the HIGHER MODES?

Higher modes carry more merger-ringdown weight, so if the shipped base is ~5x wrong in the
ringdown (ringdown_only_mismatch.py), (4,4) should improve when beta is replaced by the
parameter-free QNM-anchor value.  Nothing is fitted: the backbone transfer is rho=1 and
only the transferred quadrupole's b_E changes.
"""
import sys, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fit_scaling_hm_backbone as hm
import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_gwr_anchored2 as A2

base = hm.read_stiff_coeffs("flux_anchored")          # the shipped best base
_cases = {}
def anchor_be(q):
    if q not in _cases:
        _cases[q] = FL.add_flux(G.load_case(q, 6, 10))
    A0, ga, gb, Etot, A1, B1 = A2.anchors(q, _cases[q])
    p = base["params_at"](q)
    return gb * A0 / p[2] / Etot                       # positive -> beta rises

variants = {
    "shipped  (beta falls)": base,
    "anchor-b (beta rises)": {"name": "anchor_beta", "coord": base["coord"],
                              "params_at": lambda q: np.r_[base["params_at"](q)[:3],
                                                           anchor_be(q)]},
}
show = [(2, 2), (3, 3), (4, 4), (5, 5), (2, 1)]
for q in (4.0, 2.0):
    anchor_be(q)          # populate the case cache before either variant runs
    print(f"\n===== q = {q:g} =====")
    res = {}
    for label, cf in variants.items():
        r = hm.run_q(q, cf, verbose=False)
        res[label] = r
        p = cf["params_at"](q)
        Et = A2.anchors(q, _cases[q])[3]
        print(f"  {label}   b_E={p[3]:+8.4f}  dbeta={p[3]*Et*100:+6.2f}%")
    print(f"  (2,2) reference: shipped {res['shipped  (beta falls)']['error22']:.4e}  "
          f"anchor {res['anchor-b (beta rises)']['error22']:.4e}")
    print(f"  {'mode':>7s} {'power':>9s} {'shipped':>12s} {'anchor-beta':>12s} "
          f"{'ratio':>7s}  winner")
    for m in show:
        a = res["shipped  (beta falls)"]["rows"][m]["err_rho1"]
        b = res["anchor-b (beta rises)"]["rows"][m]["err_rho1"]
        pw = res["shipped  (beta falls)"]["rows"][m]["power_frac"]
        print(f"  {str(m):>7s} {pw:9.2e} {a:12.4e} {b:12.4e} {b/a:7.2f}  "
              f"{'ANCHOR' if b < a else 'shipped'}")
    # power-weighted mode sum over all 9 modes
    for label in res:
        rr = res[label]["rows"]
        tot = sum(rr[mo]["err_rho1"] * rr[mo]["power_frac"] for mo in rr)
        print(f"  power-weighted mode sum, {label}: {tot:.4e}")
