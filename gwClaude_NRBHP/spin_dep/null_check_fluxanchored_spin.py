"""chi1 = 0 NULL CHECK: the frozen nonspinning fluxanchored 7 evaluated on the SPIN cache.

Before any spin coefficient is fitted, establish the baseline.  The model is unchanged --
same 7 coefficients, same forms -- but the SOURCE changes from BHPTNRSur1dq1e4 to
BHPTNRSur2dq1e3 at zero spin, and the NR window is the spin pipeline's.  So this measures
what fluxanchored is worth on the grid the spin extension will actually be scored on.

Exact equality with the 1d number (median 4.6939e-04) is NOT expected and its absence is
not a bug: pn_anchored read 5.5425e-04 here against its own 1d value of 6.148e-04, i.e.
the 2d surrogate at chi1 = 0 gave a ~10% BETTER number for that model.  Whatever comes
out is the reference every later spin result must be compared against.

TWO DRIVE VARIANTS are reported, because the despike threshold is a live question:
`DESPIKE_T = 1e-2` of peak on |d2 Edot| was tuned on the dt = 0.2 nonspinning cache, and
|d2| scales as dt^2, so at this cache's dt = 0.1 it is effectively 4x tighter.  Measured:
it fires on 11-36 samples here, in a contiguous ~1 M run just before merger -- but the
phase-step ratio there is 1.005-1.062 (against 3.2-3.7 for the genuine 1d glitch) and the
dphase sequence is a smooth monotone chirp, so that is real merger steepening, not a
defect.  On this evidence the despike should be OFF for the 2d cache; both are run so the
cost of getting it wrong is on the record.

Note also that BHPTNRSur2dq1e3 was only ever verified clean AT t = -15 M, the 1d
surrogate's glitch location -- it has never been scanned for defects elsewhere.  The one
blemish found so far is q=5, chi1=+0.6, where |d2|h|| reaches 28000x the median with a
non-monotone dphase.  chi1 = 0 is unaffected.

Usage:  python null_check_fluxanchored_spin.py
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(PARENT))
warnings.filterwarnings("ignore")

import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_gwr_energy_fluxanchored as FA
import fit_scaling_gwr_fluxanchored_reg as REG
import fit_pn_anchored_spin as S          # the spin case builder (2d + NRHybSur)

COEFFS = PARENT / "gwr_fluxanchored_reg_results" / "coeffs_aF2.json"
REF_1D = {"median": 4.6939e-04, "max": 7.0480e-04}


def add_energy(case, q, despike):
    t = np.asarray(case["t_bhpt"], float)
    # spin models use E per TOTAL mass, spin-surrogate join window (2026-09-26)
    e = np.asarray(G.gwr_energy_coordinates(t, case["h_bhpt"], q, surrogate="2d",
                                            e_units="M")["e_oft"], float)
    ed = np.gradient(e, t)
    n_rep = 0
    if despike:
        ed, bad = REG.despike(ed, t, "2d")
        n_rep = int(bad.sum())
    f = ed / float(np.max(ed))
    case["losses"] = {"e_oft": e, "flux_hat": f}
    tf = case["fit_t_bhpt"]
    case["fit_losses"] = {"e_oft": np.interp(tf, t, e), "flux_hat": np.interp(tf, t, f)}
    return case, n_rep


def main():
    d = json.loads(COEFFS.read_text())
    th = np.concatenate([d["c"], d["A"], [d["b"]], d["P"]])
    print(f"frozen 7 (regularised, global): c={np.round(d['c'],5)} A={np.round(d['A'],5)} "
          f"b={d['b']:.5f} P={np.round(d['P'],4)}")
    print(f"1d reference: median {REF_1D['median']:.4e}  max {REF_1D['max']:.4e}\n")

    out = {}
    for despike in (False, True):
        tag = "despike ON " if despike else "despike OFF"
        errs = {}
        print(f"=== {tag} ===")
        print(f"{'q':>9} {'#repaired':>10} {'mismatch':>12}")
        for q in S.TRAIN_Q:
            case, n_rep = add_energy(S.build_case(q, 0.0), q, despike)
            p = FA.params_at(q, th, d["aF_degree"])
            # coarse t0 scan then refine -- the spin pipeline's NR window differs from
            # the nonspinning one, so the parent's t0 seed polynomial does not apply
            grid = np.linspace(-120.0, 40.0, 41)
            e0 = [FL.fast_mismatch(p, case, t0, "fit")[0] for t0 in grid]
            t0 = float(grid[int(np.argmin(e0))])
            _, t0 = FL.fast_mismatch(p, case, t0, "fit")
            errs[q] = FL.fast_mismatch(p, case, t0, "full")[0]
            print(f"{q:9.4f} {n_rep:10d} {errs[q]:12.4e}", flush=True)
        v = np.array(list(errs.values()))
        print(f"  median {np.median(v):.4e}   max {np.max(v):.4e}\n")
        out[tag.strip()] = {"per_q": errs, "median": float(np.median(v)),
                            "max": float(np.max(v))}

    print("SUMMARY")
    for k, r in out.items():
        print(f"  {k:12s} median {r['median']:.4e}  max {r['max']:.4e}  "
              f"vs 1d {REF_1D['median']:.4e}  ratio {r['median']/REF_1D['median']:.3f}")
    (HERE / "fluxanchored_spin_nullcheck.json").write_text(
        json.dumps({"reference_1d": REF_1D, "results": out}, indent=2))
    print(f"\nwrote {HERE/'fluxanchored_spin_nullcheck.json'}")


if __name__ == "__main__":
    main()
