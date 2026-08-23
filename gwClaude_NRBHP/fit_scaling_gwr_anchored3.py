"""gwr_anchored3 -- anchored2 with alpha's anchor placed correctly.

Fixes the two bookkeeping errors in anchored2:
  1. A1 (the peak amplitude ratio) is a value AT MERGER, so alpha's coupling is
     normalised by Ehat_merger, not by E_tot.  alpha then hits A1 at merger.
  2. alpha keeps falling after merger, so it gets a separate post-merger slope
     instead of being held constant.

    Ehat  = Eoft/E_tot                    in [0,1] over the window
    Ehm   = Ehat at the |h| peak          measured, ~0.60 (q=2) .. 0.75 (q=8)
    pre   = clip(Ehat/Ehm, 0, 1)          0 -> 1 by merger
    post  = clip((Ehat-Ehm)/(1-Ehm), 0)   0 -> 1 over the ringdown

    alpha = a_pp * [ 1 + w_a*ga*(A0/a_pp)*pre + w_p*ga*(A0/a_pp)*post ]
    beta  = b_pp * [ 1 + w_b*gb*(A0/b_pp)*Ehat ]

DERIVED anchors: A0 = X1^(6/5); A1 = peak amplitude ratio (-> ga);
                 B1 = omega_Schw/omega_220(Mf,chif) (-> gb).
FITTED: c in w_a = 1 + c*nu^2, and w_p.  w_b scanned / held.

Usage:  python fit_scaling_gwr_anchored3.py [--wb 1.0] [--scan]
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); warnings.filterwarnings("ignore")
import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_global as GG
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_gwr_anchored2 as A2

RESULTS = ROOT / "gwr_anchored3_results"; RESULTS.mkdir(exist_ok=True)
IN_Q = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
LOW_Q = (2.75, 2.5, 2.25, 2.0)
REF = {"fluxanchored (7c)": {"median": 4.7516e-04, "max": 7.1503e-04, 2.75: 6.3465e-04,
                             2.5: 5.9369e-04, 2.25: 6.8673e-04, 2.0: 1.3056e-03},
       "anchored (6c)":     {"median": 6.2551e-04, "max": 9.7884e-04, 2.75: 9.4774e-04,
                             2.5: 9.2793e-04, 2.25: 1.0163e-03, 2.0: 1.5293e-03}}


def prep(q, case):
    """Everything q-dependent and un-fitted: anchors, Ehat, the pre/post ramps."""
    A0, ga, gb, E_tot, A1, B1 = A2.anchors(q, case)
    e = np.asarray(case["losses"]["e_oft"], float)
    Ehat = (e - e[0]) / E_tot
    im = int(np.argmax(np.abs(case["h_bhpt"])))
    Ehm = float(Ehat[im])
    pre = np.clip(Ehat / Ehm, 0.0, 1.0)
    post = np.clip((Ehat - Ehm) / max(1e-6, 1.0 - Ehm), 0.0, None)
    a_pp, b_pp = A2._prefac(q, "anchored")
    return dict(A0=A0, ga=ga, gb=gb, Ehat=Ehat, Ehm=Ehm, pre=pre, post=post,
                a_pp=a_pp, b_pp=b_pp, t=np.asarray(case["t_bhpt"], float))


def shape(P, c, w_p, w_b):
    nu_c = 1.0 + c  # placeholder to keep signature simple; c passed pre-multiplied
    a = P["a_pp"] * (1.0 + nu_c * P["ga"] * P["A0"] / P["a_pp"] * P["pre"]
                     + w_p * P["ga"] * P["A0"] / P["a_pp"] * P["post"])
    b = P["b_pp"] * (1.0 + w_b * P["gb"] * P["A0"] / P["b_pp"] * P["Ehat"])
    if np.any(b <= 0) or not np.all(np.isfinite(np.r_[a, b])):
        return a, None, -1.0
    bc = G.creative.cumulative_trapezoid(b, P["t"])
    tau = bc - float(np.interp(G.T_ANCHOR, P["t"], bc))
    return a, tau, float(np.min(np.diff(tau)))


def err_at(q, P, case, c, w_p, w_b, t0_seed):
    """c here is w_a - 1 (i.e. the fractional weight excess)."""
    a, tau, dmin = shape(P, c, w_p, w_b)
    if dmin <= 0:
        return 50.0
    best = None
    for key in ("fit", "full"):
        tb, hb, tn, hn = ((case["fit_t_bhpt"], case["fit_h_bhpt"], case["fit_t_nr"],
                           case["fit_h_nr"]) if key == "fit" else
                          (case["t_bhpt"], case["h_bhpt"], case["t_nr"], case["h_nr"]))
        aa = np.interp(tb, P["t"], a); tt = np.interp(tb, P["t"], tau)
        f = lambda t0: GG.err_at_t0(t0, aa, tt, hb, tn, hn)
        r = minimize_scalar(f, bounds=(t0_seed - 120.0, t0_seed + 120.0),
                            method="bounded", options={"xatol": 1e-3, "maxiter": 60})
        t0_seed = float(r.x); best = float(r.fun)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()

    mult = json.loads(G.RESULTS_DIR.joinpath("per_q_cache_mult.json").read_text())
    rows = sorted(mult.values(), key=lambda r: r["q"])
    t0p = np.polyfit(G.get_nu(np.array([r["q"] for r in rows])),
                     np.array([r["params"] for r in rows])[:, 4], 3)
    cases, preps = {}, {}
    def get(q):
        if q not in cases:
            cases[q] = FL.add_flux(G.load_case(q, 6, 10)); preps[q] = prep(q, cases[q])
        return preps[q], cases[q]

    print("Ehat at merger (where A1 actually applies):")
    for q in list(IN_Q) + list(LOW_Q):
        P, _ = get(q); print(f"   q={q:<5g} Ehm={P['Ehm']:.4f}  ga={P['ga']:+.4f}  "
                             f"gb={P['gb']:+.4f}", flush=True)

    def score(v, w_b, qs=IN_Q):
        c, w_p = v
        out = []
        for q in qs:
            P, cs = get(q)
            out.append(err_at(q, P, cs, c, w_p, w_b, float(np.polyval(t0p, G.get_nu(q)))))
        return out

    wbs = (-1.0, -0.5, 0.0, 0.5, 1.0) if args.scan else (args.wb,)
    print(f"\n{'w_b':>6s} {'w_a':>7s} {'w_p':>7s} {'median E':>11s} {'max E':>11s}")
    best_row = None
    for w_b in wbs:
        r = minimize(lambda v: float(np.mean(score(v, w_b))), [0.3, 1.0],
                     method="Nelder-Mead",
                     options={"maxiter": 120, "xatol": 1e-3, "fatol": 1e-11})
        c, w_p = r.x; v = score([c, w_p], w_b)
        print(f"{w_b:+6.1f} {1+c:7.3f} {w_p:7.3f} {np.median(v):11.4e} "
              f"{np.max(v):11.4e}", flush=True)
        if best_row is None or np.median(v) < best_row[3]:
            best_row = (w_b, c, w_p, float(np.median(v)))

    w_b, c, w_p, _ = best_row
    allq = list(IN_Q) + list(LOW_Q)
    final = dict(zip(allq, score([c, w_p], w_b, allq)))
    med = float(np.median([final[q] for q in IN_Q]))
    mx = float(np.max([final[q] for q in IN_Q]))
    print(f"\nbest: w_b={w_b:+.2f}  w_a={1+c:.3f}  w_p={w_p:.3f}")
    print(f"{'q':>6s} {'anchored3':>12s}  |  " + "  ".join(f"{k:>18}" for k in REF))
    for q in IN_Q:
        print(f"{q:>6g} {final[q]:>12.4e}")
    print(f"{'median':>6s} {med:>12.4e}  |  "
          + "  ".join(f"{REF[k]['median']:>18.4e}" for k in REF))
    print(f"{'max':>6s} {mx:>12.4e}  |  " + "  ".join(f"{REF[k]['max']:>18.4e}" for k in REF))
    print("  --- q < 3 ---")
    for q in LOW_Q:
        print(f"{q:>6g} {final[q]:>12.4e}  |  "
              + "  ".join(f"{REF[k][q]:>15.4e} {'B' if final[q]<REF[k][q] else 'w'}"
                          for k in REF))
    (RESULTS / "coeffs.json").write_text(json.dumps(
        {"model": "gwr_anchored3", "w_b": w_b, "w_a": 1 + c, "w_p": w_p,
         "in_range": {"median": med, "max": mx},
         "low_q": {f"{q:g}": final[q] for q in LOW_Q}, "reference": REF},
        indent=2, default=float))
    print(f"\nwrote {RESULTS/'coeffs.json'}")


if __name__ == "__main__":
    main()
