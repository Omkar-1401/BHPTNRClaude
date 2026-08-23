"""gwr_qnmshape -- beta's nu-SHAPE from the QNM/RemS anchor, one global scale fitted.

    Ehat(t) = Eoft/E_tot in [0,1]

    alpha(t,q) = X1^(6/5)*(1 + c0*nu + c1*nu^2) * (1 + (A/nu)*E(t))     3 fitted
    beta (t,q) = X1^(6/5) * (1 + w*gb(q)*Ehat(t))                       1 fitted

    B0 = X1^(6/5)                                   early-inspiral anchor (derived)
    B1 = W_Schw*Mf(q)/omega_220(chi_f(q))            ringdown value from gwModelRemS
    gb = (B1-B0)/B0  > 0 at every q                  supplies beta's nu-shape only

w sets the scale, so beta at merger/ringdown is NOT pinned to B1 (w != 1 allowed).
beta rises at every q for any w > 0, since gb > 0 and Ehat is monotone.

New vs everything tried before: beta's coupling has the anchor's nu-shape with ONE
global scale (beta_monotone let a rectified line pick the coupling per nu, which is how
it zeroed high q while keeping low q positive -- a single w cannot), AND alpha's
coupling is FREE (the contamination in gwr_anchored2/3).
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); warnings.filterwarnings("ignore")
import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_global as GG
import fit_scaling_gwr_energy_flux as FL
import gwModels

RESULTS = ROOT / "gwr_qnmshape_results"; RESULTS.mkdir(exist_ok=True)
IN_Q = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
LOW_Q = (2.75, 2.5, 2.25, 2.0)
W_SCHW = 0.3737
QNM = json.loads((ROOT/"pn_anchored_results"/"coeffs.json").read_text())["qnm"]
REF = {"fluxanchored": {"median":4.7516e-04,"max":7.1503e-04,2.75:6.3465e-04,
                        2.5:5.9369e-04,2.25:6.8673e-04,2.0:1.3056e-03},
       "anchored":     {"median":6.2551e-04,"max":9.7884e-04,2.75:9.4774e-04,
                        2.5:9.2793e-04,2.25:1.0163e-03,2.0:1.5293e-03}}

def gb_of(q):
    r = gwModels.remnants.gwModelRemS(q, 0.0, 0.0)
    Mf, chif = float(r[0]), float(r[1])
    w_nr = (QNM["F1"] + QNM["F2"]*(1.0-chif)**QNM["F3"]) / Mf
    B0 = (q/(1+q))**1.2
    return (W_SCHW/w_nr - B0)/B0

def params_at(q, c0, c1, A, w, gb, Etot):
    nu = q/(1+q)**2; base = (q/(1+q))**1.2
    a_pp = base*(1.0 + c0*nu + c1*nu**2)
    return np.array([a_pp, A/nu, base, w*gb/Etot])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--wgrid", type=float, nargs="+",
        default=[-1.5,-1.0,-0.5,0.0,0.25,0.5,1.0,1.5]); args = ap.parse_args()
    mult = json.loads(G.RESULTS_DIR.joinpath("per_q_cache_mult.json").read_text())
    rows = sorted(mult.values(), key=lambda r: r["q"])
    t0p = np.polyfit(G.get_nu(np.array([r["q"] for r in rows])),
                     np.array([r["params"] for r in rows])[:,4], 3)
    cases, meta = {}, {}
    def get(q):
        if q not in cases:
            cases[q] = FL.add_flux(G.load_case(q,6,10))
            e = np.asarray(cases[q]["losses"]["e_oft"],float)
            meta[q] = (gb_of(q), float(e[-1]-e[0]))
        return cases[q], meta[q]
    print(f"{'q':>6s} {'gb (QNM shape)':>15s} {'E_tot':>9s}")
    for q in list(IN_Q)+list(LOW_Q):
        _, (gb, Et) = get(q); print(f"{q:6.2f} {gb:+15.5f} {Et:9.5f}", flush=True)

    def err(q, th, w):
        case,(gb,Et) = get(q)
        p = params_at(q, th[0], th[1], th[2], w, gb, Et)
        s = float(np.polyval(t0p, G.get_nu(q)))
        e,t0 = GG.fast_mismatch(p, case, s, "fit")
        return GG.fast_mismatch(p, case, t0, "full")[0]

    print(f"\n{'w':>6s} {'c0':>8s} {'c1':>8s} {'A':>9s} {'median':>11s} {'max':>11s} "
          f"{'dbeta q3':>9s} {'dbeta q8':>9s}")
    best = None
    for w in args.wgrid:
        r = minimize(lambda th: float(np.mean([err(q,th,w) for q in IN_Q])),
                     [0.286,-0.402,-1.072], method="Nelder-Mead",
                     options={"maxiter":250,"xatol":1e-4,"fatol":1e-12})
        th = r.x; v = [err(q,th,w) for q in IN_Q]
        d3 = w*meta[3.0][0]*100; d8 = w*meta[8.0][0]*100
        print(f"{w:+6.2f} {th[0]:8.4f} {th[1]:8.4f} {th[2]:9.4f} {np.median(v):11.4e} "
              f"{np.max(v):11.4e} {d3:+8.2f}% {d8:+8.2f}%", flush=True)
        if best is None or np.median(v) < best[0]:
            best = (float(np.median(v)), w, th.copy())
    med, w, th = best
    print(f"\nbest overall: w={w:+.2f}  median={med:.4e}")
    pos = [x for x in args.wgrid if x > 0]
    print(f"\n{'q':>6s} {'best-overall':>13s} | " + "  ".join(f"{k:>13}" for k in REF))
    allq = list(IN_Q)+list(LOW_Q)
    fin = {q: err(q, th, w) for q in allq}
    for q in IN_Q: print(f"{q:>6g} {fin[q]:>13.4e}")
    print(f"{'median':>6s} {np.median([fin[q] for q in IN_Q]):>13.4e} | " +
          "  ".join(f"{REF[k]['median']:>13.4e}" for k in REF))
    for q in LOW_Q:
        print(f"{q:>6g} {fin[q]:>13.4e} | " +
              "  ".join(f"{REF[k][q]:>11.4e} {'B' if fin[q]<REF[k][q] else 'w'}" for k in REF))
    (RESULTS/"coeffs.json").write_text(json.dumps(
        {"model":"gwr_qnmshape","w":w,"c0":th[0],"c1":th[1],"A":th[2],
         "in_range":{"median":float(np.median([fin[q] for q in IN_Q])),
                     "max":float(np.max([fin[q] for q in IN_Q]))},
         "low_q":{f"{q:g}":fin[q] for q in LOW_Q},"reference":REF}, indent=2, default=float))
    print(f"\nwrote {RESULTS/'coeffs.json'}")

if __name__ == "__main__":
    main()
