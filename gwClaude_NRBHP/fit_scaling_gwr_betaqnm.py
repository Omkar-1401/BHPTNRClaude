"""gwr_betaqnm -- beta pinned at BOTH ends, merger value FREE, ringdown = QNM plateau.

Requirement (user):
  * early inspiral: beta = X1^(6/5)                     IMPOSED
  * merger:         beta = free                         FITTED
  * ringdown:       beta = QNM value, and it PLATEAUS    IMPOSED

    Ehat(t) = Eoft/E_tot in [0,1]
    t_sat   = time where E STOPS GROWING (Ehat = 0.999) -- "merger" in the sense used
              here; measured at +38.7 (q=8) to +40.5 M (q=2) after the |h| peak, i.e.
              essentially q-universal.  ~75 M of window remains after it.
    u = clip(Ehat/0.999, 0, 1)                      0 -> 1 by t_sat; carries the WHOLE
                                                    E-driven rise in one leg
    v = clip((t-t_sat)/(t_end-t_sat), 0, 1)         0 -> 1 over the remaining ~75 M,
                                                    on TIME (E is flat there)

    beta(t,q) = B0 + (Bm - B0)*u  +  (B1 - Bm)*v
      B0 = X1^(6/5)*(1 + b*nu)                      b = 0 imposes the bare anchor
      B1 = W_Schw*Mf(q)/omega_220(chi_f(q))         gwModelRemS, IMPOSED
      Bm = B0*(1 + m*gb(q)),  gb = (B1-B0)/B0       m FITTED (merger value free)

  beta(start) = B0, beta(merger) = Bm, beta(t>=+30M) = B1 exactly, as a plateau.
  Rises through the inspiral iff m > 0; the merger->ringdown leg is free to go
  either way (falls if Bm > B1, i.e. m > 1).

    alpha(t,q) = X1^(6/5)*(1 + c0*nu + c1*nu^2) * (1 + (A/nu)*E(t))     3 fitted

NOTE on b: the measured beta_PP sits +0.08..0.16% above X1^(6/5), and RESUME records
that a 0.2% constant error in beta accumulates ~60 M of drift that t0 cannot absorb.
So `--bfree` (1 extra coefficient) is likely to matter; both are run for comparison.
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
import gwModels

RESULTS = ROOT / "gwr_betaqnm_results"; RESULTS.mkdir(exist_ok=True)
IN_Q = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0); LOW_Q = (2.75, 2.5, 2.25, 2.0)
W_SCHW = 0.3737
QNM = json.loads((ROOT/"pn_anchored_results"/"coeffs.json").read_text())["qnm"]
REF = {"fluxanchored":{"median":4.7516e-04,"max":7.1503e-04,2.75:6.3465e-04,
                       2.5:5.9369e-04,2.25:6.8673e-04,2.0:1.3056e-03},
       "anchored":    {"median":6.2551e-04,"max":9.7884e-04,2.75:9.4774e-04,
                       2.5:9.2793e-04,2.25:1.0163e-03,2.0:1.5293e-03}}

def prep(q, case):
    e = np.asarray(case["losses"]["e_oft"], float); t = np.asarray(case["t_bhpt"], float)
    Et = float(e[-1]-e[0]); Eh = (e-e[0])/Et
    # split at the |h| peak.  (The E-saturation split was tried and measured slightly
    # WORSE: median 3.83e-03 vs 3.78e-03 at 4 coef, 3.34e-03 vs 2.93e-03 at 5.)
    im = int(np.argmax(np.abs(case["h_bhpt"]))); Ehm = float(Eh[im])
    Ehr = float(np.interp(t[im]+30.0, t, Eh))
    u = np.clip(Eh/Ehm, 0.0, 1.0)
    v = np.clip((Eh-Ehm)/max(1e-9, Ehr-Ehm), 0.0, 1.0)
    r = gwModels.remnants.gwModelRemS(q, 0.0, 0.0)
    Mf, chif = float(r[0]), float(r[1])
    B1 = W_SCHW*Mf/(QNM["F1"]+QNM["F2"]*(1.0-chif)**QNM["F3"])
    return dict(t=t, e=e, Et=Et, u=u, v=v, Ehm=Ehm, Ehr=Ehr, B1=B1,
                base=(q/(1+q))**1.2, nu=q/(1+q)**2)

def shape(P, c0, c1, A, m, b):
    B0 = P["base"]*(1.0 + b*P["nu"])
    gb = (P["B1"]-B0)/B0
    Bm = B0*(1.0 + m*gb)
    beta = B0 + (Bm-B0)*P["u"] + (P["B1"]-Bm)*P["v"]
    alpha = P["base"]*(1.0+c0*P["nu"]+c1*P["nu"]**2)*(1.0+(A/P["nu"])*P["e"])
    return alpha, beta, B0, Bm, gb

def err(q, P, case, th, bfree):
    c0,c1,A,m = th[:4]; b = th[4] if bfree else 0.0
    alpha, beta, *_ = shape(P, c0, c1, A, m, b)
    if np.any(beta <= 0): return 50.0
    bc = G.creative.cumulative_trapezoid(beta, P["t"])
    tau = bc - float(np.interp(G.T_ANCHOR, P["t"], bc))
    if np.min(np.diff(tau)) <= 0: return 50.0
    out, t0 = None, -75.0
    for key in ("fit","full"):
        tb,hb,tn,hn = ((case["fit_t_bhpt"],case["fit_h_bhpt"],case["fit_t_nr"],case["fit_h_nr"])
                       if key=="fit" else
                       (case["t_bhpt"],case["h_bhpt"],case["t_nr"],case["h_nr"]))
        aa=np.interp(tb,P["t"],alpha); tt=np.interp(tb,P["t"],tau)
        r=minimize_scalar(lambda s: GG.err_at_t0(s,aa,tt,hb,tn,hn),
                          bounds=(t0-120.,t0+120.),method="bounded",
                          options={"xatol":1e-3,"maxiter":60})
        t0=float(r.x); out=float(r.fun)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--bfree",action="store_true")
    ap.add_argument("--parfree",action="store_true",
                    help="PARAMETER-FREE beta: m=1 (Bm=B1, anchors alone set beta) and "
                         "b=0 (bare X1^(6/5)).  Only alpha's 3 coefficients are fitted.")
    a=ap.parse_args()
    cases,preps={},{}
    def get(q):
        if q not in cases:
            cases[q]=FL.add_flux(G.load_case(q,6,10)); preps[q]=prep(q,cases[q])
        return preps[q],cases[q]
    print(f"{'q':>6s} {'Ehat_sat':>9s} {'t_sat-t_pk':>11s} {'B0':>8s} {'B1(QNM)':>9s} {'gb':>9s}")
    for q in list(IN_Q)+list(LOW_Q):
        P,_=get(q); B0=P["base"]
        print(f"{q:6.2f} {P['Ehm']:9.4f} {P['Ehr']:11.1f} {B0:8.5f} {P['B1']:9.5f} "
              f"{(P['B1']-B0)/B0:+9.5f}", flush=True)
    if a.parfree:
        # beta has NO free coefficients: m == 1 so Bm == B1, b == 0 so B0 == X1^(6/5).
        x0=[0.286,-0.402,-1.072]
        obj=lambda t3: float(np.mean([err(q,*get(q),np.r_[t3,1.0],False) for q in IN_Q]))
        r=minimize(obj,x0,method="Nelder-Mead",
                   options={"maxiter":600,"xatol":1e-5,"fatol":1e-13})
        th=np.r_[r.x,1.0]
    else:
        x0=[0.286,-0.402,-1.072,1.0]+([0.0126] if a.bfree else [])
        obj=lambda th: float(np.mean([err(q,*get(q),th,a.bfree) for q in IN_Q]))
        r=minimize(obj,x0,method="Nelder-Mead",
                   options={"maxiter":900,"xatol":1e-5,"fatol":1e-13})
        th=r.x
    lbl=("c0 c1 A  [beta PARAMETER-FREE: m=1, b=0]" if a.parfree
         else "c0 c1 A m" + (" b" if a.bfree else ""))
    nfit = 3 if a.parfree else len(th)
    print(f"\nfitted ({lbl}) = " + "  ".join(f"{v:.4f}" for v in th[:nfit])
          + f"     [{nfit} fitted coefficients]")
    allq=list(IN_Q)+list(LOW_Q); fin={q:err(q,*get(q),th,a.bfree) for q in allq}
    med=float(np.median([fin[q] for q in IN_Q])); mx=float(np.max([fin[q] for q in IN_Q]))
    print(f"\n{'q':>6s} {'betaqnm':>11s} | " + " ".join(f"{k:>13}" for k in REF))
    for q in IN_Q: print(f"{q:>6g} {fin[q]:>11.4e}")
    print(f"{'median':>6s} {med:>11.4e} | " + " ".join(f"{REF[k]['median']:>13.4e}" for k in REF))
    print(f"{'max':>6s} {mx:>11.4e} | " + " ".join(f"{REF[k]['max']:>13.4e}" for k in REF))
    for q in LOW_Q:
        print(f"{q:>6g} {fin[q]:>11.4e} | " +
              " ".join(f"{REF[k][q]:>11.4e} {'B' if fin[q]<REF[k][q] else 'w'}" for k in REF))
    print(f"\n[verify] beta at the three corners:")
    print(f"{'q':>6s} {'B0':>9s} {'Bm(fit)':>9s} {'B1(QNM)':>9s} {'insp rise':>10s} "
          f"{'merg->ring':>11s}")
    for q in allq:
        P,_=get(q); c0,c1,A,m=th[:4]; b=th[4] if a.bfree else 0.0
        _,beta,B0,Bm,gb=shape(P,c0,c1,A,m,b)
        print(f"{q:6.2f} {B0:9.5f} {Bm:9.5f} {P['B1']:9.5f} "
              f"{(Bm-B0)/B0*100:+9.2f}% {(P['B1']-Bm)/Bm*100:+10.2f}%")
    (RESULTS/f"coeffs{'_parfree' if a.parfree else ('_bfree' if a.bfree else '')}.json").write_text(json.dumps(
        {"model":"gwr_betaqnm","bfree":a.bfree,"theta":th.tolist(),
         "in_range":{"median":med,"max":mx},
         "low_q":{f"{q:g}":fin[q] for q in LOW_Q},"reference":REF},indent=2,default=float))
    print("\nwrote", RESULTS/f"coeffs{'_parfree' if a.parfree else ('_bfree' if a.bfree else '')}.json")

if __name__=="__main__": main()
