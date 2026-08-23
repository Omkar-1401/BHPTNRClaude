"""Pure-backbone (rho=1, dphi=0) transfer of the pn_anchored (2,2) calibration to (3,3),(4,4).
h_model_lm(tau) = [alpha22 * C_lm(nu) * beta^-((l+eps-2)/3)] * h_pp_lm(t), tau = (2,2) time map.
Standard mismatch alignment (wide t0 search + analytic phi0). Reports in-range + q=2. No files written."""
import sys, warnings, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTutils"))
sys.path.insert(0, str(ROOT.parent/"BHPTNRSurrogate"/"surrogates"))
import fit_scaling_PN_opt_creative as creative
import fit_scaling_wf_nu_q_dep as wfnu
import fit_scaling_pn_anchored as PN
import BHPTNRSur1dq1e4 as bhptsur
import gwsurrogate

theta = np.array(json.load(open(ROOT/"pn_anchored_results"/"coeffs.json"))["theta"])
T0, T1 = creative.NR_T_START, creative.NR_T_END
MODES = [(2,2),(3,3),(4,4)]
nrsur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
IN_RANGE = [3.0,4.0,5.0,6.0,7.0,8.0]; LOWQ = [2.0]

def gen(q):
    tb,hb = bhptsur.generate_surrogate(q=q,calibrated=False,modes=MODES,neg_modes=False)
    tn,hn,_ = nrsur(q,[0,0,0.0],[0,0,0.0],dt=0.1,f_low=5e-3)
    m=(tn>=T0)&(tn<=T1)
    return tb,hb,tn[m],{k:hn[k][m] for k in MODES}

def mism(g,r):
    n1=np.sum(np.abs(r)**2); Z=np.abs(np.sum(r*np.conj(g))); n2=np.sum(np.abs(g)**2)
    return float((n1+n2-2*Z)/(2*n1))

def align_err(tau_u, h_u, t_n, h_lm):
    def err(dt):
        tt=tau_u+dt; lo=max(tt[0],t_n[0]); hi=min(tt[-1],t_n[-1]); m=(t_n>=lo)&(t_n<=hi)
        if m.sum()<500: return 50.0
        g=creative.interp_complex(tt,h_u,t_n[m]); return mism(g,h_lm[m])
    grid=np.linspace(-250,250,401); c=grid[int(np.argmin([err(x) for x in grid]))]
    return minimize_scalar(err,bounds=(c-4,c+4),method="bounded",options={"xatol":1e-3}).fun

def eval_q(q, tb, hb, t_n, hn):
    nu=q/(1+q)**2
    meta={"nu":nu,"t_bhpt_merger":float(tb[np.argmax(np.abs(hb[(2,2)]))]),"t_nr_merger":0.0}
    pl=wfnu.wf_loss_coordinates(tb,hb[(2,2)],meta)["p_loss"]; x=PN.get_x(tb,hb[(2,2)])
    a22,beta=PN.model_ab(theta,q,x,pl)
    bc=creative.cumulative_trapezoid(beta,tb); m_b=int(np.argmax(np.abs(hb[(2,2)]))); tau=bc-bc[m_b]
    use=np.r_[True,np.diff(tau)>0]&(tau>=T0)&(tau<=T1)
    out={}
    # (2,2) control
    out[(2,2)]=align_err(tau[use], (a22*hb[(2,2)])[use], t_n, hn[(2,2)])
    for lm in [(3,3),(4,4)]:
        l,m=lm; eps=(l+m)%2; C=(q-1)/(q+1) if lm==(3,3) else (1-3*nu); expo=(l+eps-2)/3.0
        amp=a22*C*beta**(-expo)
        out[lm]=align_err(tau[use], (amp*hb[lm])[use], t_n, hn[lm])
    return out

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res={lm:{} for lm in MODES}
    for q in IN_RANGE+LOWQ:
        tb,hb,tn,hn=gen(q); o=eval_q(q,tb,hb,tn,hn)
        for lm in MODES: res[lm][q]=o[lm]
        print(f"q={q}: (2,2)={o[(2,2)]:.3e}  (3,3)={o[(3,3)]:.3e}  (4,4)={o[(4,4)]:.3e}", flush=True)
    print("\n=== SUMMARY (pure backbone, rho=1, dphi=0) ===")
    for lm in MODES:
        ir=[res[lm][q] for q in IN_RANGE]
        print(f"{lm}: in-range[3,8] median={np.median(ir):.3e} max={np.max(ir):.3e}   q=2={res[lm][2.0]:.3e}")
