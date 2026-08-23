"""Add the note's PP-anchored residuals to the (3,3)/(4,4) backbone and re-check.
 rho_lm  = exp[ nu*(r1*xc + r2*xc^2) ]           (amplitude, ->1 as nu->0)
 dphi_lm = nu*(p1*xc + p2*xc^2)                   (phase, ->0 as nu->0)
Fit [r1,r2,p1,p2] per mode by GLOBAL joint mismatch on even-nu [3,8]; q=2 held out. No files written."""
import sys, warnings, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, minimize_scalar
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTutils"))
sys.path.insert(0, str(ROOT.parent/"BHPTNRSurrogate"/"surrogates"))
import fit_scaling_PN_opt_creative as creative
import fit_scaling_wf_nu_q_dep as wfnu
import fit_scaling_pn_anchored as PN
import BHPTNRSur1dq1e4 as bhptsur
import gwsurrogate

theta = np.array(json.load(open(ROOT/"pn_anchored_results"/"coeffs.json"))["theta"])
T0, T1 = creative.NR_T_START, creative.NR_T_END; XCLIP = 0.26
MODES = [(2,2),(3,3),(4,4)]
nrsur = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
TRAIN = [3.0,3.2564102564,3.5128205128,3.8974358974,4.2,4.6,5.0,5.4358974359,5.9487179487,6.5897435897,7.2307692308,8.0]
EVAL = [2.0, 2.5]

def gen(q):
    tb,hb = bhptsur.generate_surrogate(q=q,calibrated=False,modes=MODES,neg_modes=False)
    tn,hn,_ = nrsur(q,[0,0,0.0],[0,0,0.0],dt=0.1,f_low=5e-3); m=(tn>=T0)&(tn<=T1)
    return tb,hb,tn[m],{k:hn[k][m] for k in MODES}

def mism(g,r):
    n1=np.sum(np.abs(r)**2); Z=np.abs(np.sum(r*np.conj(g))); n2=np.sum(np.abs(g)**2)
    return float((n1+n2-2*Z)/(2*n1))

def precompute(q, tb, hb, t_n, hn):
    nu=q/(1+q)**2
    meta={"nu":nu,"t_bhpt_merger":float(tb[np.argmax(np.abs(hb[(2,2)]))]),"t_nr_merger":0.0}
    pl=wfnu.wf_loss_coordinates(tb,hb[(2,2)],meta)["p_loss"]; x=PN.get_x(tb,hb[(2,2)])
    a22,beta=PN.model_ab(theta,q,x,pl)
    bc=creative.cumulative_trapezoid(beta,tb); m_b=int(np.argmax(np.abs(hb[(2,2)]))); tau=bc-bc[m_b]
    use=np.r_[True,np.diff(tau)>0]&(tau>=T0)&(tau<=T1)
    d={}
    for lm in [(3,3),(4,4)]:
        l,m=lm; eps=(l+m)%2; C=(q-1)/(q+1) if lm==(3,3) else (1-3*nu); expo=(l+eps-2)/3.0
        d[lm]=dict(nu=nu, tau=tau[use], amp0=(a22*C*beta**(-expo))[use],
                   xc=np.minimum(x,XCLIP)[use], hpp=hb[lm][use], t_n=t_n, hnr=hn[lm], seed=0.0)
    return d

def align(tau_u, h_u, t_n, h_lm, seed, wide):
    def err(dt):
        tt=tau_u+dt; lo=max(tt[0],t_n[0]); hi=min(tt[-1],t_n[-1]); m=(t_n>=lo)&(t_n<=hi)
        if m.sum()<500: return 50.0
        return mism(creative.interp_complex(tt,h_u,t_n[m]), h_lm[m])
    if wide:
        grid=np.linspace(-250,250,401); seed=grid[int(np.argmin([err(x) for x in grid]))]
    r=minimize_scalar(err,bounds=(seed-8,seed+8),method="bounded",options={"xatol":1e-3,"maxiter":40})
    return r.fun, r.x

def model_err(params, d, wide=False):
    r1,r2,p1,p2=params
    rho=np.exp(d["nu"]*(r1*d["xc"]+r2*d["xc"]**2))
    dphi=d["nu"]*(p1*d["xc"]+p2*d["xc"]**2)
    h=d["amp0"]*rho*np.exp(1j*dphi)*d["hpp"]
    e,s=align(d["tau"],h,d["t_n"],d["hnr"],d["seed"],wide); d["seed"]=s; return e

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print("generating waveforms...", flush=True)
    pre={q:precompute(q,*gen(q)) for q in TRAIN+EVAL}
    for lm in [(3,3),(4,4)]:
        # seed t0 once (wide) at params=0
        for q in TRAIN+EVAL: model_err([0,0,0,0], pre[q][lm], wide=True)
        def obj(p, lm=lm):
            return float(np.mean([model_err(p, pre[q][lm]) for q in TRAIN]))
        base_ir=[model_err([0,0,0,0],pre[q][lm]) for q in TRAIN]
        res=minimize(obj,[0,0,0,0],method="Powell",options={"maxiter":30,"xtol":1e-3,"ftol":1e-5})
        p=res.x
        ir=[model_err(p,pre[q][lm]) for q in TRAIN]
        print(f"\n{lm}  residual params [r1,r2,p1,p2]={np.array2string(p,precision=3)}")
        print(f"  PURE backbone : in-range median={np.median(base_ir):.3e} max={np.max(base_ir):.3e}")
        print(f"  WITH residuals: in-range median={np.median(ir):.3e} max={np.max(ir):.3e}")
        for q in EVAL:
            e0=model_err([0,0,0,0],pre[q][lm]); e1=model_err(p,pre[q][lm])
            print(f"  q={q}: backbone={e0:.3e} -> with-residuals={e1:.3e}")
