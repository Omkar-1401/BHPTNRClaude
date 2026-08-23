"""Localize the (4,4) q=2 failure: own-residual ceiling, inspiral/MR split, amp-vs-phase.
All at q=2 alone. No files written."""
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
T0,T1=creative.NR_T_START,creative.NR_T_END; XCLIP=0.26; LM=(4,4); Q=2.0
P38=[4.955,-42.351,-14.018,94.716]   # (4,4) residuals fit on [3,8] (prev run)
nrsur=gwsurrogate.LoadSurrogate("NRHybSur3dq8")

def mism(g,r):
    n1=np.sum(np.abs(r)**2); Z=np.abs(np.sum(r*np.conj(g))); n2=np.sum(np.abs(g)**2)
    return float((n1+n2-2*Z)/(2*n1))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    tb,hb=bhptsur.generate_surrogate(q=Q,calibrated=False,modes=[(2,2),LM],neg_modes=False)
    tn,hn,_=nrsur(Q,[0,0,0.0],[0,0,0.0],dt=0.1,f_low=5e-3); m=(tn>=T0)&(tn<=T1)
    t_n=tn[m]; hnr=hn[LM][m]
    nu=Q/(1+Q)**2
    meta={"nu":nu,"t_bhpt_merger":float(tb[np.argmax(np.abs(hb[(2,2)]))]),"t_nr_merger":0.0}
    pl=wfnu.wf_loss_coordinates(tb,hb[(2,2)],meta)["p_loss"]; x=PN.get_x(tb,hb[(2,2)])
    a22,beta=PN.model_ab(theta,Q,x,pl)
    bc=creative.cumulative_trapezoid(beta,tb); m_b=int(np.argmax(np.abs(hb[(2,2)]))); tau=bc-bc[m_b]
    use=np.r_[True,np.diff(tau)>0]&(tau>=T0)&(tau<=T1)
    l,mm=LM; eps=(l+mm)%2; C=1-3*nu; expo=(l+eps-2)/3.0
    amp0=(a22*C*beta**(-expo))[use]; xc=np.minimum(x,XCLIP)[use]; hpp=hb[LM][use]; tau_u=tau[use]

    def split(params):
        r1,r2,p1,p2=params
        h=amp0*np.exp(nu*(r1*xc+r2*xc**2))*np.exp(1j*nu*(p1*xc+p2*xc**2))*hpp
        def err(dt):
            tt=tau_u+dt; lo=max(tt[0],t_n[0]); hi=min(tt[-1],t_n[-1]); mk=(t_n>=lo)&(t_n<=hi)
            if mk.sum()<500: return 50.0,None
            g=creative.interp_complex(tt,h,t_n[mk]); return mism(g,hnr[mk]),mk
        grid=np.linspace(-250,250,401); c=grid[int(np.argmin([err(d)[0] for d in grid]))]
        r=minimize_scalar(lambda d:err(d)[0],bounds=(c-4,c+4),method="bounded",options={"xatol":1e-3}); dt=r.x
        _,mk=err(dt); tt=tau_u+dt; g=creative.interp_complex(tt,h,t_n[mk]); rr=hnr[mk]
        phi=np.angle(np.sum(rr*np.conj(g))); g=g*np.exp(1j*phi); tc=t_n[mk]
        full=mism(g,rr); insp=mism(g[tc<-100],rr[tc<-100]); mr=mism(g[tc>=-100],rr[tc>=-100])
        wf=np.sum(np.abs(rr[tc<-100])**2)/np.sum(np.abs(rr)**2)
        return full,insp,mr,wf
    def obj(p): return split(p)[0]

    print(f"(4,4) q=2 localization:")
    for lbl,p in [("pure backbone",[0,0,0,0]), ("[3,8] residuals",P38)]:
        f,i,r,wf=split(p); print(f"  {lbl:18s}: full={f:.3e}  insp={i:.3e}  MR={r:.3e}  (insp carries {100*wf:.0f}% power)")
    # own-ceiling fits at q=2
    full4=minimize(obj,[0,0,0,0],method="Powell",options={"maxiter":60,"xtol":1e-3,"ftol":1e-6}).x
    f,i,r,wf=split(full4); print(f"  q=2 OWN full fit  : full={f:.3e}  insp={i:.3e}  MR={r:.3e}   params={np.array2string(full4,precision=2)}")
    amp=minimize(lambda p:obj([p[0],p[1],0,0]),[0,0],method="Powell",options={"maxiter":60}).x
    fa=split([amp[0],amp[1],0,0])[0]; print(f"  q=2 amplitude-only: full={fa:.3e}   (r1,r2)={np.array2string(amp,precision=2)}")
    ph=minimize(lambda p:obj([0,0,p[0],p[1]]),[0,0],method="Powell",options={"maxiter":60}).x
    fp=split([0,0,ph[0],ph[1]])[0]; print(f"  q=2 phase-only    : full={fp:.3e}   (p1,p2)={np.array2string(ph,precision=2)}")
