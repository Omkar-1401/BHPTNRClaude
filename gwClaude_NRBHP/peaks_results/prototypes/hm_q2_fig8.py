"""q=2 waveform plot, Fig-8 style (arXiv:2204.01972): stacked panels (2,2)/(3,3)/(4,4),
Re(h_lm) vs t, NR vs calibrated model (pn_anchored (2,2) + backbone x note-residuals)."""
import sys, warnings, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib; matplotlib.use("Agg")
matplotlib.rcdefaults(); matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTutils"))
sys.path.insert(0, str(ROOT.parent/"BHPTNRSurrogate"/"surrogates"))
import fit_scaling_PN_opt_creative as creative
import fit_scaling_wf_nu_q_dep as wfnu
import fit_scaling_pn_anchored as PN
import BHPTNRSur1dq1e4 as bhptsur
import gwsurrogate
theta = np.array(json.load(open(ROOT/"pn_anchored_results"/"coeffs.json"))["theta"])
T0,T1=creative.NR_T_START,creative.NR_T_END; XCLIP=0.26; Q=2.0
RES={(3,3):[-0.26,-1.365,-18.906,56.367], (4,4):[4.955,-42.351,-14.018,94.716]}  # [3,8] fit
nrsur=gwsurrogate.LoadSurrogate("NRHybSur3dq8")

def mism(g,r):
    n1=np.sum(np.abs(r)**2); Z=np.abs(np.sum(r*np.conj(g))); n2=np.sum(np.abs(g)**2)
    return float((n1+n2-2*Z)/(2*n1))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    tb,hb=bhptsur.generate_surrogate(q=Q,calibrated=False,modes=[(2,2),(3,3),(4,4)],neg_modes=False)
    tn,hn,_=nrsur(Q,[0,0,0.0],[0,0,0.0],dt=0.1,f_low=5e-3); mk=(tn>=T0)&(tn<=T1)
    t_n=tn[mk]; nu=Q/(1+Q)**2
    meta={"nu":nu,"t_bhpt_merger":float(tb[np.argmax(np.abs(hb[(2,2)]))]),"t_nr_merger":0.0}
    pl=wfnu.wf_loss_coordinates(tb,hb[(2,2)],meta)["p_loss"]; x=PN.get_x(tb,hb[(2,2)])
    a22,beta=PN.model_ab(theta,Q,x,pl)
    bc=creative.cumulative_trapezoid(beta,tb); m_b=int(np.argmax(np.abs(hb[(2,2)]))); tau=bc-bc[m_b]
    use=np.r_[True,np.diff(tau)>0]&(tau>=T0)&(tau<=T1); xc=np.minimum(x,XCLIP)

    def model(lm):
        if lm==(2,2): h=a22*hb[(2,2)]
        else:
            l,m=lm; eps=(l+m)%2; C=(Q-1)/(Q+1) if lm==(3,3) else (1-3*nu); expo=(l+eps-2)/3.0
            r1,r2,p1,p2=RES[lm]
            h=(a22*C*beta**(-expo))*np.exp(nu*(r1*xc+r2*xc**2))*np.exp(1j*nu*(p1*xc+p2*xc**2))*hb[lm]
        return h
    def align(h, hnr):
        hu=h[use]; tu=tau[use]
        def err(dt):
            tt=tu+dt; lo=max(tt[0],t_n[0]); hi=min(tt[-1],t_n[-1]); m=(t_n>=lo)&(t_n<=hi)
            if m.sum()<500: return 50.0
            return mism(creative.interp_complex(tt,hu,t_n[m]),hnr[m])
        grid=np.linspace(-250,250,401); c=grid[int(np.argmin([err(d) for d in grid]))]
        dt=minimize_scalar(err,bounds=(c-4,c+4),method="bounded",options={"xatol":1e-3}).x
        tt=tu+dt; lo=max(tt[0],t_n[0]); hi=min(tt[-1],t_n[-1]); m=(t_n>=lo)&(t_n<=hi)
        g=creative.interp_complex(tt,hu,t_n[m]); phi=np.angle(np.sum(hnr[m]*np.conj(g)))
        return tu+dt, hu*np.exp(1j*phi), mism(g*np.exp(1j*phi),hnr[m])

    fig,ax=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for i,lm in enumerate([(2,2),(3,3),(4,4)]):
        hnr=hn[lm][mk]; tmod,hmod,e=align(model(lm),hnr)
        ax[i].plot(t_n,np.real(hnr),"k-",lw=1.2,label="NR (NRHybSur3dq8)")
        ax[i].plot(tmod,np.real(hmod),"r--",lw=1.1,label="pn-anchored (calibrated)")
        ax[i].set_xlim(-600,90); ax[i].grid(alpha=0.3); ax[i].axvline(0,ls=":",color="gray",lw=1)
        ax[i].set_ylabel(f"Re h_{{{lm[0]}{lm[1]}}}")
        ax[i].text(0.02,0.9,f"({lm[0]},{lm[1]})  mathcalE={e:.2e}",transform=ax[i].transAxes,fontsize=10,va="top")
        if i==0: ax[i].legend(fontsize=9,loc="upper right")
    ax[2].set_xlabel("t / M")
    fig.suptitle(f"q={Q:g} [EXTRAP]: pn-anchored calibrated vs NR — (2,2),(3,3),(4,4)  (Fig.8-style)")
    fig.tight_layout()
    out=ROOT/"Agentic_plots"/"pn_anchored_hm"; out.mkdir(parents=True,exist_ok=True)
    p=out/"q2_waveforms_224.png"; fig.savefig(p,dpi=130); print(f"saved {p}")
    for lm in [(2,2),(3,3),(4,4)]:
        _,_,e=align(model(lm),hn[lm][mk]); print(f"  {lm}: mathcalE={e:.3e}")
