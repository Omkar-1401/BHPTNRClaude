"""q=2: per-q TRUTH vs PN-anchored extrapolation, alpha(t) and beta(t)."""
import sys, warnings
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTutils"))
import fit_scaling_PN_opt_creative as creative
import fit_scaling_wf_nu_q_dep as wfnu
import fit_scaling_wf_nu_hybrid_q_dep as H
import surfinBH
_SFBH = surfinBH.LoadFits("NRSur3dq8Remnant")
QF1,QF2,QF3 = 1.5251,-1.1568,0.1292
THETA = np.array([0.352, 8.553, -22.9193, 3.2336, 2.2404, 1.9069, 0.6869, 0.7604, 3.9838])
XCLIP = 0.26
Q = 2.0

# ---- per-q truth (hybrid fit to q=2's own waveform), evaluated on NR common time ----
def ab_on_nr(params, case):
    ev = H.evaluate_model_hybrid(params, case["t_bhpt"], case["h_bhpt"],
                                 case["t_nr"], case["h_nr"], case["losses"])
    tau=ev["tau"]; common=ev["common"]; tc=case["t_nr"][common]
    return tc, np.interp(tc,tau,ev["alpha"]), np.interp(tc,tau,ev["beta"]), ev["error"]

# ---- PN-anchored model at q=2 ----
def get_x(t,h):
    ph=np.unwrap(np.angle(h)); n=len(ph); win=min(401,n-(1-n%2)); win=win-1 if win%2==0 else win
    if win>=11: ph=savgol_filter(ph,win,3,mode="interp")
    return np.clip((0.5*np.abs(np.gradient(ph,t)))**(2/3),1e-8,0.6)
def beta_r_phys(q):
    mf,_=_SFBH.mf(q,[0,0,0],[0,0,0]); chif,_=_SFBH.chif(q,[0,0,0],[0,0,0]); chif=float(chif[2])
    return (0.3683*(1+q)/q)/((QF1+QF2*(1-chif)**QF3)/float(mf))
def sig(z): return 1/(1+np.exp(-np.clip(z,-60,60)))
def pn_anchored(case):
    t,h = case["t_bhpt"], case["h_bhpt"]; pl=case["losses"]["p_loss"]; x=get_x(t,h)
    b1,b2,a2,p00,p01,w0,r0,m0,m1=THETA
    X1=Q/(1+Q); nu=Q/(1+Q)**2; base=X1**1.2; xc=np.minimum(x,XCLIP)
    S=sig((pl-(p00+p01*nu))/max(w0,1e-3))
    beta=(1-S)*base*(1+nu*(b1*xc+b2*xc**2))+S*beta_r_phys(Q)*r0
    alpha=(1-S)*base*(1+(55/42)*nu*xc+nu*a2*xc**2)+S*base*(m0+m1*nu)
    bc=creative.cumulative_trapezoid(beta,t); m_b=int(np.argmax(np.abs(h))); tau=bc-bc[m_b]
    use=np.r_[True,np.diff(tau)>0]&(tau>=creative.NR_T_START)&(tau<=creative.NR_T_END)
    return tau[use], alpha[use], beta[use]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    row=H.optimize_case_worker((Q,3,5,4,6000)); truth=np.array(row["params"],float)
    case=H.load_case(Q,3,5)
    tct,at,bt,et=ab_on_nr(truth,case)
    tcp,ap,bp=pn_anchored(case)
print(f"q=2: per-q truth mathcalE={et:.3e};  PN-anchored (from fit) ~2.55e-3")

fig,ax=plt.subplots(2,2,figsize=(13,8))
for col,(lo,hi) in enumerate([(-1500,80),(-200,80)]):
    ax[0,col].plot(tct,at,lw=1.9,label=f"per-q truth (E={et:.2e})")
    ax[0,col].plot(tcp,ap,"--",lw=1.5,label="PN-anchored extrap (E~2.6e-3)")
    ax[1,col].plot(tct,bt,lw=1.9,label="per-q truth")
    ax[1,col].plot(tcp,bp,"--",lw=1.5,label="PN-anchored extrap")
    for r in (0,1): ax[r,col].set_xlim(lo,hi); ax[r,col].grid(alpha=0.3); ax[r,col].axvline(0,ls=":",color="gray",lw=1)
ax[0,0].set_ylabel("alpha(t)"); ax[1,0].set_ylabel("beta(t)")
ax[0,0].set_title("alpha (full)"); ax[0,1].set_title("alpha (merger/ringdown zoom)")
ax[1,0].set_title("beta (full)"); ax[1,1].set_title("beta (merger/ringdown zoom)")
for a in ax[1]: a.set_xlabel("t_NR / M")
ax[0,0].legend(fontsize=8); ax[1,0].legend(fontsize=8)
fig.suptitle("q=2 [EXTRAP]: PN-anchored fit vs per-q truth  —  alpha & beta")
fig.tight_layout()
p=ROOT/"Agentic_plots"/"pn_anchored"/"pn_anchored_q2_vs_perq_truth.png"; fig.savefig(p,dpi=120)
print(f"saved {p}")
