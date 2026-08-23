"""alpha(t) of the PN-anchored (2,2) fit for q=2,3,5,8 (single panel, q=2 dashed)."""
import sys, warnings
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTutils"))
sys.path.insert(0, str(ROOT.parent/"BHPTNRSurrogate"/"surrogates"))
import fit_scaling_PN_opt_creative as creative
import fit_scaling_wf_nu_q_dep as wfnu
import surfinBH
_SFBH = surfinBH.LoadFits("NRSur3dq8Remnant")
QF1, QF2, QF3 = 1.5251, -1.1568, 0.1292
CACHE = ROOT/".cache"/"q_dep"
THETA = np.array([0.352, 8.553, -22.9193, 3.2336, 2.2404, 1.9069, 0.6869, 0.7604, 3.9838])
XCLIP = 0.26

def load_bhpt(q):
    d = np.load(CACHE/f"waveforms_q{q:.10f}.npz"); return d["t_bhpt"], d["h_bhpt_re"]+1j*d["h_bhpt_im"]
def get_x(t, h):
    ph = np.unwrap(np.angle(h)); n=len(ph); win=min(401,n-(1-n%2)); win=win-1 if win%2==0 else win
    if win>=11: ph = savgol_filter(ph, win, 3, mode="interp")
    return np.clip((0.5*np.abs(np.gradient(ph,t)))**(2/3), 1e-8, 0.6)
def beta_r_phys(q):
    mf,_=_SFBH.mf(q,[0,0,0],[0,0,0]); chif,_=_SFBH.chif(q,[0,0,0],[0,0,0]); chif=float(chif[2])
    return (0.3683*(1+q)/q)/((QF1+QF2*(1-chif)**QF3)/float(mf))
def sig(z): return 1/(1+np.exp(-np.clip(z,-60,60)))
def alpha_of_t(q):
    t,h = load_bhpt(q)
    meta={"nu":q/(1+q)**2,"t_bhpt_merger":float(t[np.argmax(np.abs(h))]),"t_nr_merger":0.0}
    pl = wfnu.wf_loss_coordinates(t,h,meta)["p_loss"]; x=get_x(t,h)
    b1,b2,a2,p00,p01,w0,r0,m0,m1=THETA
    X1=q/(1+q); nu=q/(1+q)**2; base=X1**1.2; xc=np.minimum(x,XCLIP)
    S=sig((pl-(p00+p01*nu))/max(w0,1e-3))
    beta=(1-S)*base*(1+nu*(b1*xc+b2*xc**2))+S*beta_r_phys(q)*r0
    alpha=(1-S)*base*(1+(55/42)*nu*xc+nu*a2*xc**2)+S*base*(m0+m1*nu)
    bc=creative.cumulative_trapezoid(beta,t); m_b=int(np.argmax(np.abs(h))); tau=bc-bc[m_b]
    use=np.r_[True,np.diff(tau)>0]&(tau>=creative.NR_T_START)&(tau<=creative.NR_T_END)
    return tau[use], alpha[use]

QS=[8.0,5.0,3.0,2.0]; COL={8.0:"#1f77b4",5.0:"#2ca02c",3.0:"#ff7f0e",2.0:"#d62728"}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    data={q:alpha_of_t(q) for q in QS}
fig,ax=plt.subplots(figsize=(9,5.5))
for q in QS:
    t,a=data[q]; ls="--" if q==2.0 else "-"; lw=2.2 if q==2.0 else 1.5
    ax.plot(t,a,ls,color=COL[q],lw=lw,label=f"q={q:g}"+(" [extrap]" if q<3 else ""))
ax.set_xlim(-800,80); ax.axvline(0,ls=":",color="gray",lw=1); ax.grid(alpha=0.3)
ax.set_xlabel("t_NR / M"); ax.set_ylabel("alpha(t)")
ax.set_title("PN-anchored (2,2) fit: alpha(t) for q=2,3,5,8  (q=2 dashed)")
ax.legend(fontsize=9)
fig.tight_layout()
out=ROOT/"Agentic_plots"/"pn_anchored"/"pn_anchored_alpha_2358.png"; fig.savefig(out,dpi=120)
print(f"saved {out}")
