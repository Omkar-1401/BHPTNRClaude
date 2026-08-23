"""alpha(t) of the PN-anchored (2,2) fit across q=2.0..3.0 step 0.1, two panels.
Uses the fitted theta from pn_anchor_fit.py. BHPT-only (cache or surrogate). Diagnostic."""
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
# fitted theta from pn_anchor_fit.py
THETA = np.array([0.352, 8.553, -22.9193, 3.2336, 2.2404, 1.9069, 0.6869, 0.7604, 3.9838])
XCLIP = 0.26

def load_bhpt(q):
    p = CACHE/f"waveforms_q{q:.10f}.npz"
    if p.exists():
        d = np.load(p); return d["t_bhpt"], d["h_bhpt_re"]+1j*d["h_bhpt_im"]
    import BHPTNRSur1dq1e4 as bhptsur
    t, hd = bhptsur.generate_surrogate(q=q, calibrated=False, modes=[(2,2)], neg_modes=False)
    print(f"  (surrogate q={q})", flush=True)
    return t, hd[(2,2)]

def get_x(t, h):
    ph = np.unwrap(np.angle(h)); n = len(ph)
    win = min(401, n-(1-n%2)); win = win-1 if win%2==0 else win
    if win >= 11: ph = savgol_filter(ph, win, 3, mode="interp")
    om = np.abs(np.gradient(ph, t))
    return np.clip((0.5*om)**(2/3), 1e-8, 0.6)

def beta_r_phys(q):
    mf,_=_SFBH.mf(q,[0,0,0],[0,0,0]); chif,_=_SFBH.chif(q,[0,0,0],[0,0,0]); chif=float(chif[2])
    return (0.3683*(1+q)/q)/((QF1+QF2*(1-chif)**QF3)/float(mf))

def sig(z): return 1/(1+np.exp(-np.clip(z,-60,60)))

def alpha_of_t(q):
    t, h = load_bhpt(q)
    meta = {"nu": q/(1+q)**2, "t_bhpt_merger": float(t[np.argmax(np.abs(h))]), "t_nr_merger": 0.0}
    los = wfnu.wf_loss_coordinates(t, h, meta); pl = los["p_loss"]
    x = get_x(t, h)
    b1,b2,a2,p00,p01,w0,r0,m0,m1 = THETA
    X1=q/(1+q); nu=q/(1+q)**2; base=X1**1.2; xc=np.minimum(x,XCLIP)
    S = sig((pl-(p00+p01*nu))/max(w0,1e-3))
    beta = (1-S)*base*(1+nu*(b1*xc+b2*xc**2)) + S*beta_r_phys(q)*r0
    alpha = (1-S)*base*(1+(55/42)*nu*xc+nu*a2*xc**2) + S*base*(m0+m1*nu)
    bc = creative.cumulative_trapezoid(beta, t)
    m_b = int(np.argmax(np.abs(h)))
    tau = bc - bc[m_b]
    use = np.r_[True, np.diff(tau)>0] & (tau >= creative.NR_T_START) & (tau <= creative.NR_T_END)
    return tau[use], alpha[use]

LEFT  = [2.0, 2.1, 2.2, 2.3, 2.4]
RIGHT = [2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    data = {q: alpha_of_t(q) for q in LEFT+RIGHT}

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
cmL = plt.cm.viridis(np.linspace(0,0.9,len(LEFT)))
cmR = plt.cm.plasma(np.linspace(0,0.9,len(RIGHT)))
for q,c in zip(LEFT,cmL):
    t,a = data[q]; ax[0].plot(t,a,color=c,lw=1.6,label=f"q={q:g}")
for q,c in zip(RIGHT,cmR):
    t,a = data[q]; ax[1].plot(t,a,color=c,lw=1.6,label=f"q={q:g}")
for i,ttl in enumerate(["q = 2.0 – 2.4","q = 2.5 – 3.0"]):
    ax[i].set_xlim(-600,80); ax[i].axvline(0,ls=":",color="gray",lw=1); ax[i].grid(alpha=0.3)
    ax[i].set_xlabel("t_NR / M"); ax[i].set_title(ttl); ax[i].legend(fontsize=8,ncol=2)
ax[0].set_ylabel("alpha(t)")
fig.suptitle("PN-anchored (2,2) fit: alpha(t) across q=2..3 (step 0.1)")
fig.tight_layout()
out = ROOT/"Agentic_plots"/"pn_anchored"; out.mkdir(parents=True, exist_ok=True)
p = out/"pn_anchored_alpha_fine_2to3.png"; fig.savefig(p, dpi=120)
print(f"saved {p}", flush=True)
