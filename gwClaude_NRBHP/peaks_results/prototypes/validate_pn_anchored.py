"""Validate BHPTNRPNAnchored at q=8,5,3,2 (q=2 extrap): aligned mismatch + waveform plots.
Uses cached BHPT (monkeypatch) to skip the surrogate load."""
import sys, warnings, numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTutils"))
import fit_scaling_PN_opt_creative as creative
import fit_scaling_PN_opt_creative_q_dep as qdep
import BHPTNRPNAnchored as B
PLOT = ROOT/"Agentic_plots"/"pn_anchored"; PLOT.mkdir(parents=True, exist_ok=True)

def cached_bhpt(q):
    d = np.load(qdep.waveform_cache_path(q)); return d["t_bhpt"], d["h_bhpt_re"]+1j*d["h_bhpt_im"]
B._load_bhpt = cached_bhpt
def load_nr(q):
    d = np.load(qdep.waveform_cache_path(q)); return d["t_nr"], d["h_nr_re"]+1j*d["h_nr_im"]

def aligned(t, h, t_nr, h_nr):
    def err(dt):
        tt=t+dt; lo=max(tt[0],t_nr[0]); hi=min(tt[-1],t_nr[-1]); m=(t_nr>=lo)&(t_nr<=hi)
        if m.sum()<100: return 50.0
        g=creative.interp_complex(tt,h,t_nr[m]); r=h_nr[m]
        n1=np.sum(np.abs(r)**2); Z=np.abs(np.sum(r*np.conj(g))); n2=np.sum(np.abs(g)**2)
        return float((n1+n2-2*Z)/(2*n1))
    # wide coarse scan + refine (avoid narrow-bound bug)
    grid=np.linspace(-250,250,501); e=[err(x) for x in grid]; c=grid[int(np.argmin(e))]
    r=minimize_scalar(err,bounds=(c-4,c+4),method="bounded",options={"xatol":1e-3}); dt=r.x
    tt=t+dt; lo=max(tt[0],t_nr[0]); hi=min(tt[-1],t_nr[-1]); m=(t_nr>=lo)&(t_nr<=hi)
    g=creative.interp_complex(tt,h,t_nr[m]); phi=float(np.angle(np.sum(h_nr[m]*np.conj(g))))
    return r.fun, dt, phi

qs=[8.0,5.0,3.0,2.0]
fig,ax=plt.subplots(4,2,figsize=(14,12),gridspec_kw={"width_ratios":[3,2]})
print(f"{'q':>5} {'mathcalE (aligned)':>18}")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for r,q in enumerate(qs):
        t,h=B.generate_pn_anchored_calibrated(q); t_nr,h_nr=load_nr(q)
        e,dt,phi=aligned(t,h,t_nr,h_nr); hd=h*np.exp(1j*phi)
        print(f"{q:5.1f} {e:18.3e}")
        for col,(lo,hi) in enumerate([(-1000,-150),(-150,80)]):
            a=ax[r,col]; a.plot(t_nr,np.real(h_nr),"k-",lw=1.1,label="NR 22")
            a.plot(t+dt,np.real(hd),"r--",lw=1.0,alpha=0.85,label="PN-anchored")
            a.set_xlim(lo,hi); a.grid(alpha=0.3)
            if col==1: a.set_xlabel("t/M")
        ex="  [EXTRAP]" if q<3 else ""
        ax[r,0].set_ylabel(f"q={q}{ex}  E={e:.2e}")
        if r==0: ax[r,0].legend(fontsize=8,loc="upper left")
fig.suptitle("PN-anchored (2,2) vs NRHybSur3dq8  —  q=8,5,3 in-range, q=2 extrapolation")
fig.tight_layout(); p=PLOT/"pn_anchored_validation.png"; fig.savefig(p,dpi=95)
print(f"saved {p}")
