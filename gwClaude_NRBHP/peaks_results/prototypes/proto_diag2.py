"""Is q<3 recoverable with the peaks method itself? Reconstruct q=2,2.5 with
their OWN measured alpha,beta (interpolated), to separate method vs extrapolation."""
import sys, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
ROOT=Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT.parent/"BHPTNRSurrogate"/"surrogates"))
import fit_scaling_PN_opt_creative as creative
import fit_scaling_peaks as peaks
WAVE=ROOT/".cache"/"q_dep"
def load(q):
    d=np.load(WAVE/f"waveforms_q{q:.10f}.npz")
    return d["t_bhpt"],d["h_bhpt_re"]+1j*d["h_bhpt_im"],d["t_nr"],d["h_nr_re"]+1j*d["h_nr_im"]
def mism(a,b): return creative.mathcalE_error(a,b)

def recon_measured(q):
    c=peaks.extract(q)
    t_b,h_b,t_n,h_n=load(q)
    psib=np.unwrap(np.angle(h_b)); m_b=int(np.argmax(np.abs(h_b))); m_n=int(np.argmax(np.abs(h_n)))
    trel=t_b-t_b[m_b]
    o=np.argsort(c["t_bhpt"]); tb=c["t_bhpt"][o]; ap=c["alpha"][o]; bp=c["beta_abs"][o]
    a=np.interp(trel,tb,ap); b=np.interp(trel,tb,bp)
    tau=b*trel; tau_abs=tau+t_n[m_n]; mono=np.r_[True,np.diff(tau)>0]
    use=(tau_abs>=t_n[0])&(tau_abs<=t_n[-1])&mono
    phi0=np.unwrap(np.angle(h_n))[m_n]-psib[m_b]
    def f(d):
        hs=a*np.exp(1j*(phi0+d))*h_b
        hi=creative.interp_complex(tau_abs[use],hs[use],t_n)
        cov=(t_n>=tau_abs[use][0])&(t_n<=tau_abs[use][-1])
        return mism(h_n[cov],hi[cov])
    r=minimize_scalar(f,bounds=(-np.pi,np.pi),method="bounded")
    return f(r.x), c

print("q     measured-recon mathcalE    n_peaks  x-range        a(lo->hi)      b(lo->hi)")
for q in [5.0,3.0,2.75,2.5,2.25,2.0,1.75,1.5]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e,c=recon_measured(q)
    o=np.argsort(c["x"])
    print(f"{q:4.2f}  {e:.4e} ({100*e:6.3f}%)   {c['n_peaks']:3d}   "
          f"[{c['x'].min():.3f},{c['x'].max():.3f}]  "
          f"{c['alpha'][o][0]:.3f}->{c['alpha'][o][-1]:.3f}  "
          f"{c['beta_abs'][o][0]:.3f}->{c['beta_abs'][o][-1]:.3f}")
