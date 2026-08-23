"""Reconstruction with poly alpha,beta, x clipped to measured range (no switch/QNM).
Isolates polynomial regression quality. Then compare adding QNM-corrected ringdown."""
import sys, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
ROOT=Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT.parent/"BHPTNRSurrogate"/"surrogates"))
import fit_scaling_PN_opt_creative as creative
WAVE=ROOT/".cache"/"q_dep"; PER_Q=ROOT/"peaks_results"/"per_q"

MA,NA=4,3; MB,NB=3,2
def design(x,nu,Mx,Nnu):
    return np.vstack([(nu**n)*(x**m) for n in range(1,Nnu+1) for m in range(0,Mx+1)]).T
cases={float(p.stem.replace("peaks_q","")):np.load(p) for p in sorted(PER_Q.glob("peaks_q*.npz"))}
def pool(k):
    out=[]
    for c in cases.values():
        v=c[k]; n=len(c["x"]); out.append(v if v.ndim else np.full(n,float(v)))
    return np.concatenate(out)
x_,nu_,a_,b_=pool("x"),pool("nu"),pool("alpha"),pool("beta_abs")
XLO,XHI=x_.min(),x_.max()
ca=np.linalg.lstsq(design(x_,nu_,MA,NA),a_-1,rcond=None)[0]
cb=np.linalg.lstsq(design(x_,nu_,MB,NB),b_-1,rcond=None)[0]
def amod(x,nu): xc=np.clip(x,XLO,XHI); return 1+design(np.atleast_1d(xc),np.atleast_1d(nu),MA,NA)@ca
def bmod(x,nu): xc=np.clip(x,XLO,XHI); return 1+design(np.atleast_1d(xc),np.atleast_1d(nu),MB,NB)@cb

def load(q):
    d=np.load(WAVE/f"waveforms_q{q:.10f}.npz")
    return d["t_b"] if False else (d["t_bhpt"],d["h_bhpt_re"]+1j*d["h_bhpt_im"],d["t_nr"],d["h_nr_re"]+1j*d["h_nr_im"])
def mism(a,b): return creative.mathcalE_error(a,b)

def recon(q):
    t_b,h_b,t_n,h_n=load(q); nu=q/(1+q)**2
    psib=np.unwrap(np.angle(h_b)); om=np.abs(np.gradient(psib,t_b)); xb=(om/2)**(2/3)
    m_b=int(np.argmax(np.abs(h_b))); m_n=int(np.argmax(np.abs(h_n)))
    trel=t_b-t_b[m_b]
    nuv=np.full_like(xb,nu)
    a=amod(xb,nuv); b=bmod(xb,nuv)
    tau=b*trel; tau_abs=tau+t_n[m_n]
    mono=np.r_[True,np.diff(tau)>0]
    use=(tau_abs>=t_n[0])&(tau_abs<=t_n[-1])&mono
    phi0=np.unwrap(np.angle(h_n))[m_n]-psib[m_b]
    def f(d):
        hs=a*np.exp(1j*(phi0+d))*h_b
        hi=creative.interp_complex(tau_abs[use],hs[use],t_n)
        cov=(t_n>=tau_abs[use][0])&(t_n<=tau_abs[use][-1])
        return mism(h_n[cov],hi[cov]),np.count_nonzero(cov)/len(t_n)
    r=minimize_scalar(lambda d:f(d)[0],bounds=(-np.pi,np.pi),method="bounded")
    e,cov=f(r.x); return e,cov

print(f"fit range x in [{XLO:.3f},{XHI:.3f}]; alpha {len(ca)} coeffs, beta {len(cb)} coeffs")
print("\nq      mathcalE       %       cov   note")
ir=[]
for q in [3.0,3.6,4.0,4.6,5.0,5.6,6.0,7.0,8.0]:
    with warnings.catch_warnings(): warnings.simplefilter("ignore"); e,cov=recon(q)
    ir.append(e); print(f"{q:5.2f}  {e:.4e}  {100*e:6.3f}  {cov:.3f}")
print(f"in-range median {np.median(ir):.3e}, max {np.max(ir):.3e}")
print()
for q in [2.75,2.5,2.25,2.0,1.75,1.5]:
    with warnings.catch_warnings(): warnings.simplefilter("ignore"); e,cov=recon(q)
    gate="  OK" if (q==2.0 and e<0.01) else ("  <-- q=2 GATE (<1%)" if q==2.0 else "")
    print(f"{q:5.2f}  {e:.4e}  {100*e:6.3f}  {cov:.3f}  q<3 EXTRAP{gate}")
