"""How bad is the merger+ringdown exclusion, and can a simple merger anchor fix it?
Time-map recon (train [2.5,8]); build the map from inspiral peaks PLUS the exact
merger anchor (trel=0 -> tau=0), and test full-NR-window mismatch vs the
inspiral-only window.  For alpha past the last peak, hold the last-peak value
(placeholder for a proper QNM/remnant amplitude branch).
Prints coverage so we know exactly what window is scored.
"""
import sys, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTNRSurrogate"/"surrogates"))
import fit_scaling_PN_opt_creative as creative
import fit_scaling_peaks as peaks
WAVE = ROOT/".cache"/"q_dep"; PER_Q = ROOT/"peaks_results"/"per_q"
base = {float(p.stem.replace("peaks_q","")): dict(np.load(p))
        for p in sorted(PER_Q.glob("peaks_q*.npz"))}
def get_case(q):
    if q in base: return base[q]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); return peaks.extract(q)
for q in [2.25,2.5,2.75]: base.setdefault(q, get_case(q))
allx=np.concatenate([base[q]["x"] for q in sorted(base)]); XLO,XHI=allx.min(),allx.max()
XMID,XHALF=0.5*(XLO+XHI),0.5*(XHI-XLO)
def u_of(x): return (np.clip(x,XLO,XHI)-XMID)/XHALF
def load(q):
    d=np.load(WAVE/f"waveforms_q{q:.10f}.npz")
    return d["t_bhpt"],d["h_bhpt_re"]+1j*d["h_bhpt_im"],d["t_nr"],d["h_nr_re"]+1j*d["h_nr_im"]
def cheb_fit(u,y,deg): return np.linalg.lstsq(np.polynomial.chebyshev.chebvander(u,deg),y-1.0,rcond=None)[0]
def cheb_eval(c,u): return 1.0+np.polynomial.chebyshev.chebval(u,c)
def peaks_of(t,h):
    amp=np.abs(h); t_m=float(t[int(np.argmax(amp))]); psi=np.unwrap(np.angle(h))
    sign=1.0 if psi[-1]>psi[0] else -1.0; psim=sign*psi
    psi_m=float(np.interp(t_m,t,psim)); n=int((psi_m-psim[0])/np.pi)
    tgt=psi_m-np.pi*np.arange(0,n+1); tgt=tgt[tgt>=psim[0]]
    t_pk=np.interp(tgt,psim,t); om=np.abs(np.gradient(psi,t)); return t_m,t_pk,np.interp(t_pk,t,om)

DA,DB,DNA,DNB=5,4,4,4
QT=[q for q in sorted(base) if q>=2.5-1e-9]
nus=np.array([q/(1+q)**2 for q in QT])
A=np.array([cheb_fit(u_of(base[q]["x"]),base[q]["alpha"],DA) for q in QT])
B=np.array([cheb_fit(u_of(base[q]["x"]),base[q]["beta_abs"],DB) for q in QT])
def reg(C,dz):
    Az=np.vstack([nus**k for k in range(1,dz+1)]).T
    return [np.linalg.lstsq(Az,C[:,m],rcond=None)[0] for m in range(C.shape[1])]
GA,GB=reg(A,DNA),reg(B,DNB)
def evalc(G,nu): return np.array([sum(g[k]*nu**(k+1) for k in range(len(g))) for g in G])

def recon(q, add_merger):
    t_b,h_b,t_n,h_n=load(q); nu=q/(1+q)**2
    m_b=int(np.argmax(np.abs(h_b))); m_n=int(np.argmax(np.abs(h_n)))
    tm_b,tpk_b,om_pk=peaks_of(t_b,h_b); trel_pk=tpk_b-tm_b; x_pk=(om_pk/2)**(2/3)
    a_pk=cheb_eval(evalc(GA,nu),u_of(x_pk)); b_pk=cheb_eval(evalc(GB,nu),u_of(x_pk))
    trel_nr_pk=b_pk*trel_pk
    o=np.argsort(trel_pk); tb_s=trel_pk[o]; tn_s=trel_nr_pk[o]; a_s=a_pk[o]
    keep=np.r_[True,np.diff(tn_s)>0]; tb_s=tb_s[keep]; tn_s=tn_s[keep]; a_s=a_s[keep]
    if add_merger:                       # exact anchor: merger->merger, alpha held
        tb_s=np.r_[tb_s, 0.0]; tn_s=np.r_[tn_s, 0.0]; a_s=np.r_[a_s, a_s[-1]]
    trel=t_b-t_b[m_b]
    tau=np.interp(trel,tb_s,tn_s); a_t=np.interp(trel,tb_s,a_s)
    tau_abs=tau+t_n[m_n]; mono=np.r_[True,np.diff(tau_abs)>0]
    use=(tau_abs>=t_n[0])&(tau_abs<=t_n[-1])&mono
    phi0=np.unwrap(np.angle(h_n))[m_n]-np.unwrap(np.angle(h_b))[m_b]
    def f(d):
        hs=a_t*np.exp(1j*(phi0+d))*h_b; hi=creative.interp_complex(tau_abs[use],hs[use],t_n)
        cov=(t_n>=tau_abs[use][0])&(t_n<=tau_abs[use][-1])
        return creative.mathcalE_error(h_n[cov],hi[cov]), (t_n[cov][0],t_n[cov][-1]), np.mean(cov)
    r=minimize_scalar(lambda d:f(d)[0],bounds=(-np.pi,np.pi),method="bounded")
    return f(r.x)

print("q     inspiral-only          +merger-anchor (fuller window)")
print("      mathcalE   NRwin        mathcalE   NRwin            cov")
for q in [5.0,3.0,2.5,2.0]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e0,w0,c0=recon(q,False); e1,w1,c1=recon(q,True)
    print(f"{q:4.1f}  {100*e0:6.3f}% [{w0[0]:7.1f},{w0[1]:6.1f}]   "
          f"{100*e1:6.3f}% [{w1[0]:7.1f},{w1[1]:6.1f}]  cov={c1:.3f}")
