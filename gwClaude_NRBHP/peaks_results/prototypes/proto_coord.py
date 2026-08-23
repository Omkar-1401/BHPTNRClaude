"""Does the coefficient-regression COORDINATE change beta's q=2 extrapolation?
Time-map recon fixed; sweep the variable in which the per-q Chebyshev coeffs are
regressed: nu, 1/q, delta=sqrt(1-4nu) (mass asymmetry).  All PP-anchored so the
correction ->0 at the point-particle limit (nu->0 / q->inf / delta->1).
"""
import sys, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTNRSurrogate"/"surrogates"))
import fit_scaling_PN_opt_creative as creative
WAVE = ROOT/".cache"/"q_dep"; PER_Q = ROOT/"peaks_results"/"per_q"
cases = {float(p.stem.replace("peaks_q","")): dict(np.load(p))
         for p in sorted(PER_Q.glob("peaks_q*.npz"))}
QS = sorted(cases)
x_all = np.concatenate([cases[q]["x"] for q in QS]); XLO, XHI = x_all.min(), x_all.max()
XMID, XHALF = 0.5*(XLO+XHI), 0.5*(XHI-XLO)
def u_of(x): return (np.clip(x,XLO,XHI)-XMID)/XHALF
def load(q):
    d=np.load(WAVE/f"waveforms_q{q:.10f}.npz")
    return (d["t_bhpt"],d["h_bhpt_re"]+1j*d["h_bhpt_im"],d["t_nr"],d["h_nr_re"]+1j*d["h_nr_im"])
def cheb_fit(u,y,deg): return np.linalg.lstsq(np.polynomial.chebyshev.chebvander(u,deg),y-1.0,rcond=None)[0]
def cheb_eval(c,u): return 1.0+np.polynomial.chebyshev.chebval(u,c)

# regression variable transforms (all -> 0 in PP limit so coeff = z*poly(z))
def zc(name, q):
    nu=q/(1+q)**2
    if name=="nu":   return nu
    if name=="invq": return 1.0/q
    if name=="delta":return 1.0-np.sqrt(max(1.0-4*nu,0.0))   # 1-delta_asym -> 0 at q->inf
qs_train=QS
DA,DB=5,4
A=np.array([cheb_fit(u_of(cases[q]["x"]),cases[q]["alpha"],DA) for q in QS])
B=np.array([cheb_fit(u_of(cases[q]["x"]),cases[q]["beta_abs"],DB) for q in QS])

def regress(C, name, dz):
    z=np.array([zc(name,q) for q in QS])
    Az=np.vstack([z**k for k in range(1,dz+1)]).T
    return [np.linalg.lstsq(Az,C[:,m],rcond=None)[0] for m in range(C.shape[1])]
def evalc(G, name, q, dz):
    z=zc(name,q); return np.array([sum(g[k]*z**(k+1) for k in range(len(g))) for g in G])

def peaks_of(t,h):
    amp=np.abs(h); t_m=float(t[int(np.argmax(amp))]); psi=np.unwrap(np.angle(h))
    sign=1.0 if psi[-1]>psi[0] else -1.0; psim=sign*psi
    psi_m=float(np.interp(t_m,t,psim)); n=int((psi_m-psim[0])/np.pi)
    tgt=psi_m-np.pi*np.arange(0,n+1); tgt=tgt[tgt>=psim[0]]
    t_pk=np.interp(tgt,psim,t); om=np.abs(np.gradient(psi,t))
    return t_m,t_pk,np.interp(t_pk,t,om)

def recon(q, GA, GB, name, dna, dnb):
    t_b,h_b,t_n,h_n=load(q)
    m_b=int(np.argmax(np.abs(h_b))); m_n=int(np.argmax(np.abs(h_n)))
    tm_b,tpk_b,om_pk=peaks_of(t_b,h_b); trel_pk=tpk_b-tm_b; x_pk=(om_pk/2)**(2/3)
    ca=evalc(GA,name,q,dna); cb=evalc(GB,name,q,dnb)
    a_pk=cheb_eval(ca,u_of(x_pk)); b_pk=cheb_eval(cb,u_of(x_pk))
    trel_nr_pk=b_pk*trel_pk
    o=np.argsort(trel_pk); tb_s=trel_pk[o]; tn_s=trel_nr_pk[o]; a_s=a_pk[o]
    keep=np.r_[True,np.diff(tn_s)>0]; tb_s=tb_s[keep]; tn_s=tn_s[keep]; a_s=a_s[keep]
    trel=t_b-t_b[m_b]; tau=np.interp(trel,tb_s,tn_s); a_t=np.interp(trel,tb_s,a_s)
    tau_abs=tau+t_n[m_n]; mono=np.r_[True,np.diff(tau_abs)>0]
    use=(tau_abs>=t_n[0])&(tau_abs<=t_n[-1])&mono
    phi0=np.unwrap(np.angle(h_n))[m_n]-np.unwrap(np.angle(h_b))[m_b]
    def f(d):
        hs=a_t*np.exp(1j*(phi0+d))*h_b; hi=creative.interp_complex(tau_abs[use],hs[use],t_n)
        cov=(t_n>=tau_abs[use][0])&(t_n<=tau_abs[use][-1])
        return creative.mathcalE_error(h_n[cov],hi[cov])
    r=minimize_scalar(f,bounds=(-np.pi,np.pi),method="bounded"); return f(r.x)

for name in ["nu","invq","delta"]:
    for dna,dnb in [(4,4),(4,5),(5,5)]:
        GA=regress(A,name,dna); GB=regress(B,name,dnb)
        ir=[]
        for q in [3.0,4.0,5.0,6.0,7.0,8.0]:
            with warnings.catch_warnings(): warnings.simplefilter("ignore"); ir.append(recon(q,GA,GB,name,dna,dnb))
        lo={}
        for q in [2.5,2.25,2.0,1.75]:
            with warnings.catch_warnings(): warnings.simplefilter("ignore"); lo[q]=recon(q,GA,GB,name,dna,dnb)
        gate="OK" if lo[2.0]<0.01 else "  "
        print(f"coord={name:5s} dna={dna} dnb={dnb}: in-range med={np.median(ir):.2e} max={np.max(ir):.2e} | "
              f"q2.5={100*lo[2.5]:.3f}% q2.25={100*lo[2.25]:.3f}% q2={100*lo[2.0]:.3f}% q1.75={100*lo[1.75]:.3f}% [{gate}]")
    print()
