"""Does extending the TRAINING range down to q=2.25 (cached low-q, NOT using q=2
itself) turn the q=2 result into a near-interpolation and clear the <1% gate?
Compares training sets: [3,8] (fair vs remnant_partial) vs [2.25,8] vs [2.5,8].
Time-map recon, nu coord, Chebyshev x, dnb=4.
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

# extract low-q anchors from the cache (not stored in per_q)
def get_case(q):
    if q in base: return base[q]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); return peaks.extract(q)
LOWQ = [2.25, 2.5, 2.75]
for q in LOWQ: base.setdefault(q, get_case(q))

allx = np.concatenate([base[q]["x"] for q in sorted(base)]); XLO,XHI=allx.min(),allx.max()
XMID,XHALF=0.5*(XLO+XHI),0.5*(XHI-XLO)
def u_of(x): return (np.clip(x,XLO,XHI)-XMID)/XHALF
def load(q):
    d=np.load(WAVE/f"waveforms_q{q:.10f}.npz")
    return (d["t_bhpt"],d["h_bhpt_re"]+1j*d["h_bhpt_im"],d["t_nr"],d["h_nr_re"]+1j*d["h_nr_im"])
def cheb_fit(u,y,deg): return np.linalg.lstsq(np.polynomial.chebyshev.chebvander(u,deg),y-1.0,rcond=None)[0]
def cheb_eval(c,u): return 1.0+np.polynomial.chebyshev.chebval(u,c)
def peaks_of(t,h):
    amp=np.abs(h); t_m=float(t[int(np.argmax(amp))]); psi=np.unwrap(np.angle(h))
    sign=1.0 if psi[-1]>psi[0] else -1.0; psim=sign*psi
    psi_m=float(np.interp(t_m,t,psim)); n=int((psi_m-psim[0])/np.pi)
    tgt=psi_m-np.pi*np.arange(0,n+1); tgt=tgt[tgt>=psim[0]]
    t_pk=np.interp(tgt,psim,t); om=np.abs(np.gradient(psi,t)); return t_m,t_pk,np.interp(t_pk,t,om)

DA,DB,DNA,DNB=5,4,4,4
def build(qlo):
    QT=[q for q in sorted(base) if q>=qlo-1e-9]
    nus=np.array([q/(1+q)**2 for q in QT])
    A=np.array([cheb_fit(u_of(base[q]["x"]),base[q]["alpha"],DA) for q in QT])
    B=np.array([cheb_fit(u_of(base[q]["x"]),base[q]["beta_abs"],DB) for q in QT])
    def reg(C,dz):
        Az=np.vstack([nus**k for k in range(1,dz+1)]).T
        return [np.linalg.lstsq(Az,C[:,m],rcond=None)[0] for m in range(C.shape[1])]
    return reg(A,DNA),reg(B,DNB),len(QT)
def evalc(G,nu): return np.array([sum(g[k]*nu**(k+1) for k in range(len(g))) for g in G])

def recon(q,GA,GB):
    t_b,h_b,t_n,h_n=load(q); nu=q/(1+q)**2
    m_b=int(np.argmax(np.abs(h_b))); m_n=int(np.argmax(np.abs(h_n)))
    tm_b,tpk_b,om_pk=peaks_of(t_b,h_b); trel_pk=tpk_b-tm_b; x_pk=(om_pk/2)**(2/3)
    a_pk=cheb_eval(evalc(GA,nu),u_of(x_pk)); b_pk=cheb_eval(evalc(GB,nu),u_of(x_pk))
    trel_nr_pk=b_pk*trel_pk; o=np.argsort(trel_pk)
    tb_s=trel_pk[o]; tn_s=trel_nr_pk[o]; a_s=a_pk[o]
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

for qlo,tag in [(3.0,"[3,8] fair"),(2.75,"[2.75,8]"),(2.5,"[2.5,8]"),(2.25,"[2.25,8]")]:
    GA,GB,nq=build(qlo)
    ir=[]
    for q in [3.0,4.0,5.0,6.0,7.0,8.0]:
        with warnings.catch_warnings(): warnings.simplefilter("ignore"); ir.append(recon(q,GA,GB))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e2=recon(2.0,GA,GB); e175=recon(1.75,GA,GB)
    gate="GATE OK" if e2<0.01 else "fail"
    print(f"train {tag:12s} (n={nq:2d}): in-range med={np.median(ir):.2e} | "
          f"q2.0={100*e2:.3f}% q1.75={100*e175:.3f}%  [{gate}]  (vs remnant_partial q2=0.37%)")
