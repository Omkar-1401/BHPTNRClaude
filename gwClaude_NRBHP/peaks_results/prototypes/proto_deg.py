"""Can a higher-degree x-poly for beta recover the q=2 ceiling, or is the x-map
the wall? Sweep beta x-degree; also test a q-universal INSPIRAL time coordinate
s = trel/|trel_start| (normalized time) as an alternative regression coordinate.
All per-q self-fits (no nu-regression) -> pure representation-ceiling test.
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
cases = {float(p.stem.replace("peaks_q","")): dict(np.load(p))
         for p in sorted(PER_Q.glob("peaks_q*.npz"))}
QS = sorted(cases)
x_all = np.concatenate([cases[q]["x"] for q in QS]); XLO, XHI = x_all.min(), x_all.max()
XMID, XHALF = 0.5*(XLO+XHI), 0.5*(XHI-XLO)
def load(q):
    d = np.load(WAVE/f"waveforms_q{q:.10f}.npz")
    return (d["t_bhpt"], d["h_bhpt_re"]+1j*d["h_bhpt_im"], d["t_nr"], d["h_nr_re"]+1j*d["h_nr_im"])

def cheb_fit(u, y, deg):    # u in [-1,1]; fit y-1 in Chebyshev basis
    A = np.polynomial.chebyshev.chebvander(u, deg)
    return np.linalg.lstsq(A, y-1.0, rcond=None)[0]
def cheb_eval(c, u):
    return 1.0 + np.polynomial.chebyshev.chebval(u, c)

def recon_x(q, ca, cb, deg_a, deg_b):
    t_b,h_b,t_n,h_n = load(q); psib=np.unwrap(np.angle(h_b))
    om=np.abs(np.gradient(psib,t_b)); xb=(om/2)**(2/3)
    m_b=int(np.argmax(np.abs(h_b))); m_n=int(np.argmax(np.abs(h_n))); trel=t_b-t_b[m_b]
    u=(np.clip(xb,XLO,XHI)-XMID)/XHALF
    a=cheb_eval(ca,u); b=cheb_eval(cb,u)
    tau=b*trel; tau_abs=tau+t_n[m_n]; mono=np.r_[True,np.diff(tau)>0]
    use=(tau_abs>=t_n[0])&(tau_abs<=t_n[-1])&mono
    phi0=np.unwrap(np.angle(h_n))[m_n]-psib[m_b]
    def f(d):
        hs=a*np.exp(1j*(phi0+d))*h_b; hi=creative.interp_complex(tau_abs[use],hs[use],t_n)
        cov=(t_n>=tau_abs[use][0])&(t_n<=tau_abs[use][-1])
        return creative.mathcalE_error(h_n[cov],hi[cov])
    r=minimize_scalar(f,bounds=(-np.pi,np.pi),method="bounded"); return f(r.x)

print("=== per-q SELF-FIT x-map ceiling: sweep beta x-degree (alpha deg5 fixed) ===")
print(f"{'q':>4} " + " ".join(f"db={d:<2d}" for d in [3,4,6,8,10]))
for q in [2.0, 2.5, 3.0, 5.0, 8.0]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m=peaks.extract(q); u=(np.clip(m["x"],XLO,XHI)-XMID)/XHALF
        ca=cheb_fit(u, m["alpha"], 5)
        row=[]
        for db in [3,4,6,8,10]:
            cb=cheb_fit(u, m["beta_abs"], db)
            e=recon_x(q, ca, cb, 5, db); row.append(e)
    print(f"{q:>4.1f} " + " ".join(f"{100*e:6.3f}%" for e in row))
