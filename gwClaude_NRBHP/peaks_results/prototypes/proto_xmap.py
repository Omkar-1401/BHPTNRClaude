"""Is x-based reconstruction itself limited at q=2, or only the nu-extrapolation?
Compare at q=2 (all with MEASURED alpha,beta -- no nu-regression):
  (1) time-based interp on trel        [RESUME ceiling recipe, proto_diag2]
  (2) x-based interp of measured cloud [proto_sens meas/meas]
  (3) x-based SMOOTH poly fit to q2's own points (centered basis, deg 5/3)
If (3) ~ (1): x-based is fine, problem is purely nu-extrapolation of beta.
If (3) >> (1): x-based map loses q=2; must reconstruct on the time-map instead.
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
def fit_upoly(x, y, deg):
    u=(x-XMID)/XHALF; A=np.vstack([u**m for m in range(deg+1)]).T
    return np.linalg.lstsq(A, y-1.0, rcond=None)[0]
def eval_upoly(c, x):
    u=(np.clip(x,XLO,XHI)-XMID)/XHALF; return 1.0+sum(c[m]*u**m for m in range(len(c)))

def recon_general(q, a_of_trel_x, b_of_trel_x, mode):
    """mode='time': a,b are fns of trel; mode='x': fns of x."""
    t_b,h_b,t_n,h_n = load(q); psib=np.unwrap(np.angle(h_b))
    om=np.abs(np.gradient(psib,t_b)); xb=(om/2)**(2/3)
    m_b=int(np.argmax(np.abs(h_b))); m_n=int(np.argmax(np.abs(h_n))); trel=t_b-t_b[m_b]
    if mode=='time': a=a_of_trel_x(trel); b=b_of_trel_x(trel)
    else:            a=a_of_trel_x(xb);   b=b_of_trel_x(xb)
    tau=b*trel; tau_abs=tau+t_n[m_n]; mono=np.r_[True,np.diff(tau)>0]
    use=(tau_abs>=t_n[0])&(tau_abs<=t_n[-1])&mono
    phi0=np.unwrap(np.angle(h_n))[m_n]-psib[m_b]
    def f(d):
        hs=a*np.exp(1j*(phi0+d))*h_b
        hi=creative.interp_complex(tau_abs[use],hs[use],t_n)
        cov=(t_n>=tau_abs[use][0])&(t_n<=tau_abs[use][-1])
        return creative.mathcalE_error(h_n[cov],hi[cov])
    r=minimize_scalar(f,bounds=(-np.pi,np.pi),method="bounded"); return f(r.x)

print("q    (1)time-interp  (2)x-interp   (3)x-smoothpoly  (deg5/3 self-fit RMS a,b)")
for q in [2.0, 2.5, 3.0, 5.0]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = peaks.extract(q)
        ot=np.argsort(m["t_bhpt"]); tb=m["t_bhpt"][ot]; at=m["alpha"][ot]; bt=m["beta_abs"][ot]
        ox=np.argsort(m["x"]); xs=m["x"][ox]; ax=m["alpha"][ox]; bx=m["beta_abs"][ox]
        # (1) time
        e1=recon_general(q, lambda tr: np.interp(tr, tb, at),
                            lambda tr: np.interp(tr, tb, bt), 'time')
        # (2) x interp
        e2=recon_general(q, lambda xx: np.interp(np.clip(xx,XLO,XHI), xs, ax),
                            lambda xx: np.interp(np.clip(xx,XLO,XHI), xs, bx), 'x')
        # (3) x smooth poly (fit to this q's own points)
        ca=fit_upoly(m["x"], m["alpha"], 5); cb=fit_upoly(m["x"], m["beta_abs"], 3)
        rms_a=np.sqrt(np.mean((m["alpha"]-eval_upoly(ca,m["x"]))**2))
        rms_b=np.sqrt(np.mean((m["beta_abs"]-eval_upoly(cb,m["x"]))**2))
        e3=recon_general(q, lambda xx: eval_upoly(ca,xx),
                            lambda xx: eval_upoly(cb,xx), 'x')
    print(f"{q:4.1f}  {100*e1:8.4f}%    {100*e2:8.4f}%    {100*e3:8.4f}%     "
          f"({rms_a:.2e},{rms_b:.2e})")
