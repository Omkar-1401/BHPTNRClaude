"""Diagnose the reconstruction: is tau = beta_abs * t the right time map?
Test q=5 with MEASURED alpha,beta interpolated (no poly, no switch)."""
import sys, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTNRSurrogate"/"surrogates"))
import fit_scaling_PN_opt_creative as creative
WAVE=ROOT/".cache"/"q_dep"; PER_Q=ROOT/"peaks_results"/"per_q"

def load(q):
    d=np.load(WAVE/f"waveforms_q{q:.10f}.npz")
    return d["t_bhpt"],d["h_bhpt_re"]+1j*d["h_bhpt_im"],d["t_nr"],d["h_nr_re"]+1j*d["h_nr_im"]
def mism(a,b): return creative.mathcalE_error(a,b)

Q=5.0
c=np.load(PER_Q/f"peaks_q{Q:.10f}.npz")
t_b,h_b,t_n,h_n=load(Q)
psib=np.unwrap(np.angle(h_b)); om=np.abs(np.gradient(psib,t_b)); xb=(om/2)**(2/3)
m_b=int(np.argmax(np.abs(h_b))); m_n=int(np.argmax(np.abs(h_n)))
trel=t_b-t_b[m_b]

# measured peak clouds are in merger-relative BHPT time c["t_bhpt"] (<0), sorted by k.
# Build alpha(t), beta(t) by interpolating measured values vs merger-relative BHPT time.
tb_pk=c["t_bhpt"]; a_pk=c["alpha"]; b_pk=c["beta_abs"]
order=np.argsort(tb_pk)
tb_pk,a_pk,b_pk=tb_pk[order],a_pk[order],b_pk[order]
print(f"measured peaks: {len(tb_pk)} in trel [{tb_pk[0]:.0f},{tb_pk[-1]:.1f}]")

# interpolate measured alpha,beta onto all BHPT samples (extrapolate flat at ends)
a_of_t=np.interp(trel,tb_pk,a_pk)
b_of_t=np.interp(trel,tb_pk,b_pk)

def evaluate(tau, alpha, tag):
    tau_abs=tau+t_n[m_n]
    mono=np.r_[True,np.diff(tau)>0]
    use=(tau_abs>=t_n[0])&(tau_abs<=t_n[-1])&mono
    phi0=psib_align= np.unwrap(np.angle(h_n))[m_n]-psib[m_b]
    def f(dphi):
        hs=alpha*np.exp(1j*(phi0+dphi))*h_b
        hi=creative.interp_complex(tau_abs[use],hs[use],t_n)
        cov=(t_n>=tau_abs[use][0])&(t_n<=tau_abs[use][-1])
        return mism(h_n[cov],hi[cov]),cov
    r=minimize_scalar(lambda d:f(d)[0],bounds=(-np.pi,np.pi),method="bounded")
    e,cov=f(r.x)
    print(f"{tag}: mathcalE={e:.4e} cov={np.count_nonzero(cov)/len(t_n):.3f}")
    return e

# TEST 1: abs-ratio map tau = beta_abs * t
evaluate(b_of_t*trel, a_of_t, "abs-ratio map  tau=beta*t, measured a,b")
# TEST 2: amplitude only right, beta=1 (no time stretch) -> should be bad (control)
evaluate(1.0*trel, a_of_t, "control beta=1 (no stretch)")
# TEST 3: rate-integral map tau = cumtrapz(beta_slope) using measured beta_slope
bs=c["beta_slope"][order]
bs_of_t=np.interp(trel,tb_pk,bs)
# integrate rate from merger outward: tau(t)=int_{tm}^{t} beta dt' ; merger-relative
tau_int=creative.cumulative_trapezoid(bs_of_t, t_b)
tau_int=tau_int-np.interp(t_b[m_b],t_b,tau_int)   # zero at merger
evaluate(tau_int, a_of_t, "rate-integral map tau=int beta_slope dt")
# TEST 4: abs map but alpha=1 (isolate amplitude contribution)
evaluate(b_of_t*trel, np.ones_like(a_of_t), "abs map, alpha=1 (phase/time only)")
