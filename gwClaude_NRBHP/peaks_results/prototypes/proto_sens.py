"""Sensitivity decomposition at q=2: which of alpha/beta throws away the ceiling?
Reconstruct with (measured a, measured b) [ceiling], then swap in regressed a only,
regressed b only, both.  Also test a well-conditioned (centered/scaled) x-basis.
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

# --- centered/scaled (well-conditioned) x-poly ---
def u_of(x): return (np.clip(x, XLO, XHI) - XMID)/XHALF
def fit_upoly(x, y, deg):
    u = (x - XMID)/XHALF
    A = np.vstack([u**m for m in range(deg+1)]).T
    c,*_ = np.linalg.lstsq(A, y-1.0, rcond=None); return c
def eval_upoly(c, x):
    u = u_of(x); return 1.0 + sum(c[m]*u**m for m in range(len(c)))
def regress(C, nus, dnu):
    Anu = np.vstack([nus**k for k in range(1, dnu+1)]).T
    return [np.linalg.lstsq(Anu, C[:,m], rcond=None)[0] for m in range(C.shape[1])]
def evalc(G, nu): return np.array([sum(g[k]*nu**(k+1) for k in range(len(g))) for g in G])

DA, DB, DNA, DNB = 5, 3, 4, 4
nus = np.array([cases[q]["nu"] for q in QS])
A = np.array([fit_upoly(cases[q]["x"], cases[q]["alpha"], DA) for q in QS])
B = np.array([fit_upoly(cases[q]["x"], cases[q]["beta_abs"], DB) for q in QS])
print("=== conditioning: max|coeff| monomial vs centered basis (alpha deg5) ===")
Amono = np.array([np.polyfit(cases[q]["x"], cases[q]["alpha"]-1, DA) for q in QS])
print(f"  monomial max|coeff| = {np.abs(Amono).max():.3e}")
print(f"  centered max|coeff| = {np.abs(A).max():.3e}")
GA = regress(A, nus, DNA); GB = regress(B, nus, DNB)

def recon(q, a_fn, b_fn):
    t_b,h_b,t_n,h_n = load(q)
    psib = np.unwrap(np.angle(h_b)); om = np.abs(np.gradient(psib, t_b)); xb=(om/2)**(2/3)
    m_b=int(np.argmax(np.abs(h_b))); m_n=int(np.argmax(np.abs(h_n))); trel=t_b-t_b[m_b]
    a = a_fn(xb); b = b_fn(xb)
    tau = b*trel; tau_abs = tau+t_n[m_n]; mono=np.r_[True, np.diff(tau)>0]
    use=(tau_abs>=t_n[0])&(tau_abs<=t_n[-1])&mono
    phi0=np.unwrap(np.angle(h_n))[m_n]-psib[m_b]
    def f(d):
        hs=a*np.exp(1j*(phi0+d))*h_b
        hi=creative.interp_complex(tau_abs[use],hs[use],t_n)
        cov=(t_n>=tau_abs[use][0])&(t_n<=tau_abs[use][-1])
        return creative.mathcalE_error(h_n[cov],hi[cov])
    r=minimize_scalar(f,bounds=(-np.pi,np.pi),method="bounded"); return f(r.x)

# measured interpolants for a given q (the ceiling inputs)
def meas_fns(q):
    m = peaks.extract(q); o=np.argsort(m["t_bhpt"]); tb=m["t_bhpt"][o]
    ap=m["alpha"][o]; bp=m["beta_abs"][o]
    # measured are functions of trel via the peak time-map; interp on trel
    def a_fn(xb, _tb=tb, _ap=ap): return None
    return m

print("\n=== q=2 sensitivity (mathcalE) ===")
for q2 in [2.0, 2.5, 3.0]:
    nu2 = q2/(1+q2)**2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = peaks.extract(q2)
        o = np.argsort(m["t_bhpt"]); tb=m["t_bhpt"][o]; ap=m["alpha"][o]; bp=m["beta_abs"][o]
        # measured as fn of trel (need trel->interp). Build via bhpt phase x-map:
        t_b,h_b,t_n,h_n = load(q2); psib=np.unwrap(np.angle(h_b))
        m_b=int(np.argmax(np.abs(h_b))); trel_full=t_b-t_b[m_b]
        a_meas_fn = lambda xb: np.interp(np.clip(xb, XLO, XHI), np.sort(m["x"]),
                                          m["alpha"][np.argsort(m["x"])])
        b_meas_fn = lambda xb: np.interp(np.clip(xb, XLO, XHI), np.sort(m["x"]),
                                          m["beta_abs"][np.argsort(m["x"])])
        ca = evalc(GA, nu2); cb = evalc(GB, nu2)
        a_reg_fn = lambda xb: eval_upoly(ca, xb)
        b_reg_fn = lambda xb: eval_upoly(cb, xb)
        e_mm = recon(q2, a_meas_fn, b_meas_fn)   # ceiling (interp in x)
        e_rm = recon(q2, a_reg_fn,  b_meas_fn)   # regressed alpha only
        e_mr = recon(q2, a_meas_fn, b_reg_fn)    # regressed beta only
        e_rr = recon(q2, a_reg_fn,  b_reg_fn)    # both regressed
    print(f"q={q2}: meas/meas={100*e_mm:6.3f}%  reg-a/meas-b={100*e_rm:6.3f}%  "
          f"meas-a/reg-b={100*e_mr:6.3f}%  reg/reg={100*e_rr:6.3f}%")
