"""Validate the BHPTNRHybridGlobal generator at q=2,3,5,8 and make the model plots.
Uses cached BHPT (monkeypatch) to skip the ~1min surrogate load. Compares to cached NR
with optimized alignment (analytic phi0 + 1D time shift), and writes waveform + shape
plots into Agentic_plots/wf_nu_hybrid_global/.
"""
import sys, warnings, numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTutils"))
import fit_scaling_PN_opt_creative as creative
import fit_scaling_PN_opt_creative_q_dep as qdep
import BHPTNRHybridGlobal as B
PLOT = ROOT/"Agentic_plots"/"wf_nu_hybrid_global"; PLOT.mkdir(parents=True, exist_ok=True)

def cached_bhpt(q):
    d = np.load(qdep.waveform_cache_path(q))
    return d["t_bhpt"], d["h_bhpt_re"] + 1j*d["h_bhpt_im"]
B._load_bhpt = cached_bhpt

def load_nr(q):
    d = np.load(qdep.waveform_cache_path(q))
    return d["t_nr"], d["h_nr_re"] + 1j*d["h_nr_im"]

def aligned_mismatch(t, h, t_nr, h_nr):
    """min over time-shift (analytic phi0) mismatch of model vs NR on common support."""
    def err(dt):
        tt = t + dt
        lo = max(tt[0], t_nr[0]); hi = min(tt[-1], t_nr[-1])
        m = (t_nr >= lo) & (t_nr <= hi)
        if m.sum() < 100: return 50.0
        hi_m = creative.interp_complex(tt, h, t_nr[m]); href = h_nr[m]
        n1 = np.sum(np.abs(href)**2); n2 = np.sum(np.abs(hi_m)**2)
        Z = np.abs(np.sum(href*np.conj(hi_m)))
        return float((n1+n2-2*Z)/(2*n1))
    r = minimize_scalar(err, bounds=(-30, 30), method="bounded", options={"xatol":1e-3})
    return r.fun, r.x

qs = [8.0, 5.0, 3.0, 2.0]
colors = {8.0:"#1f77b4",5.0:"#2ca02c",3.0:"#ff7f0e",2.0:"#d62728"}
figW, axW = plt.subplots(4, 2, figsize=(14,12), gridspec_kw={"width_ratios":[3,2]})
figS, axS = plt.subplots(1, 2, figsize=(13,4.5))
print(f"{'q':>5} {'mathcalE (aligned)':>18}")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for r, q in enumerate(qs):
        t, h = B.generate_hybrid_global_calibrated(q)
        t_nr, h_nr = load_nr(q)
        e, dt = aligned_mismatch(t, h, t_nr, h_nr)
        print(f"{q:5.1f} {e:18.3e}")
        # waveform overlay (shift model by dt for display)
        for col,(lo,hi) in enumerate([(-1000,-150),(-150,80)]):
            ax = axW[r,col]
            ax.plot(t_nr, np.real(h_nr), "k-", lw=1.1, label="NR")
            ax.plot(t+dt, np.real(h), "r--", lw=1.0, alpha=0.85, label="hybrid_global")
            ax.set_xlim(lo,hi); ax.grid(True,alpha=0.3)
            if col==1: ax.set_xlabel("t/M")
        ex = "  [EXTRAP]" if q<3 else ""
        axW[r,0].set_ylabel(f"q={q}{ex}  mathcalE={e:.2e}")
        if r==0: axW[r,0].legend(fontsize=8, loc="upper left")
        # alpha/beta shapes
        tb, a, b = B.get_alpha_beta(q)
        ls = "--" if q==2.0 else "-"; lw = 2.2 if q==2.0 else 1.3
        axS[0].plot(tb, a, ls, color=colors[q], lw=lw, label=f"q={q}")
        axS[1].plot(tb, b, ls, color=colors[q], lw=lw, label=f"q={q}")
figW.suptitle("wf_nu_hybrid_global vs NRHybSur3dq8  —  q=8,5,3 in-range, q=2 extrapolation")
figW.tight_layout(); figW.savefig(PLOT/"hybrid_global_waveforms.png", dpi=95)
for a,tl,yl in [(axS[0],"alpha(t)","alpha"),(axS[1],"beta(t)","beta")]:
    a.set_xlim(-800,80); a.grid(True); a.set_xlabel("t_NR / M"); a.set_ylabel(yl); a.set_title(tl); a.legend(fontsize=8)
figS.suptitle("wf_nu_hybrid_global: alpha/beta shapes across q  (q=2 dashed)")
figS.tight_layout(); figS.savefig(PLOT/"hybrid_global_shapes.png", dpi=95)
print(f"saved plots -> {PLOT}")
