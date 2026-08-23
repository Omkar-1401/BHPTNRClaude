"""Low-q validation panels for wf_nu_hybrid_global (even-nu coeffs): q=2, 2.25, 2.5, 2.75.
Waveform (inspiral + merger zoom) and alpha/beta params vs NR. Uses cached BHPT.
Writes hybrid_global_q{2,2p25,2p5,2p75}_{waveform,params}.png into Agentic_plots/wf_nu_hybrid_global/.
"""
import sys, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTutils"))
import fit_scaling_PN_opt_creative as creative
import fit_scaling_wf_nu_hybrid_q_dep as wfhyb
import BHPTNRHybridGlobal as B
PLOT = ROOT/"Agentic_plots"/"wf_nu_hybrid_global"

def cached_bhpt(q):
    d = np.load(wfhyb.waveform_cache_path(q) if hasattr(wfhyb, "waveform_cache_path")
                else ROOT/".cache"/"q_dep"/f"waveforms_q{q:.10f}.npz")
    return d["t_bhpt"], d["h_bhpt_re"] + 1j*d["h_bhpt_im"]
B._load_bhpt = cached_bhpt

def load_nr(q):
    d = np.load(ROOT/".cache"/"q_dep"/f"waveforms_q{q:.10f}.npz")
    return d["t_nr"], d["h_nr_re"] + 1j*d["h_nr_im"]

def aligned(t, h, t_nr, h_nr):
    """min over time-shift of the analytic-phi0 mismatch. Returns (err, dt, phi0_opt)
    where phi0_opt is the phase to rotate the model by so the DISPLAYED curve matches."""
    def err(dt):
        tt = t + dt; lo = max(tt[0], t_nr[0]); hi = min(tt[-1], t_nr[-1])
        m = (t_nr >= lo) & (t_nr <= hi)
        if m.sum() < 100: return 50.0
        g = creative.interp_complex(tt, h, t_nr[m]); r = h_nr[m]
        n1 = np.sum(np.abs(r)**2); Z = np.abs(np.sum(r*np.conj(g))); n2 = np.sum(np.abs(g)**2)
        return float((n1+n2-2*Z)/(2*n1))
    res = minimize_scalar(err, bounds=(-30, 30), method="bounded", options={"xatol":1e-3})
    dt = res.x
    # optimal phase: rotate model by angle of the complex overlap <r|g>
    tt = t + dt; lo = max(tt[0], t_nr[0]); hi = min(tt[-1], t_nr[-1])
    m = (t_nr >= lo) & (t_nr <= hi)
    g = creative.interp_complex(tt, h, t_nr[m])
    phi0 = float(np.angle(np.sum(h_nr[m] * np.conj(g))))
    return res.fun, dt, phi0

def tagof(q): return f"q{q:g}".replace(".", "p")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for q in [2.0, 2.25, 2.5, 2.75]:
        t, h = B.generate_hybrid_global_calibrated(q)
        t_nr, h_nr = load_nr(q)
        e, dt, phi0 = aligned(t, h, t_nr, h_nr)
        h_disp = h * np.exp(1j * phi0)                 # apply optimal phase for display
        tb, a, b = B.get_alpha_beta(q)
        # waveform
        figW, axW = plt.subplots(1, 2, figsize=(11, 4))
        for ax, (lo, hi) in zip(axW, [(-1000, -150), (-150, 80)]):
            ax.plot(t_nr, h_nr.real, "k-", lw=1.1, label="NR 22")
            ax.plot(t+dt, h_disp.real, "r--", lw=1.0, alpha=0.85, label="hybrid-global (even-nu)")
            ax.set_xlim(lo, hi); ax.grid(alpha=0.3); ax.set_xlabel("t/M")
        axW[0].legend(fontsize=8)
        figW.suptitle(f"q={q:g} [EXTRAP] even-nu fit:  mathcalE={e:.3e}  (nu={wfhyb.get_nu(q):.4f})")
        figW.tight_layout(); figW.savefig(PLOT/f"hybrid_global_{tagof(q)}_waveform.png", dpi=110); plt.close(figW)
        # params
        figP, axP = plt.subplots(1, 2, figsize=(11, 4))
        axP[0].plot(tb, a, "r-", lw=1.5); axP[0].set_ylabel("alpha(t)"); axP[0].set_title("alpha")
        axP[1].plot(tb, b, "r-", lw=1.5); axP[1].set_ylabel("beta(t)"); axP[1].set_title("beta")
        for ax in axP: ax.set_xlabel("t_NR / M"); ax.grid(alpha=0.3); ax.axvline(0, ls=":", color="gray", lw=1)
        figP.suptitle(f"q={q:g} [EXTRAP] even-nu params  (mathcalE={e:.3e})")
        figP.tight_layout(); figP.savefig(PLOT/f"hybrid_global_{tagof(q)}_params.png", dpi=110); plt.close(figP)
        print(f"q={q:g}: mathcalE={e:.3e}  -> hybrid_global_{tagof(q)}_{{waveform,params}}.png")
print(f"saved low-q even-nu panels to {PLOT}")
