"""Per-q PDF plot set for pn_anchored, matching the wf_nu_hybrid_global style:
  pn_anchored_q{N}[_extrap]_{waveform,zoomed,params,loss_coords}.pdf
into Agentic_plots/pn_anchored/.  q = 8, 5, 3, 2 (q=2 is extrapolation).
Uses cached BHPT (monkeypatch) + cached NR; mathcalE (t0/phi0-aligned) in each title.
"""
import sys, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTutils"))
import bhpt_utils
matplotlib.rcdefaults()                       # match wf_nu_hybrid_global plot styling exactly
matplotlib.rcParams["text.usetex"] = False    # (latex not installed in this env)
import fit_scaling_PN_opt_creative as creative
import fit_scaling_PN_opt_creative_q_dep as qdep
import fit_scaling_wf_nu_q_dep as wfnu
import BHPTNRPNAnchored as B
PLOT = ROOT/"Agentic_plots"/"pn_anchored"; PLOT.mkdir(parents=True, exist_ok=True)
Q_INPUTS = [8.0, 5.0, 3.0, 2.0]

def cached_bhpt(q):
    d = np.load(qdep.waveform_cache_path(q)); return d["t_bhpt"], d["h_bhpt_re"]+1j*d["h_bhpt_im"]
B._load_bhpt = cached_bhpt
def load_nr(q):
    d = np.load(qdep.waveform_cache_path(q)); return d["t_nr"], d["h_nr_re"]+1j*d["h_nr_im"]

def tag(q):
    s = f"q{int(q)}" if q == int(q) else f"q{q}"
    return s + ("_extrap" if q < 3.0 else "")
def title(q, err=None):
    nu = q/(1+q)**2; ex = " [extrapolation]" if q < 3.0 else ""
    es = f";  mathcalE = {err:.3g}" if err is not None else ""
    return f"pn-anchored  q={q}{ex};  nu={nu:.4f}{es}"

def aligned(t, h, t_nr, h_nr):
    def err(dt):
        tt=t+dt; lo=max(tt[0],t_nr[0]); hi=min(tt[-1],t_nr[-1]); m=(t_nr>=lo)&(t_nr<=hi)
        if m.sum()<100: return 50.0
        g=creative.interp_complex(tt,h,t_nr[m]); r=h_nr[m]
        n1=np.sum(np.abs(r)**2); Z=np.abs(np.sum(r*np.conj(g))); n2=np.sum(np.abs(g)**2)
        return float((n1+n2-2*Z)/(2*n1))
    grid=np.linspace(-250,250,501); c=grid[int(np.argmin([err(x) for x in grid]))]
    r=minimize_scalar(err,bounds=(c-4,c+4),method="bounded",options={"xatol":1e-3}); dt=r.x
    tt=t+dt; lo=max(tt[0],t_nr[0]); hi=min(tt[-1],t_nr[-1]); m=(t_nr>=lo)&(t_nr<=hi)
    g=creative.interp_complex(tt,h,t_nr[m]); phi=float(np.angle(np.sum(h_nr[m]*np.conj(g))))
    return r.fun, dt, phi

def waveform_plot(q, t_nr, h_nr, t_m, h_m, err):
    fig,(a1,a2)=plt.subplots(1,2,figsize=(10,4),gridspec_kw={"width_ratios":[3,2]})
    for ax,xl in [(a1,(-1000,-200)),(a2,(-200,100))]:
        ax.plot(t_nr,np.real(h_nr),label="NR 22 mode")
        ax.plot(t_m,np.real(h_m),label="pn-anchored",alpha=0.8)
        ax.set_xlim(*xl); ax.grid(True); ax.set_xlabel("t/M")
    a1.legend(fontsize=8); fig.suptitle(title(q,err)); fig.tight_layout()
    fig.savefig(PLOT/f"pn_anchored_{tag(q)}_waveform.pdf"); plt.close(fig)

def zoomed_plot(q, t_nr, h_nr, t_m, h_m, err):
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot(t_nr,np.real(h_nr),label="NR 22 mode"); ax.plot(t_m,np.real(h_m),label="pn-anchored",alpha=0.8)
    ax.set_xlim(-100,100); ax.grid(True); ax.set_xlabel("t/M"); ax.legend(fontsize=8)
    ax.set_title(title(q,err)); fig.tight_layout()
    fig.savefig(PLOT/f"pn_anchored_{tag(q)}_zoomed.pdf"); plt.close(fig)

def params_plot(q, tau, alpha, beta, x, err):
    ab = bhpt_utils.all_alpha_beta_1D_scaling_params(q,[creative.MODE])
    base=(q/(1+q))**1.2; lbl=f"pn-anchored  nu={q/(1+q)**2:.4f}"
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    ax[0].plot(tau,alpha,label=lbl); ax[0].axhline(ab["alpha_l2m2"],color="k",ls="--",label="BHPTNRSurrogate alpha")
    ax[0].axhline(base,color="r",ls=":",label="X1^(6/5) anchor")
    ax[0].set_xlim(-1000,100); ax[0].grid(True); ax[0].set_xlabel("t_NR / M"); ax[0].set_ylabel("alpha(t)"); ax[0].legend(fontsize=7)
    ax[1].plot(tau,beta,label=lbl); ax[1].axhline(ab["beta"],color="k",ls="--",label="BHPTNRSurrogate beta")
    ax[1].axhline(base,color="r",ls=":",label="X1^(6/5) anchor")
    ax[1].set_xlim(-1000,100); ax[1].grid(True); ax[1].set_xlabel("t_NR / M"); ax[1].set_ylabel("beta(t)"); ax[1].legend(fontsize=7)
    ax[2].plot(x,alpha,lw=1); ax[2].set_xlabel("x = (M omega_orb)^(2/3)"); ax[2].set_ylabel("alpha"); ax[2].set_title("alpha vs x"); ax[2].grid(True)
    fig.suptitle(title(q,err)); fig.tight_layout()
    fig.savefig(PLOT/f"pn_anchored_{tag(q)}_params.pdf"); plt.close(fig)

def loss_plot(q, t_bhpt, los):
    fig,ax=plt.subplots(1,3,figsize=(14,4)); m=(t_bhpt>=-1500)&(t_bhpt<=150)
    for key,a,lb in [("e_hat",ax[0],"Ehat"),("j_hat",ax[1],"Jhat"),("p_loss",ax[2],"p_loss")]:
        a.plot(t_bhpt[m],los[key][m],lw=1); a.axvline(0,color="gray",ls="--",lw=0.8)
        a.axhline(0,color="k",ls=":",lw=0.6,alpha=0.5); a.set_xlabel("t / M"); a.set_ylabel(lb)
        a.set_title(f"{lb} (wf flux)"); a.grid(True)
    fig.suptitle(f"pn-anchored loss coordinates  q={q}  nu={q/(1+q)**2:.4f}")
    fig.tight_layout(); fig.savefig(PLOT/f"pn_anchored_{tag(q)}_loss_coords.pdf"); plt.close(fig)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for q in Q_INPUTS:
        t_m, h_m = B.generate_pn_anchored_calibrated(q)
        t_nr, h_nr = load_nr(q)
        err, dt, phi = aligned(t_m, h_m, t_nr, h_nr)
        t_disp = t_m + dt; h_disp = h_m*np.exp(1j*phi)
        # alpha,beta,tau,x all on one consistent BHPT grid (same 'use' mask)
        tb, hb = cached_bhpt(q)
        los = wfnu.wf_loss_coordinates(tb, hb, {"nu":q/(1+q)**2,"t_bhpt_merger":float(tb[np.argmax(np.abs(hb))]),"t_nr_merger":0.0})
        x = B._get_x(tb, hb)
        alpha, beta = B._alpha_beta(q, tb, hb, los["p_loss"])
        bc = creative.cumulative_trapezoid(beta, tb); tau = bc - bc[int(np.argmax(np.abs(hb)))]
        use = np.r_[True, np.diff(tau) > 0] & (tau >= creative.NR_T_START) & (tau <= creative.NR_T_END)
        waveform_plot(q, t_nr, h_nr, t_disp, h_disp, err)
        zoomed_plot(q, t_nr, h_nr, t_disp, h_disp, err)
        params_plot(q, tau[use], alpha[use], beta[use], np.minimum(x, 0.26)[use], err)
        loss_plot(q, tb, los)
        print(f"q={q}: mathcalE={err:.3e}  -> pn_anchored_{tag(q)}_*.pdf", flush=True)
print(f"saved per-q PDFs to {PLOT}")
