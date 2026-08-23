"""Two SEPARATE 4-panel PDFs of each model's free parameters vs q (q in [2,8], q=3 = train edge):
  param_hybrid_global.pdf  — wf_nu_hybrid_global's 9 params (each a fitted cubic in nu)
  param_pn_anchored.pdf    — pn_anchored's parameter functions (analytic q-dep + fitted constants)
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
matplotlib.rcdefaults(); matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTutils"))
import fit_scaling_wf_nu_hybrid_q_dep as H
import surfinBH
_SFBH = surfinBH.LoadFits("NRSur3dq8Remnant"); QF1, QF2, QF3 = 1.5251, -1.1568, 0.1292
OUT = ROOT/"Agentic_plots"/"pn_anchored"
qg = np.linspace(2.0, 8.0, 200)

def nu(q): return q/(1+q)**2
def evp(q, c): return sum(c[k]*nu(q)**k for k in range(len(c)))
def beta_r_phys(q):
    mf,_=_SFBH.mf(q,[0,0,0],[0,0,0]); cf,_=_SFBH.chif(q,[0,0,0],[0,0,0]); cf=float(cf[2])
    return (0.3683*(1+q)/q)/((QF1+QF2*(1-cf)**QF3)/float(mf))
def vfmt(a, title):
    a.axvline(3.0, ls=":", color="gray", lw=1.2, label="q=3 (train edge)")
    a.grid(alpha=0.3); a.set_xlabel("q"); a.set_title(title); a.legend(fontsize=8)

# ---------------- hybrid_global ----------------
hyb = {n: np.array(v) for n, v in json.load(open(ROOT/"wf_nu_hybrid_global_results"/"coeffs.json")).items()
       if n in H.PARAM_NAMES}
def hy(n): return np.array([evp(q, hyb[n]) for q in qg])
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
for n in ("alpha_i", "alpha_L"): ax[0,0].plot(qg, hy(n), label=n)
vfmt(ax[0,0], "alpha inspiral (alpha_i, alpha_L)")
for n in ("alpha_E", "alpha_J"): ax[0,1].plot(qg, hy(n), label=n)
vfmt(ax[0,1], "alpha merger flux couplings (alpha_E, alpha_J)")
for n in ("beta_i", "beta_L", "beta_r"): ax[1,0].plot(qg, hy(n), label=n)
vfmt(ax[1,0], "beta (beta_i, beta_L, beta_r)")
for n in ("p0", "w"): ax[1,1].plot(qg, hy(n), label=n)
vfmt(ax[1,1], "switch (p0, w)")
fig.suptitle("wf_nu_hybrid_global — free parameters vs q  (each a fitted cubic in nu; q<3 extrapolated)",
             fontsize=12)
fig.tight_layout(); fig.savefig(OUT/"param_hybrid_global.pdf"); plt.close(fig)
print("saved param_hybrid_global.pdf")

# ---------------- pn_anchored ----------------
pnc = json.load(open(ROOT/"pn_anchored_results"/"coeffs.json")); th = dict(zip(pnc["param_names"], pnc["theta"]))
base = (qg/(1+qg))**1.2; XREF = 0.15                         # reference late-inspiral frequency
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    br = np.array([beta_r_phys(q)*th["r0"] for q in qg])
a_insp = base*(1 + (55/42)*nu(qg)*XREF + nu(qg)*th["a2"]*XREF**2)
b_insp = base*(1 + nu(qg)*(th["b1"]*XREF + th["b2"]*XREF**2))
amr = base*(th["m0"] + th["m1"]*nu(qg))
p0 = th["p0_0"] + th["p0_1"]*nu(qg)

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0,0].plot(qg, base, label="base = X1^(6/5)  (x->0 anchor)")
ax[0,0].plot(qg, a_insp, "--", label=f"alpha_insp at x={XREF} (slope 55/42 fixed, a2={th['a2']:.2f})")
vfmt(ax[0,0], "alpha inspiral (PN-anchored)")
ax[0,1].plot(qg, amr, "C2", label=f"alpha_mr = base*(m0+m1*nu),  m0={th['m0']:.3f}, m1={th['m1']:.3f}")
vfmt(ax[0,1], "alpha merger/ringdown branch")
ax[1,0].plot(qg, base, label="base = X1^(6/5)  (x->0 anchor)")
ax[1,0].plot(qg, b_insp, "--", label=f"beta_insp at x={XREF}  (b1={th['b1']:.2f}, b2={th['b2']:.2f})")
ax[1,0].plot(qg, br, "C1", label=f"beta_r = QNM*r0,  r0={th['r0']:.3f}")
vfmt(ax[1,0], "beta (inspiral anchor + remnant beta_r)")
ax[1,1].plot(qg, p0, label=f"p0 = p0_0 + p0_1*nu  ({th['p0_0']:.2f}, {th['p0_1']:.2f})")
ax[1,1].axhline(th["w0"], color="C1", ls="--", label=f"w0 = {th['w0']:.3f} (const)")
vfmt(ax[1,1], "switch (p0, w)")
fig.suptitle("pn_anchored — parameter functions vs q  (q-dep from X1^(6/5) + QNM; residuals are "
             "q-independent constants)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT/"param_pn_anchored.pdf"); plt.close(fig)
print("saved param_pn_anchored.pdf")
