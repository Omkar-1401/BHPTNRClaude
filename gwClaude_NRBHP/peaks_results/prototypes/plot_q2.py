"""Plot measured peak alpha, beta (joined, NOT fit) for q=2 with in-range overlays."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT.parent/"BHPTNRSurrogate"/"surrogates"))
import fit_scaling_peaks as peaks

QS=[2.0,3.0,5.0,8.0]
COL={2.0:"crimson",3.0:"#1f77b4",5.0:"#2ca02c",8.0:"#9467bd"}
LW ={2.0:2.4,3.0:1.3,5.0:1.3,8.0:1.3}

data={}
for q in QS:
    c=peaks.extract(q)
    o=np.argsort(c["x"])
    data[q]=dict(x=c["x"][o], t=c["t_bhpt"][np.argsort(c["t_bhpt"])],
                 a=c["alpha"][o], b=c["beta_abs"][o],
                 tt=c["t_bhpt"], aa=c["alpha"], bb=c["beta_abs"])

fig,ax=plt.subplots(2,2,figsize=(12,8))
for q in QS:
    d=data[q]; lab=f"q={q:g}"+(" (extrap target)" if q==2 else "")
    # vs x (inspiral coordinate)
    ax[0,0].plot(d["x"],d["a"],'-o',ms=2.5,color=COL[q],lw=LW[q],label=lab)
    ax[1,0].plot(d["x"],d["b"],'-o',ms=2.5,color=COL[q],lw=LW[q],label=lab)
    # vs merger-relative time (paper's "temporal variation")
    oo=np.argsort(d["tt"])
    ax[0,1].plot(d["tt"][oo],d["aa"][oo],'-o',ms=2.5,color=COL[q],lw=LW[q],label=lab)
    ax[1,1].plot(d["tt"][oo],d["bb"][oo],'-o',ms=2.5,color=COL[q],lw=LW[q],label=lab)

ax[0,0].set_xlabel("x = (M omega_orb)^(2/3)"); ax[0,0].set_ylabel("alpha  (|h|_NR/|h|_BHPT)")
ax[1,0].set_xlabel("x = (M omega_orb)^(2/3)"); ax[1,0].set_ylabel("beta_abs  (t_NR/t_BHPT)")
ax[0,1].set_xlabel("t - t_merger  [M]  (BHPT)"); ax[0,1].set_ylabel("alpha")
ax[1,1].set_xlabel("t - t_merger  [M]  (BHPT)"); ax[1,1].set_ylabel("beta_abs")
ax[0,1].set_xscale("symlog"); ax[1,1].set_xscale("symlog")
for a in ax.flat: a.grid(alpha=0.3); a.legend(fontsize=8)
ax[0,0].set_title("alpha vs frequency coordinate"); ax[1,0].set_title("beta vs frequency coordinate")
ax[0,1].set_title("alpha vs time (paper Fig.3 style)"); ax[1,1].set_title("beta vs time")
fig.suptitle("Measured peak values (joined, NOT fitted) — q=2 extrapolation target vs in-range",fontsize=12)
fig.tight_layout()
out=ROOT/"peaks_results"/"plots"; out.mkdir(parents=True,exist_ok=True)
p=out/"q2_alpha_beta_measured.png"; fig.savefig(p,dpi=130)
print(f"saved {p}")

# also print the q=2 endpoint values and inter-q spacing for the discussion
print("\nq   a@x=0.06  a@merger  b@x=0.06  b@merger")
for q in QS:
    d=data[q]
    print(f"{q:g}  {np.interp(0.06,d['x'],d['a']):.4f}   {d['a'][-1]:.4f}   "
          f"{np.interp(0.06,d['x'],d['b']):.4f}   {d['b'][-1]:.4f}")
