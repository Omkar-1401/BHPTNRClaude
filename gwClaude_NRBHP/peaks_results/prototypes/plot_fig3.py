"""Reproduce Fig-3-style alpha,beta vs time for q=3, extended THROUGH ringdown,
with the naive mass-scale reference 1/(1+1/q)."""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT.parent/"BHPTNRSurrogate"/"surrogates"))
WAVE=ROOT/".cache"/"q_dep"
def load(q):
    d=np.load(WAVE/f"waveforms_q{q:.10f}.npz")
    return d["t_bhpt"],d["h_bhpt_re"]+1j*d["h_bhpt_im"],d["t_nr"],d["h_nr_re"]+1j*d["h_nr_im"]

def peaks_bidir(t,h,dphi=np.pi):
    """extrema spaced by dphi across the FULL waveform (inspiral + ringdown)."""
    amp=np.abs(h); m=int(np.argmax(amp)); t_m=t[m]
    psi=np.unwrap(np.angle(h)); sign=1.0 if psi[-1]>psi[0] else -1.0; psi=sign*psi
    psi_m=float(np.interp(t_m,t,psi))
    kmax_back=int((psi_m-psi[0])/dphi); kmax_fwd=int((psi[-1]-psi_m)/dphi)
    ks=np.arange(-kmax_back,kmax_fwd+1)            # negative=inspiral, positive=ringdown
    targets=psi_m+dphi*ks
    good=(targets>=psi[0])&(targets<=psi[-1]); ks=ks[good]; targets=targets[good]
    t_pk=np.interp(targets,psi,t); amp_pk=np.interp(t_pk,t,amp)
    return t_m,ks,t_pk,amp_pk

Q=3.0
t_b,h_b,t_n,h_n=load(Q)
tmb,kb,tpb,ab=peaks_bidir(t_b,h_b)
tmn,kn,tpn,an=peaks_bidir(t_n,h_n)
# match by k (signed, from merger)
kk=np.intersect1d(kb,kn)
ib={k:i for i,k in enumerate(kb)}; iN={k:i for i,k in enumerate(kn)}
tb_rel=np.array([tpb[ib[k]]-tmb for k in kk])
tn_rel=np.array([tpn[iN[k]]-tmn for k in kk])
alpha=np.array([an[iN[k]]/ab[ib[k]] for k in kk])
with np.errstate(invalid="ignore",divide="ignore"):
    beta=np.where(tb_rel!=0, tn_rel/tb_rel, np.nan)
mass_scale=1.0/(1.0+1.0/Q)

fig,ax=plt.subplots(1,2,figsize=(13,5))
ax[0].plot(tb_rel,alpha,'-o',ms=3,color="crimson",label=r"$\alpha_{\rm peak}=|h|_{NR}/|h|_{BHPT}$")
ax[0].axhline(mass_scale,ls="--",color="k",label=fr"mass scale $1/(1+1/q)={mass_scale:.3f}$")
ax[0].axvline(0,ls=":",color="gray",lw=1,label="merger")
ax[0].set_ylabel(r"$\alpha$"); ax[0].set_title(f"alpha vs time, q={Q:g} (through ringdown)")
ax[1].plot(tb_rel,beta,'-o',ms=3,color="#1f77b4",label=r"$\beta_{\rm peak}=t_{NR}/t_{BHPT}$")
ax[1].axvline(0,ls=":",color="gray",lw=1,label="merger")
ax[1].set_ylabel(r"$\beta$"); ax[1].set_title(f"beta vs time, q={Q:g}")
for a in ax:
    a.set_xlabel(r"$t-t_{\rm merger}$ [M] (BHPT)"); a.set_xscale("symlog"); a.grid(alpha=0.3); a.legend(fontsize=9)
fig.tight_layout()
out=ROOT/"peaks_results"/"plots"/"q3_fig3_style_thru_ringdown.png"; fig.savefig(out,dpi=130)
print(f"saved {out}")

# print inspiral / merger / ringdown alpha values
def near(tr,arr,tt): return arr[np.argmin(np.abs(tr-tt))]
print(f"mass-scale 1/(1+1/q) = {mass_scale:.4f}")
print(f"alpha: early(-3000)={near(tb_rel,alpha,-3000):.4f}  "
      f"late-insp(-100)={near(tb_rel,alpha,-100):.4f}  "
      f"merger(0)={near(tb_rel,alpha,0):.4f}  "
      f"ringdown(+30)={near(tb_rel,alpha,30):.4f}")
rd=kk>0
print(f"n ringdown peaks matched: {np.count_nonzero(rd)}; "
      f"alpha ringdown range [{np.nanmin(alpha[rd]):.3f},{np.nanmax(alpha[rd]):.3f}]" if rd.any() else "no ringdown peaks")
