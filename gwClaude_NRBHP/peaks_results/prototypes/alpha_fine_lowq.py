"""alpha(t) for the even-nu wf_nu_hybrid_global model, finely spaced q in [2,3].
Left panel: q = 2.0, 2.15, 2.25, 2.35, 2.45.  Right panel: q = 2.55..2.95 (+3.0).
Uses cached BHPT where available, else the surrogate (~1 min/q)."""
import sys, warnings
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/home/omkarnm1401/UT_Austin/gwClaude_NRBHP")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent/"BHPTutils"))
sys.path.insert(0, str(ROOT.parent/"BHPTNRSurrogate"/"surrogates"))
import BHPTNRHybridGlobal as B
CACHE = ROOT/".cache"/"q_dep"

def load_bhpt(q):
    p = CACHE/f"waveforms_q{q:.10f}.npz"
    if p.exists():
        d = np.load(p); return d["t_bhpt"], d["h_bhpt_re"] + 1j*d["h_bhpt_im"]
    import BHPTNRSur1dq1e4 as bhptsur
    t, hd = bhptsur.generate_surrogate(q=q, calibrated=False, modes=[(2, 2)], neg_modes=False)
    print(f"  (surrogate-loaded q={q})", flush=True)
    return t, hd[(2, 2)]
B._load_bhpt = load_bhpt

LEFT  = [2.0, 2.15, 2.25, 2.35, 2.45]
RIGHT = [2.55, 2.65, 2.75, 2.85, 2.95, 3.0]

def curves(qs):
    out = {}
    for q in qs:
        try:
            t, a, b = B.get_alpha_beta(q)
            out[q] = (t, a)
        except Exception as e:
            print(f"  q={q} FAILED: {e}", flush=True)
    return out

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print("computing left panel...", flush=True); L = curves(LEFT)
    print("computing right panel...", flush=True); R = curves(RIGHT)

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
cmL = plt.cm.viridis(np.linspace(0, 0.9, len(LEFT)))
cmR = plt.cm.plasma(np.linspace(0, 0.9, len(RIGHT)))
for (q, (t, a)), c in zip(L.items(), cmL):
    ax[0].plot(t, a, color=c, lw=1.6, label=f"q={q:g}")
for (q, (t, a)), c in zip(R.items(), cmR):
    ax[1].plot(t, a, color=c, lw=1.6, label=f"q={q:g}")
for i, ttl in enumerate(["q = 2.0 – 2.45", "q = 2.55 – 3.0"]):
    ax[i].set_xlim(-600, 80); ax[i].axvline(0, ls=":", color="gray", lw=1)
    ax[i].grid(alpha=0.3); ax[i].set_xlabel("t_NR / M"); ax[i].set_title(ttl)
    ax[i].legend(fontsize=8, ncol=2)
ax[0].set_ylabel("alpha(t)")
fig.suptitle("wf_nu_hybrid_global (even-nu): alpha(t) across finely-spaced q in [2, 3]")
fig.tight_layout()
out = ROOT/"Agentic_plots"/"wf_nu_hybrid_global"/"alpha_fine_lowq_2to3.png"
fig.savefig(out, dpi=120)
print(f"saved {out}", flush=True)
