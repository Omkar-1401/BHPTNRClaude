"""Heatmaps for the fluxanchored_spin_split variant on T96 (results/flux_spin_split_T96_T-1.json)."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "results" / "flux_spin_split_T96_T-1.json").read_text())
new = {(r["q"], r["chi1"]): r for r in d["cases"]}; old = {(r["q"], r["chi1"]): r for r in d["baselines"]["pn13"]}
qs = sorted({k[0] for k in new}); cs = sorted({k[1] for k in new})
def grid(src, key, f=np.log10):
    Z = np.full((len(cs), len(qs)), np.nan)
    for (q, c), r in src.items():
        Z[cs.index(c), qs.index(q)] = f(r[key])
    return Z
panels = [(grid(new, "full_E_full"), r"$\log_{10}\mathcal{E}$ full window (full-aligned)", "viridis", None),
          (grid(new, "full_E_full", lambda v: v) / grid(old, "full_E_full", lambda v: v), "ratio to installed pn_anchored_spin (full window)", "RdBu_r", (0, 2)),
          (grid(new, "insp_E_insp"), r"$\log_{10}\mathcal{E}_{\rm insp}$ (insp-aligned)", "viridis", None),
          (grid(new, "insp_E_MR"), r"$\log_{10}\mathcal{E}_{\rm MR}$ (insp-aligned)", "viridis", None)]
fig, axs = plt.subplots(2, 2, figsize=(13, 8.6))
for ax, (Z, ttl, cm, lim) in zip(axs.ravel(), panels):
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cm, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None)
    for i in range(len(cs)):
        for j in range(len(qs)):
            ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="k" if cm == "RdBu_r" else "w")
    ax.set_xticks(range(len(qs))); ax.set_xticklabels([f"{q:.2f}" for q in qs], rotation=60, fontsize=8)
    ax.set_yticks(range(len(cs))); ax.set_yticklabels([f"{c:+.1f}" for c in cs], fontsize=8)
    ax.set_xlabel("q"); ax.set_ylabel(r"$\chi_1$"); ax.set_title(ttl, fontsize=10); fig.colorbar(im, ax=ax, shrink=0.9)
fig.suptitle("fluxanchored_spin_split (variant, 18 coefficients, T96 fit): flux form + derived PN spin-orbit inspiral, "
             "flux-form MR copy, Ori–Thorne window", fontsize=10)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(HERE / "Agentic_spin_split_plots" / "fluxanchored_spin_split" / f"T96_heatmaps.{e}", dpi=150)
print("wrote T96_heatmaps")
