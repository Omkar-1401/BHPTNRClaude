"""Full plot set for gwr_betaqnm (bfree variant), in the house format.

Mirrors NRBHP_gwr_energy_stiff_plots.py: per-q waveform / zoomed / params panels, the
alpha-beta overlay, and a coefficients-vs-q figure.  The params panel's third row shows
beta's THREE CORNERS (B0 imposed, Bm fitted, B1 = QNM imposed) rather than a drive
coordinate, since that is what distinguishes this model.

rcdefaults() AFTER the model imports -- gw_remnant's gw_plotter sets font.size=18 /
STIXGeneral / figsize (14,10) at import time.

Usage:  python NRBHP_gwr_betaqnm_plots.py [bfree|parfree]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
import fit_scaling_gw_remnant_energy as G
import fit_scaling_gwr_energy_global as GG
import fit_scaling_gwr_energy_flux as FL
import fit_scaling_gwr_betaqnm as BQ

mpl.rcdefaults(); mpl.rcParams["text.usetex"] = False

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "bfree"
FN = {"bfree": "coeffs_bfree.json", "parfree": "coeffs_parfree.json"}[VARIANT]
d = json.loads((ROOT/"gwr_betaqnm_results"/FN).read_text())
TH = np.asarray(d["theta"], float); BFREE = bool(d.get("bfree", False))
PRE = f"gwr_betaqnm_{VARIANT}"
PLOT_DIR = ROOT/"Agentic_plots"/"gwr_betaqnm"; PLOT_DIR.mkdir(parents=True, exist_ok=True)
Q_INPUTS = [2.0, 3.0, 5.0, 8.0]
ALLQ = [2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
_PARAM_RC = {"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
             "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
             "axes.edgecolor": "#6b6a63", "axes.labelcolor": "#1a1a19",
             "text.color": "#1a1a19", "xtick.color": "#6b6a63",
             "ytick.color": "#6b6a63", "axes.linewidth": 0.6, "lines.linewidth": 1.3,
             "figure.dpi": 140, "savefig.bbox": "tight", "text.usetex": False}
C_A, C_B, C_E = "#eb6834", "#2a78d6", "#4a3aa7"; MUTED, GRID = "#6b6a63", "#e3e2dd"

def build(q):
    case = FL.add_flux(G.load_case(q, 6, 10)); P = BQ.prep(q, case)
    c0, c1, A, m = TH[:4]; b = TH[4] if BFREE else 0.0
    alpha, beta, B0, Bm, gb = BQ.shape(P, c0, c1, A, m, b)
    bc = G.creative.cumulative_trapezoid(beta, P["t"])
    tau0 = bc - float(np.interp(G.T_ANCHOR, P["t"], bc))
    r = minimize_scalar(lambda s: GG.err_at_t0(s, alpha, tau0, case["h_bhpt"],
                                               case["t_nr"], case["h_nr"]),
                        bounds=(-195., 45.), method="bounded",
                        options={"xatol": 1e-3, "maxiter": 80})
    t0 = float(r.x); tau = t0 + tau0
    t_nr, h_nr = case["t_nr"], case["h_nr"]
    msk = (t_nr >= max(t_nr[0], tau[0])) & (t_nr <= min(t_nr[-1], tau[-1]))
    tc, href = t_nr[msk], h_nr[msk]
    h0 = np.interp(tc, tau, alpha)*(np.interp(tc, tau, case["h_bhpt"].real)
                                    + 1j*np.interp(tc, tau, case["h_bhpt"].imag))
    z = np.sum(href*h0.conjugate())
    return dict(P=P, case=case, tau=tau, alpha=alpha, beta=beta, err=float(r.fun),
                B0=B0, Bm=Bm, B1=P["B1"], tc=tc, href=href,
                hmod=h0*np.exp(1j*float(np.angle(z))))

res = {q: build(q) for q in Q_INPUTS}
paths = []
def _st(ax):
    ax.grid(True, color=GRID, lw=0.5, alpha=0.9); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
def tag(q): return (f"q{int(q)}" if q == int(q) else f"q{q}") + ("_extrap" if q < 3 else "")

for q in Q_INPUTS:
    r = res[q]
    with plt.rc_context(_PARAM_RC):
        fig, ax = plt.subplots(3, 1, figsize=(9.0, 6.4), sharex=True)
        mk = (r["tau"] >= -1000) & (r["tau"] <= 100)
        ax[0].plot(r["tau"][mk], r["alpha"][mk], color=C_A); ax[0].set_ylabel("$\\alpha(t)$")
        ax[1].plot(r["tau"][mk], r["beta"][mk], color=C_B); ax[1].set_ylabel("$\\beta(t)$")
        for y, c, lab in ((r["B0"], "#888", "$B_0=X_1^{6/5}$"),
                          (r["Bm"], "#c33", "$B_m$ (fitted)"),
                          (r["B1"], "#2a7", "$B_1$ (QNM)")):
            ax[1].axhline(y, color=c, lw=0.7, ls=(0, (4, 3)), label=lab)
        ax[1].legend(fontsize=7, loc="lower right", frameon=False)
        ax[2].plot(r["tau"][mk], r["P"]["u"][mk], color=C_E, label="$u$ (to merger)")
        ax[2].plot(r["tau"][mk], r["P"]["v"][mk], color="#a33", ls="--",
                   label="$v$ (merger$\\to$ringdown)")
        ax[2].set_ylabel("ramps"); ax[2].legend(fontsize=7, frameon=False)
        ax[2].set_xlabel("$t\\ [M]$   (merger at $t=0$)")
        ax[0].set_title(f"$q={q:g}$  $\\nu={G.get_nu(q):.4f}$  "
                        f"$\\mathcal{{E}}={r['err']:.3g}$"
                        f"   ({'extrapolation' if q < 3 else 'training range [3, 8]'})\n"
                        f"$B_0={r['B0']:.5f}$, $B_m={r['Bm']:.5f}$, $B_1={r['B1']:.5f}$",
                        pad=6, fontsize=8.5)
        for a in ax: _st(a); a.axvline(0, color=MUTED, lw=0.6, ls=(0, (4, 3)))
        fig.tight_layout(); p = PLOT_DIR/f"{PRE}_{tag(q)}_params.pdf"
        fig.savefig(p); plt.close(fig); paths.append(p)

    for zoom, nm in ((False, "waveform"), (True, "zoomed")):
        fig, a2 = plt.subplots(1, 2, figsize=(10, 4) if not zoom else (6, 4),
                               squeeze=False)
        axx = a2[0]
        for k, (lo, hi) in enumerate((((-1000, 100),) if zoom else ((-1000, 100), (-150, 60)))):
            m = (r["tc"] >= lo) & (r["tc"] <= hi)
            axx[k].plot(r["tc"][m], r["href"][m].real, "k-", lw=1.0, label="NR")
            axx[k].plot(r["tc"][m], r["hmod"][m].real, "r--", lw=1.0, label="betaqnm")
            axx[k].set_xlabel("t / M"); axx[k].set_ylabel("Re h22"); axx[k].grid(True)
            axx[k].legend(fontsize=8)
            if zoom: break
        fig.suptitle(f"q={q:g}   mathcalE={r['err']:.3g}"); fig.tight_layout()
        p = PLOT_DIR/f"{PRE}_{tag(q)}_{nm}.pdf"; fig.savefig(p); plt.close(fig); paths.append(p)

# corners vs q
cs = {q: build(q) for q in ALLQ}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
qs = np.array(ALLQ)
a1.plot(qs, [cs[q]["B0"] for q in ALLQ], "o-", color="#888", label="$B_0=X_1^{6/5}$ (imposed)")
a1.plot(qs, [cs[q]["Bm"] for q in ALLQ], "s-", color="#c33", label="$B_m$ (fitted)")
a1.plot(qs, [cs[q]["B1"] for q in ALLQ], "^-", color="#2a7", label="$B_1$ QNM (imposed)")
a1.axvline(3, color=MUTED, ls=":", lw=0.8); a1.set_xlabel("q"); a1.set_ylabel("beta corner")
a1.set_title("beta's three corners"); a1.legend(fontsize=8); a1.grid(True)
a2.semilogy(qs, [cs[q]["err"] for q in ALLQ], "o-", color="#2a78d6", label="betaqnm bfree")
a2.axhline(4.7516e-4, color="#c33", ls="--", lw=1.0, label="fluxanchored median")
a2.axvline(3, color=MUTED, ls=":", lw=0.8); a2.set_xlabel("q")
a2.set_ylabel("mathcalE"); a2.set_title("mismatch vs q"); a2.legend(fontsize=8); a2.grid(True)
fig.suptitle(f"gwr-betaqnm {VARIANT}: anchors and accuracy across q"); fig.tight_layout()
for _e in ("pdf","png"):
    p = PLOT_DIR/f"{PRE}_corners_vs_q.{_e}"; fig.savefig(p); paths.append(p)
plt.close(fig)
for p in paths: print("wrote", p)
