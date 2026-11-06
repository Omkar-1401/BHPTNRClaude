"""figE6_testmass.png: alpha(t), beta(t)/X1 at q = 1000 (nu ~ 1e-3, the test-mass limit) for pass 40, lean-12, GPR, RBF.
pass 40 / lean-12 from training_size/testmass_curves.py (their chain on the ppBHPT waveform; O(nu) ISCO-plunge remnant above
q = 10, user 2026-11-03); GPR / RBF are the efit_lin models (T96-trained, refined through E) evaluated on S_C.
Identity map (alpha = 1, beta = X1) is the exact limit.  No NR exists here: this is behaviour, not error.

    python plot_testmass.py
"""
import numpy as np
import common as C, efit as F, efit_lin as L, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_efit import STYLE, INK, INK2, CAT, RES, OUT

Q, CHIS = 1000.0, (0.0, 0.6)
X1 = Q / (1 + Q)
YL, YT = (0.5, 2.0), (0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)


def main():
    keys = [(Q, c) for c in CHIS]
    X = F.pfeat(list(C.T96)).numpy(); Y = L.init_targets(list(C.T96))
    cur = {}
    for meth in ("gpr", "rbf"):
        m, _ = L.build(meth, X, Y); m.load_state_dict(torch.load(F.MOD / f"efit_lin_{meth}.pt"))
        a, b = F.ab_numpy(m, keys); cur[meth] = [(F.S_C, a[j], b[j]) for j in range(len(keys))]
    for mod in ("p40", "lean12"):
        z = np.load(RES / f"testmass_{mod}.npz")
        cur[mod] = [(z[f"q{Q:g}_chi{c:+.1f}_s"], z[f"q{Q:g}_chi{c:+.1f}_alpha"], z[f"q{Q:g}_chi{c:+.1f}_beta"]) for c in CHIS]
    plt.rcParams.update(STYLE)
    sty = dict(p40=(INK, 2.6, "-", "pass 40 (18 coef.)"), lean12=(INK2, 2.2, "--", "lean-12 (12 coef.)"),
               gpr=(CAT[1], 1.7, "-", "GPR"), rbf=(CAT[0], 1.7, "-", "RBF"))
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.6), sharex=True)
    for j, c in enumerate(CHIS):
        for k in ("gpr", "rbf", "lean12", "p40"):
            s, a, b = cur[k][j]; m = (s >= -2000) & (s <= 100)
            col, lw, ls, lab = sty[k]
            axs[0, j].plot(s[m], a[m], color=col, lw=lw, ls=ls, label=lab)
            axs[1, j].plot(s[m], b[m] / X1, color=col, lw=lw, ls=ls)
            i = np.argmin(abs(s - (-1000)))
            print(f"chi1={c:+.1f} {k:6s}: alpha(-1000) {a[i]:.4f}  beta/X1(-1000) {b[i] / X1:.4f}  "
                  f"max|alpha-1| {np.max(abs(a[m] - 1)):.3f}  max|beta/X1-1| {np.max(abs(b[m] / X1 - 1)):.3f}")
        for r in (0, 1):
            axs[r, j].axhline(1.0, color=INK, lw=0.8, ls=":")
            axs[r, j].set_yscale("log"); axs[r, j].set_ylim(*YL)
            axs[r, j].set_yticks(YT); axs[r, j].set_yticklabels([f"{v:g}" for v in YT]); axs[r, j].minorticks_off()
        offs = []                                  # off-scale peaks, written out instead of drawn
        for k in ("gpr", "rbf"):
            s, a, b = cur[k][j]; m = (s >= -2000) & (s <= 100)
            for r, (y, nm) in enumerate(((a, "α"), (b / X1, "β/X1"))):
                i = np.argmax(np.abs(np.log(y[m]))); v = y[m][i]
                if not YL[0] <= v <= YL[1]:
                    offs.append((r, f"{sty[k][3]}: {nm} reaches {v:.3g} at s = {s[m][i]:+.0f}"))
        for r in (0, 1):
            txt = "\n".join(t for rr, t in offs if rr == r)
            if txt:
                axs[r, j].text(0.02, 0.04, "off scale: " + txt, transform=axs[r, j].transAxes, fontsize=8, color=INK2, va="bottom")
        axs[0, j].set_title(f"q = 1000 (ν ≈ 1.0e-3), χ1 = {c:+.1f}", loc="left", fontsize=10)
        axs[1, j].set_xlabel("s = t − t_peak,pp  [m1]")
    axs[0, 0].set_ylabel("α"); axs[1, 0].set_ylabel("β / X1")
    axs[0, 0].plot([], [], color=INK, lw=0.8, ls=":", label="test-mass limit (identity map)")
    axs[0, 0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    fig.tight_layout(); fig.savefig(OUT / "figE6_testmass.png", dpi=150); plt.close(fig)
    print("wrote", OUT / "figE6_testmass.png")


if __name__ == "__main__":
    main()
