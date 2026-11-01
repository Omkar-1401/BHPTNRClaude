"""Figures for the pass-36 D1 leader (18c), in the style of spin_dep/plot_spin_paper_figs.py.  No fitting.

    1. p36_calibration_chi+0.6     before / after calibration, Re h22, q = 3, 5, 8 at chi1 = +0.6 (Islam+ Fig. 7 style)
    2. p36_alpha_beta              alpha(s), beta(s) vs s = t - t_peak,pp (m1), q = 3, 5, 8, one curve per T96 chi1
    3. p36_mismatch_vs_chi1        T96 E vs chi1 at q = 3, 5, 8
    4. p36_hist_train_val_test     log10 E: train T96, validation H270, test R100 (test: score VALUES only, no locations)
    5. p36_mismatch_map            E over (q, chi1) on T96 + H270 (test points deliberately not shown)
The calibrated waveform is rebuilt from the model's own alpha and tau; its E must equal the scorer's to 1e-6, and the
recomputed T96 must equal the stored per-case scores (ratio 1 within 2e-3).
    python plot_p36.py      -> exploration/plots_p36/*.{pdf,png} + p36_scores.json (T96 + H270 per case)
"""
from __future__ import annotations
import json, sys
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
argv0 = sys.argv[0]; sys.argv = [argv0, "D1"]
import p36_rdslope as D                         # noqa: E402
P6 = D.P6; P6.get = D.get                       # BEFORE the fork
import fit_pn_anchored_spin as S                # noqa: E402
import plot_spin_paper_figs as PF               # noqa: E402  (style, palette, held-out points)
import matplotlib                               # noqa: E402
matplotlib.use("Agg")
matplotlib.rcdefaults()                         # after the model imports (CLAUDE.md plotting trap)
import matplotlib.pyplot as plt                 # noqa: E402
import matplotlib.colors as mcolors             # noqa: E402

R = json.loads((HERE / "p36" / "p36_D1.json").read_text())
TH = list(map(float, R["theta"]))
OUT = HERE / "plots_p36"; OUT.mkdir(exist_ok=True)
LABEL = "pass 36 D1 (18 coef)"
k6 = lambda q, c: f"{q:.6f}_{c:+.4f}"
QS3 = [3.0, 5.0, 8.0]


def _full(k):
    return P6._full((TH, k))


def shape(k):
    """alpha(t), tau(t) (NR clock) and E, exactly as P6._full aligns them."""
    _, t0 = P6.err(np.asarray(TH), k, "fit")
    e, t1 = P6.err(np.asarray(TH), k, "full", seed=t0)
    c = P6.get(k); case = c["case"]; t = case["t_bhpt"]
    alpha, beta = P6.ab(np.asarray(TH), k, "full")
    bc = P6.creative.cumulative_trapezoid(beta, t); ts = bc - np.interp(S.T_ANCHOR, t, bc)
    return dict(e=e, alpha=alpha, beta=beta, tau=t1 + ts, s=t - c["tpk"], case=case)


def calibrated(k):
    w = shape(k); case = w["case"]
    t, h, tn, hn, tau, alpha = case["t_bhpt"], case["h_bhpt"], case["t_nr"], case["h_nr"], w["tau"], w["alpha"]
    m = (tn >= max(tn[0], tau[0])) & (tn <= min(tn[-1], tau[-1]))
    tc, r = tn[m], hn[m]
    g = np.interp(tc, tau, alpha) * (np.interp(tc, tau, h.real) + 1j * np.interp(tc, tau, h.imag))
    Z = np.sum(r * np.conj(g)); n1, n2 = np.sum(np.abs(r) ** 2), np.sum(np.abs(g) ** 2)
    e2 = float((n1 + n2 - 2 * abs(Z)) / (2 * n1))
    assert abs(e2 / w["e"] - 1) < 1e-6, (k, w["e"], e2)
    return dict(t=t, h=h, tn=tn, hn=hn, tc=tc, hnc=r, hcal=g * Z / abs(Z), E=w["e"])


def _ab(k):
    w = shape(k); keep = (w["s"] >= -1000) & (w["s"] <= 80)
    return k, w["s"][keep][::4], w["alpha"][keep][::4], w["beta"][keep][::4]


def fig_wave(waves):
    fig, axs = plt.subplots(3, 2, figsize=(12, 7.8), sharex=True)
    for i, q in enumerate(QS3):
        w = waves[q]; a, b = axs[i]
        a.plot(w["tn"], w["hn"].real, color=PF.C_NR, lw=1.6, label="NR (NRHybSur3dq8), $t$ in $M$")
        a.plot(w["t"], w["h"].real, color=PF.C_RAW, lw=1.2, label="raw ppBHPT, $t$ in $m_1$")
        b.plot(w["tc"], w["hnc"].real, color="#e34948", lw=1.6, ls="--", label="NR (NRHybSur3dq8)")
        b.plot(w["tc"], w["hcal"].real, color="#000000", lw=1.1, label="calibrated ppBHPT")
        a.set_ylabel(r"Re $h_{22}$" + f"\nq = {q:g}")
        b.text(0.02, 0.9, rf"$\mathcal{{E}}$ = {w['E']:.2e}", transform=b.transAxes, fontsize=10)
        for ax in (a, b):
            ax.set_xlim(*PF.XLIM); PF.style(ax)
    axs[0, 0].set_title("before calibration", fontsize=11); axs[0, 1].set_title("after calibration", fontsize=11)
    for ax in axs[0]:
        ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2)
    for ax in axs[-1]:
        ax.set_xlabel("$t$")
    fig.suptitle(rf"{LABEL}: ppBHPT vs NR at $\chi_1$ = +0.6", fontsize=12, y=1.0); fig.tight_layout()
    return fig


def fig_ab(ab):
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True)
    cm = plt.get_cmap("coolwarm"); nrm = mcolors.Normalize(-0.6, 0.8)
    for j, q in enumerate(QS3):
        for c in S.CHIS:
            s, a, b = ab[(q, c)]
            axs[0, j].plot(s, a, color=cm(nrm(c)), lw=1.3)
            axs[1, j].plot(s, b, color=cm(nrm(c)), lw=1.3)
        axs[0, j].set_title(f"q = {q:g}", fontsize=11)
        axs[1, j].set_xlabel(r"$s = t - t_{\rm peak,pp}$  ($m_1$)")
        for ax in axs[:, j]:
            ax.axvline(0, color="#999999", lw=0.8, ls=":"); PF.style(ax)
    axs[0, 0].set_ylabel(r"$\alpha(t)$"); axs[1, 0].set_ylabel(r"$\beta(t)$")
    sm = plt.cm.ScalarMappable(cmap=cm, norm=nrm)
    fig.colorbar(sm, ax=axs, shrink=0.85, pad=0.015, label=r"$\chi_1$")
    fig.suptitle(rf"{LABEL}: calibration functions on the T96 spins", fontsize=12)
    return fig


def fig_hist(train, val, test):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    sets = [(train, "training T96", "--", PF.C_TRAIN), (val, "validation H270", "-", PF.C_BLUE),
            (test, "test R100 (unseen)", "-", PF.C_ORANGE)]
    allv = np.log10(np.concatenate([train, val, test]))
    bins = np.linspace(np.floor(allv.min() * 4) / 4, np.ceil(allv.max() * 4) / 4, 25); top = 0
    for v, lab, ls, col in sets:
        lv = np.log10(v)
        ax.hist(lv, bins=bins, density=True, histtype="step", ls=ls, lw=1.8, color=col,
                label=f"{lab} ({len(v)}), median {np.median(v):.2e}")
        top = max(top, np.histogram(lv, bins, density=True)[0].max())
    ax.set_ylim(0, 1.5 * top); ax.set_xlabel(r"$\log_{10}\,\mathcal{E}$"); ax.set_ylabel("probability density")
    ax.set_title(LABEL, fontsize=11); ax.legend(frameon=False, fontsize=9, loc="upper left"); PF.style(ax)
    return fig


def fig_map(pts):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    q = np.array([p[0] for p in pts]); c = np.array([p[1] for p in pts]); e = np.array([p[2] for p in pts])
    tr = np.array([p[3] for p in pts])
    nrm = mcolors.LogNorm(vmin=e.min(), vmax=e.max())
    sc = ax.scatter(q[~tr], c[~tr], c=e[~tr], cmap="viridis", norm=nrm, marker="s", s=46, lw=0)
    ax.scatter(q[tr], c[tr], c=e[tr], cmap="viridis", norm=nrm, marker="o", s=70, edgecolors="white", linewidths=1.0)
    cb = fig.colorbar(sc, ax=ax, label=r"$\mathcal{E}$")
    cb.set_ticks([5e-5, 1e-4, 2e-4, 5e-4]); cb.set_ticklabels(["5e-5", "1e-4", "2e-4", "5e-4"])
    cb.ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("$q$"); ax.set_ylabel(r"$\chi_1$")
    ax.set_title(f"{LABEL}: circles = training T96, squares = validation H270", fontsize=10); PF.style(ax)
    return fig


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig); print(f"  wrote plots_p36/{name}.{{pdf,png}}", flush=True)


def main():
    H = PF.heldout_points()
    with mp.get_context("fork").Pool(96) as pool:
        e96 = np.array(pool.map(_full, P6.T96)); eh = np.array(pool.map(_full, H))
        ab = {k: (s, a, b) for k, s, a, b in pool.map(_ab, [(q, c) for q in QS3 for c in S.CHIS])}
        waves = dict(zip(QS3, pool.map(calibrated, [(q, 0.6) for q in QS3])))
    r = e96 / np.array([R["T96_cases"][k6(*k)] for k in P6.T96])
    print(f"T96 recomputed / stored: {r.min():.4f}..{r.max():.4f}", flush=True)
    assert np.allclose(r, 1, atol=2e-3), r
    print(f"T96 median {np.median(e96):.4e} | H270 median {np.median(eh):.4e} (stored {R['H270'][1]:.4e})", flush=True)
    (OUT / "p36_scores.json").write_text(json.dumps(dict(
        T96={k6(*k): float(v) for k, v in zip(P6.T96, e96)}, H270={k6(*k): float(v) for k, v in zip(H, eh)}), indent=1))
    test = np.array(list(json.loads((HERE / "t100" / "p36.json").read_text())["cases"].values()))[:100]   # R100 values
    save(fig_wave(waves), "p36_calibration_chi+0.6")
    save(fig_ab(ab), "p36_alpha_beta")
    fig, ax = plt.subplots(figsize=(6, 4.2)); PF.draw_vschi(ax, {PF.key(*k): v for k, v in zip(P6.T96, e96)}, LABEL)
    save(fig, "p36_mismatch_vs_chi1")
    save(fig_hist(e96, eh, test), "p36_hist_train_val_test")
    save(fig_map([(q, c, v, True) for (q, c), v in zip(P6.T96, e96)] + [(q, c, v, False) for (q, c), v in zip(H, eh)]),
         "p36_mismatch_map")


if __name__ == "__main__":
    main()
