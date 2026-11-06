"""Figures for the two CV-selected split models (2026-10-15), same set and style as exploration/plot_p36.py.  No fitting.

    A = slim split, Ori–Thorne window from T = -2      (results/explore/slim__win-2.0.json; 14 coef)
    B = inspiral b = 0, slim MR, window from T = -3     (results/cv/tline__foldall.json; 13 coef + fitted T0)
Per model, in Agentic_spin_split_plots/<name>/:
    1. <name>_calibration_chi+0.6     before / after calibration, Re h22, q = 3, 5, 8 at chi1 = +0.6 (Islam+ Fig. 7 style)
    2. <name>_alpha_beta              alpha(s), beta(s), s = t - t_peak,pp (m1), q = 3, 5, 8, one curve per T96 chi1
    3. <name>_mismatch_vs_chi1        T96 E vs chi1 at q = 3, 5, 8 (Fig. 9a style)
    4. <name>_hist_train_val_test     log10 E: training T96, validation H270, test F100; one bin set for A and B
    5. <name>_mismatch_map            E over (q, chi1) on T96 + H270 (test points not shown, as plot_p36)
Scores: T96 / H270 from the stored fits, F100 from results/f100/<name>.json (score_f100.py, the one logged look).  E is
full-window, full alignment.  The calibrated waveform is rebuilt from the model's alpha, beta and aligned t0; its E must
equal the scorer's to 1e-6, and the rebuilt T96 E must match the stored per-case score (within 2e-3).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from multiprocessing import get_context

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.append(str(HERE.parent / "spin_dep"))   # append: spin_dep has a same-named split module
import cv_split as C
X, FF, M = C.X, C.FF, C.M
import plot_spin_paper_figs as PF
import matplotlib
matplotlib.use("Agg")
matplotlib.rcdefaults()                       # after the model imports (plotting trap)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

QS3 = [3.0, 5.0, 8.0]; CHIS = list(M.S.CHIS)
OUTD = HERE / "Agentic_spin_split_plots"


def models():
    a = json.loads((X.OUT / "slim__win-2.0.json").read_text())
    b = json.loads((HERE / "results" / "cv" / "tline__foldall.json").read_text())
    fa = json.loads((HERE / "results" / "f100" / "slim__win-2.0.json").read_text())
    fb = json.loads((HERE / "results" / "f100" / "tline.json").read_text())
    rows = lambda rs: {(r["q"], r["chi1"]): r["full_E_full"] for r in rs}
    return {
        "split_A_slim_T-2": dict(label="split A: slim, window from T = -2 (14 coef)", mr="slim",
                                 ti=np.array(a["theta_insp"]), x=a["x_mr"], window=("const", -2.0),
                                 T96=rows(a["cases"]["T96"]), H270=rows(a["cases"]["H270"]), F100=rows(fa["cases"]["F100"])),
        "split_B_b0_T-3": dict(label="split B: inspiral b = 0, window from T = -3 (13 coef + T0)", mr="slim",
                               ti=np.array(b["theta_insp"]), x=b["x_mr"], window=("line", b["chosen"]["T0"], b["chosen"]["T1"]),
                               T96=rows(b["train_rows"]), H270=rows(fb["cases"]["H270"]), F100=rows(fb["cases"]["F100"]))}


def set_win(w):
    if w[0] == "const":
        X.set_window(w[1])
    else:
        C.set_tline(w[1], w[2])


MOD = {}


def shape(k):
    """alpha, beta, tau (NR clock) and E with the scorer's own full alignment (fit-grid seed, then full grid)."""
    d = MOD["m"]; c = M.CASES[k]
    al_f, be_f = X.ab("split", d["ti"], ("form", d["mr"], d["x"]), *k, c, "fit")
    _, t0 = M.score(al_f, be_f, c, "fit", "full", -50.0, wide=True)
    al, be = X.ab("split", d["ti"], ("form", d["mr"], d["x"]), *k, c, "full")
    errs, t1 = M.score(al, be, c, "full", "full", t0, wide=True)
    t = c["t_bhpt"]; ts = M.F.tau_shape(be, t)
    tpk = t[int(np.argmax(np.abs(c["h_bhpt"])))]
    return dict(e=errs[2], alpha=al, beta=be, tau=t1 + ts, s=t - tpk, c=c)


def _e(k):
    return shape(k)["e"]


def calibrated(k):
    w = shape(k); c = w["c"]
    t, h, tn, hn, tau, alpha = c["t_bhpt"], c["h_bhpt"], c["t_nr"], c["h_nr"], w["tau"], w["alpha"]
    m = (tn >= max(tn[0], tau[0])) & (tn <= min(tn[-1], tau[-1]))
    tc, r = tn[m], hn[m]
    g = np.interp(tc, tau, alpha) * (np.interp(tc, tau, h.real) + 1j * np.interp(tc, tau, h.imag))
    Z = np.vdot(g, r); g = g * Z / abs(Z)
    e2 = float(np.sum(np.abs(r - g) ** 2) / (2 * np.sum(np.abs(r) ** 2)))
    assert abs(e2 / w["e"] - 1) < 1e-6, (k, w["e"], e2)
    return dict(t=t, h=h, tn=tn, hn=hn, tc=tc, hnc=r, hcal=g, E=w["e"])


def _ab(k):
    w = shape(k); keep = (w["s"] >= -1000) & (w["s"] <= 80)
    return k, w["s"][keep][::4], w["alpha"][keep][::4], w["beta"][keep][::4]


def fig_wave(waves, label):
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
    fig.suptitle(rf"{label}: ppBHPT vs NR at $\chi_1$ = +0.6", fontsize=12, y=1.0); fig.tight_layout()
    return fig


def fig_ab(ab, label):
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True)
    cm = plt.get_cmap("coolwarm"); nrm = mcolors.Normalize(-0.6, 0.8)
    for j, q in enumerate(QS3):
        for c in CHIS:
            s, a, b = ab[(q, c)]
            axs[0, j].plot(s, a, color=cm(nrm(c)), lw=1.3); axs[1, j].plot(s, b, color=cm(nrm(c)), lw=1.3)
        axs[0, j].set_title(f"q = {q:g}", fontsize=11)
        axs[1, j].set_xlabel(r"$s = t - t_{\rm peak,pp}$  ($m_1$)")
        for ax in axs[:, j]:
            ax.axvline(0, color="#999999", lw=0.8, ls=":"); PF.style(ax)
    axs[0, 0].set_ylabel(r"$\alpha(t)$"); axs[1, 0].set_ylabel(r"$\beta(t)$")
    fig.colorbar(plt.cm.ScalarMappable(cmap=cm, norm=nrm), ax=axs, shrink=0.85, pad=0.015, label=r"$\chi_1$")
    fig.suptitle(rf"{label}: calibration functions on the T96 spins", fontsize=12)
    return fig


def fig_hist(d, bins):
    fig, ax = plt.subplots(figsize=(6.4, 4.4)); top = 0
    for g, lab, ls, col in (("T96", "training T96", "--", PF.C_TRAIN), ("H270", "validation H270", "-", PF.C_BLUE),
                            ("F100", "test F100 (unseen)", "-", PF.C_ORANGE)):
        v = np.array(list(d[g].values())); lv = np.log10(v)
        ax.hist(lv, bins=bins, density=True, histtype="step", ls=ls, lw=1.8, color=col,
                label=f"{lab} ({len(v)}), median {np.median(v):.2e}")
        top = max(top, np.histogram(lv, bins, density=True)[0].max())
    ax.set_xlim(bins[0], bins[-1]); ax.set_ylim(0, 1.5 * top)
    ax.set_xlabel(r"$\log_{10}\,\mathcal{E}$"); ax.set_ylabel("probability density")
    ax.set_title(d["label"], fontsize=11); ax.legend(frameon=False, fontsize=9, loc="upper left"); PF.style(ax)
    return fig


def fig_map(d):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    pts = [(q, c, v, True) for (q, c), v in d["T96"].items()] + [(q, c, v, False) for (q, c), v in d["H270"].items()]
    q = np.array([p[0] for p in pts]); c = np.array([p[1] for p in pts]); e = np.array([p[2] for p in pts])
    tr = np.array([p[3] for p in pts]); nrm = mcolors.LogNorm(vmin=e.min(), vmax=e.max())
    sc = ax.scatter(q[~tr], c[~tr], c=e[~tr], cmap="viridis", norm=nrm, marker="s", s=46, lw=0)
    ax.scatter(q[tr], c[tr], c=e[tr], cmap="viridis", norm=nrm, marker="o", s=70, edgecolors="white", linewidths=1.0)
    fig.colorbar(sc, ax=ax, label=r"$\mathcal{E}$")
    ax.set_xlabel("$q$"); ax.set_ylabel(r"$\chi_1$")
    ax.set_title(f"{d['label']}\ncircles = training T96, squares = validation H270", fontsize=10); PF.style(ax)
    return fig


def save(fig, name, sub):
    (OUTD / sub).mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTD / sub / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig); print(f"  wrote Agentic_spin_split_plots/{sub}/{name}.{{pdf,png}}", flush=True)


def main():
    D = models()
    for k in X.T96:
        M.CASES[k] = M.prepare(*k)
    allv = np.log10(np.concatenate([list(d[g].values()) for d in D.values() for g in ("T96", "H270", "F100")]))
    bins = np.arange(np.floor(allv.min() * 4) / 4, np.ceil(allv.max() * 4) / 4 + 1e-9, 0.1)
    for name, d in D.items():
        set_win(d["window"]); MOD["m"] = d               # before the fork
        with get_context("fork").Pool(96) as pool:
            e96 = np.array(pool.map(_e, X.T96))
            ab = {k: (s, a, b) for k, s, a, b in pool.map(_ab, [(q, c) for q in QS3 for c in CHIS])}
            waves = dict(zip(QS3, pool.map(calibrated, [(q, 0.6) for q in QS3])))
        r = e96 / np.array([d["T96"][k] for k in X.T96])
        print(f"{name}: T96 rebuilt / stored {r.min():.4f}..{r.max():.4f}", flush=True)
        assert np.allclose(r, 1, atol=2e-3), r
        for g in ("T96", "H270", "F100"):
            v = np.array(list(d[g].values())); print(f"  {g}: {np.median(v):.3e} / {v.mean():.3e} / {v.max():.2e}")
        save(fig_wave(waves, d["label"]), f"{name}_calibration_chi+0.6", name)
        save(fig_ab(ab, d["label"]), f"{name}_alpha_beta", name)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        PF.draw_vschi(ax, {PF.key(*k): v for k, v in d["T96"].items()}, d["label"])
        save(fig, f"{name}_mismatch_vs_chi1", name)
        save(fig_hist(d, bins), f"{name}_hist_train_val_test", name)
        save(fig_map(d), f"{name}_mismatch_map", name)


if __name__ == "__main__":
    main()
