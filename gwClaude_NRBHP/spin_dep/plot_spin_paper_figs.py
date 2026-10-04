"""Paper-style figures for the spin models, after Islam et al. arXiv:2407.18319v2.

  1. hist   -- mathcalE histogram, training vs held-out (their Fig. 5, top-left panel style:
               x = log10(mathcalE), y = probability density; training black dashed step,
               held-out blue solid step; each normalised to unit area).
  2. wave   -- before vs after calibration (their Fig. 7), but rows are three mass ratios at
               one spin instead of three modes: left = raw ppBHPT (m1 clock) vs NR (M clock),
               right = calibrated ppBHPT vs NR.  Re(h22).
  3. vschi  -- mathcalE vs chi1, one curve per q (their Fig. 9a), q = 3, 5, 8.

Models (current coefficients, 2026-09-27):
    pn_anchored_spin      fit_pn_anchored_spin: model9() + coeffs_spin.json theta_spin
    pn_anc_spin_physical  fit_pn_anc_spin_physical: coeffs_physical.json
    fluxanchored_spin     fit_fluxanchored_spin: fluxanchored_spin_results/coeffs_nu.json
    energy_anchored_spin  fit_energy_anchored_spin: energy_anchored_spin_results/coeffs_nu.json
Every number is the model's OWN scoring, full resolution vs NRHybSur3dq8:
    pn_anchored_spin      full_mm (t0 scan on the fit grid, refined at full resolution)
    pn_anc_spin_physical  mismatch (cold t0 scan, full resolution)
    fluxanchored_spin     its score(): coarse t0 scan + refine on the fit grid, then full
    energy_anchored_spin  mismatch (cold t0 scan, full resolution)
Training scores are RECOMPUTED on the current code and compared with the stored per-case
values (the flux/energy fits stored warm-started t0 searches, so tiny differences are
expected; > 1% stops the script).  The figures use the recomputed values.
The calibrated waveform is rebuilt from each model's own alpha(t) and tau(t); its mismatch
must equal the model's to 1e-6.

HELD-OUT = never trained on, and INSIDE the training range only (user rule 2026-09-26: no
extrapolation for validation, so no q < 3): the interior of the dense-heatmap grid,
q in linspace(2, 8, 20) with 3 < q < 8 and chi1 in linspace(-0.6, 0.8, 20) without the two
ends -- 15 x 18 = 270 points, none on the training grid (waveforms already cached).

Outputs: Agentic_spin_plots/<model>/paper/*.{pdf,png} (+ heldout_mismatch.json, which also
holds the recomputed training scores); side-by-sides in Agentic_spin_plots/compare_<tag>/
only with --compare (retired by default 2026-10-01).

    python plot_spin_paper_figs.py                                   # the two pn models
    python plot_spin_paper_figs.py --models fluxanchored_spin energy_anchored_spin --tag energy_flux_spin
    python plot_spin_paper_figs.py --plot-only --models ... --tag ...
"""
from __future__ import annotations
import argparse, json, multiprocessing as mp, os, sys, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
warnings.filterwarnings("ignore")
import fit_pn_anchored_spin as S

PLOTS = HERE / "Agentic_spin_plots"
# validated reference palette (dataviz skill, references/palette.md), fixed slot order
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_NR, C_RAW, C_TRAIN = "#8c8c8c", C_BLUE, "#000000"
Q_WAVE, CHI_WAVE = [3.0, 5.0, 8.0], 0.6
Q_VSCHI = [(3.0, C_BLUE, "o", "-"), (5.0, C_ORANGE, "s", "--"), (8.0, C_AQUA, "^", "-.")]
XLIM = (-1000.0, 80.0)


def key(q, c):
    return f"{q:.4f}_{c:+.1f}"


def _tau_shape(beta, t, t_anchor):
    bc = S.creative.cumulative_trapezoid(beta, t)
    return bc - float(np.interp(t_anchor, t, bc))


# ------------------------------------------------------------------ the four models
# Each entry: label, stored per-case dict, case builder, score(q, c, case) -> E, and
# shape(q, c, case) -> (E, alpha(t), tau(t)) on the FULL ppBHPT grid by the model's own path.
def _pn_spin():
    th9 = S.model9(); ths = np.array(json.loads(S.COEFFS.read_text())["theta_spin"], float)

    def shape(q, c, case):
        _, t0 = S.mismatch(th9, ths, q, c, case, "fit", -50.0)
        e, t0 = S.mismatch(th9, ths, q, c, case, "full", t0)
        alpha, beta = S.model_ab(th9, ths, q, c, case["x"], case["p_loss"])
        return e, alpha, t0 + _tau_shape(beta, case["t_bhpt"], S.T_ANCHOR)
    return dict(label="pn_anchored_spin (13 coef)", build=S.build_case, shape=shape,
                score=lambda q, c, case: S.full_mm(th9, ths, q, c, case),
                stored=json.loads(S.COEFFS.read_text())["mismatches"])


def _physical():
    import fit_pn_anc_spin_physical as P
    th, _ = P.shipped11(False)

    def shape(q, c, case):
        e, t0 = P.mismatch(th, q, c, case, None)
        alpha, beta = P.model_ab(th, q, c, case["x"], case["p_loss"])
        return e, alpha, t0 + _tau_shape(beta, case["t_bhpt"], S.T_ANCHOR)
    return dict(label="pn_anc_spin_physical (11 coef)", build=S.build_case, shape=shape,
                score=lambda q, c, case: P.mismatch(th, q, c, case, None)[0],
                stored=json.loads(P.COEFFS.read_text())["mismatches"])


def _flux():
    import fit_fluxanchored_spin as FS
    FL = FS.FL
    d = json.loads((FS.RESULTS / "coeffs_nu.json").read_text())
    th = np.array(d["theta"], float); FS.WEIGHT = d["weight"]

    def run(q, c, case):                    # exactly fit_fluxanchored_spin.score, cold nuisance
        p = FS.params_at(q, c, th)
        g = np.linspace(-250.0, 150.0, 81)
        t0 = float(g[int(np.argmin([FL.fast_mismatch(p, case, x, "fit")[0] for x in g]))])
        _, t0 = FL.fast_mismatch(p, case, t0, "fit")
        e, t0 = FL.fast_mismatch(p, case, t0, "full")
        return p, e, t0

    def shape(q, c, case):
        p, e, t0 = run(q, c, case)
        alpha, tau_shape, _ = FL.model_shape(p, case["t_bhpt"], case["losses"])
        return e, alpha, t0 + tau_shape
    return dict(label="fluxanchored_spin (9 coef)", build=FS.build_case, shape=shape,
                score=lambda q, c, case: run(q, c, case)[1], stored=d["mismatches"])


def _energy():
    import fit_energy_anchored_spin as EA
    import fit_fluxanchored_spin as FS
    d = json.loads((EA.RESULTS / "coeffs_nu.json").read_text())
    th = np.array(d["theta"], float); EA.WEIGHT = d["weight"]

    def shape(q, c, case):
        e, t0 = EA.mismatch(th, q, c, case, None)
        a_pp, a_e, b_pp, b_e = EA.params_at(q, c, th)
        E = np.asarray(case["losses"]["e_oft"], float)
        alpha, beta = a_pp * (1.0 + a_e * E), b_pp * (1.0 + b_e * E)
        return e, alpha, t0 + _tau_shape(beta, case["t_bhpt"], EA.G.T_ANCHOR)
    return dict(label="energy_anchored_spin (8 coef)", build=FS.build_case, shape=shape,
                score=lambda q, c, case: EA.mismatch(th, q, c, case, None)[0],
                stored=d["mismatches"])


LOADERS = {"pn_anchored_spin": _pn_spin, "pn_anc_spin_physical": _physical,
           "fluxanchored_spin": _flux, "energy_anchored_spin": _energy}
MODELS: dict = {}


def heldout_points():
    qs = [q for q in np.linspace(2.0, 8.0, 20) if 3.0 < q < 8.0]
    cs = list(np.linspace(-0.6, 0.8, 20))[1:-1]
    on = {(round(q, 6), round(c, 6)) for q in S.TRAIN_Q for c in S.CHIS}
    pts = [(float(q), float(c)) for q in qs for c in cs if (round(q, 6), round(c, 6)) not in on]
    assert len(pts) == 270
    return pts


def _work(a):
    model, q, c = a
    M = MODELS[model]
    return model, q, c, M["score"](q, c, M["build"](q, c))


def calibrated(model, q, c):
    """Raw and calibrated Re(h22) on the NR axis, by the model's own alpha and tau; the
    rebuilt mismatch must equal the model's."""
    M = MODELS[model]
    case = M["build"](q, c)
    t, h, tn, hn = case["t_bhpt"], case["h_bhpt"], case["t_nr"], case["h_nr"]
    e, alpha, tau = M["shape"](q, c, case)
    m = (tn >= max(tn[0], tau[0])) & (tn <= min(tn[-1], tau[-1]))
    tc, r = tn[m], hn[m]
    g = np.interp(tc, tau, alpha) * (np.interp(tc, tau, h.real) + 1j * np.interp(tc, tau, h.imag))
    Z = np.sum(r * np.conj(g))
    n1, n2 = np.sum(np.abs(r) ** 2), np.sum(np.abs(g) ** 2)
    e2 = float((n1 + n2 - 2 * abs(Z)) / (2 * n1))
    assert abs(e2 / e - 1) < 1e-6, (model, q, c, e, e2)
    return dict(t=t, h=h, tn=tn, hn=hn, tc=tc, hnc=r, hcal=g * Z / abs(Z), E=e)


def style(ax):
    ax.grid(True, color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {folder.relative_to(HERE)}/{name}.{{pdf,png}}", flush=True)


def draw_hist(ax, train, held, title):
    lt, lh = np.log10(train), np.log10(held)
    bins = np.linspace(np.floor(min(lt.min(), lh.min()) * 4) / 4,
                       np.ceil(max(lt.max(), lh.max()) * 4) / 4, 25)
    ax.hist(lt, bins=bins, density=True, histtype="step", ls="--", lw=1.8, color=C_TRAIN,
            label=f"training ({len(train)}), median {np.median(train):.2e}")
    ax.hist(lh, bins=bins, density=True, histtype="step", ls="-", lw=1.8, color=C_BLUE,
            label=f"held-out ({len(held)}), median {np.median(held):.2e}")
    top = max(np.histogram(lt, bins, density=True)[0].max(),
              np.histogram(lh, bins, density=True)[0].max())
    ax.set_ylim(0, 1.45 * top)                       # headroom so the legend clears the bins
    ax.set_xlabel(r"$\log_{10}\,\mathcal{E}$"); ax.set_ylabel("probability density")
    ax.set_title(title, fontsize=11); ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)


def _dodge(ys, gap=0.09):
    """Label heights in log10, pushed apart to at least `gap` decades, order kept."""
    order = np.argsort(ys); out = np.array(ys, float)
    for k in range(1, len(order)):
        a, b = order[k - 1], order[k]
        out[b] = max(out[b], out[a] + gap)
    return out


def draw_vschi(ax, train, title):
    ends = []
    for q, col, mk, ls in Q_VSCHI:
        y = [train[key(q, c)] for c in S.CHIS]
        ax.semilogy(S.CHIS, y, ls=ls, lw=2, color=col, marker=mk, ms=8,
                    mec="white", mew=1.2, label=f"q = {q:g}")
        ends.append(np.log10(y[-1]))
    for (q, *_), ly, ly0 in zip(Q_VSCHI, _dodge(ends), ends):
        ax.annotate(f"q = {q:g}", (S.CHIS[-1], 10 ** ly0), xytext=(S.CHIS[-1] + 0.04, 10 ** ly),
                    textcoords="data", va="center", fontsize=9, color="#333333")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:.0e}".replace("e-0", "e-")))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi * 1.5)  # room for the dodged end labels
    ax.set_xlabel(r"$\chi_1$"); ax.set_ylabel(r"$\mathcal{E}$")
    ax.set_xlim(-0.65, 0.95); ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper center"); style(ax)


def draw_wave(model, label):
    fig, axs = plt.subplots(len(Q_WAVE), 2, figsize=(12, 2.6 * len(Q_WAVE)), sharex=True)
    for i, q in enumerate(Q_WAVE):
        w = calibrated(model, q, CHI_WAVE)
        a, b = axs[i]
        a.plot(w["tn"], w["hn"].real, color=C_NR, lw=1.6, label="NR (NRHybSur3dq8), $t$ in $M$")
        a.plot(w["t"], w["h"].real, color=C_RAW, lw=1.2, label="raw ppBHPT, $t$ in $m_1$")
        b.plot(w["tc"], w["hnc"].real, color="#e34948", lw=1.6, ls="--", label="NR (NRHybSur3dq8)")
        b.plot(w["tc"], w["hcal"].real, color="#000000", lw=1.1, label="calibrated ppBHPT")
        a.set_ylabel(r"Re $h_{22}$" + f"\nq = {q:g}")
        b.text(0.02, 0.9, rf"$\mathcal{{E}}$ = {w['E']:.2e}", transform=b.transAxes, fontsize=10)
        for ax in (a, b):
            ax.set_xlim(*XLIM); style(ax)
    axs[0, 0].set_title("before calibration", fontsize=11)
    axs[0, 1].set_title("after calibration", fontsize=11)
    for ax in axs[0]:
        ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2)
    for ax in axs[-1]:
        ax.set_xlabel("$t$")
    fig.suptitle(f"{label}: ppBHPT vs NR at $\\chi_1$ = {CHI_WAVE:+.1f}", fontsize=12, y=1.0)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=list(LOADERS),
                    default=["pn_anchored_spin", "pn_anc_spin_physical"])
    ap.add_argument("--tag", default="pn_spin", help="compare_<tag>/ folder for side-by-sides")
    ap.add_argument("--compare", action="store_true", help="also write compare_<tag>/ side-by-sides")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--nproc", type=int, default=min(64, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    for m in a.models:
        MODELS[m] = LOADERS[m]()
    matplotlib.rcdefaults()     # some fit modules set global rcParams on import; use defaults
    held = heldout_points(); hs = set(held)
    res = {}
    if a.plot_only:
        for m in MODELS:
            res[m] = json.loads((PLOTS / m / "paper" / "heldout_mismatch.json").read_text())
    else:
        jobs = [(m, q, c) for m in MODELS for q, c in held] + \
               [(m, q, c) for m in MODELS for q in S.TRAIN_Q for c in S.CHIS]
        with mp.get_context("fork").Pool(a.nproc) as pool:
            out = pool.map(_work, jobs, chunksize=2)
        for m in MODELS:
            tr = {key(q, c): e for mm, q, c, e in out if mm == m and (q, c) not in hs}
            r = np.array([tr[k] / MODELS[m]["stored"][k] for k in tr])
            print(f"{m}: training recomputed / stored in [{r.min():.6f}, {r.max():.6f}], "
                  f"median {np.median(r):.6f}", flush=True)
            if np.any(np.abs(r - 1) > 1e-2):
                sys.exit(f"{m}: recomputed training scores differ from the stored ones by > 1% "
                         "-- stop and investigate")
            ho = {f"{q:.6f}_{c:+.6f}": e for mm, q, c, e in out if mm == m and (q, c) in hs}
            res[m] = {"heldout": ho, "training": tr,
                      "training_vs_stored": [float(r.min()), float(r.max())],
                      "note": "270 in-range held-out points + the 96 training points recomputed "
                              "on the current code; see plot_spin_paper_figs.py"}
            (PLOTS / m / "paper").mkdir(parents=True, exist_ok=True)
            (PLOTS / m / "paper" / "heldout_mismatch.json").write_text(json.dumps(res[m], indent=2))
    for m, info in MODELS.items():
        trd = res[m].get("training", info["stored"])
        train = np.array(list(trd.values())); ho = np.array(list(res[m]["heldout"].values()))
        print(f"{m}: training mean {train.mean():.4e} median {np.median(train):.4e} max "
              f"{train.max():.4e} | held-out mean {ho.mean():.4e} median {np.median(ho):.4e} "
              f"max {ho.max():.4e}")
        out = PLOTS / m / "paper"
        fig, ax = plt.subplots(figsize=(6, 4.2)); draw_hist(ax, train, ho, info["label"])
        save(fig, out, f"{m}_hist_train_vs_heldout")
        fig, ax = plt.subplots(figsize=(6, 4.2)); draw_vschi(ax, trd, info["label"])
        save(fig, out, f"{m}_mismatch_vs_chi1")
        save(draw_wave(m, info["label"]), out, f"{m}_calibration_chi{CHI_WAVE:+.1f}")
    ms = list(MODELS)
    # Side-by-side compare_<tag>/ figures retired 2026-10-01 (user): every figure lives in its
    # model's own folder.  Pass --compare to make them anyway.
    if len(ms) > 1 and a.compare:
        cmp_ = PLOTS / f"compare_{a.tag}"
        fig, axs = plt.subplots(1, len(ms), figsize=(6 * len(ms), 4.2), sharex=True, sharey=True)
        for ax, m in zip(axs, ms):
            trd = res[m].get("training", MODELS[m]["stored"])
            draw_hist(ax, np.array(list(trd.values())),
                      np.array(list(res[m]["heldout"].values())), MODELS[m]["label"])
        save(fig, cmp_, "hist_train_vs_heldout")
        fig, axs = plt.subplots(1, len(ms), figsize=(6 * len(ms), 4.2), sharey=True)
        for ax, m in zip(axs, ms):
            draw_vschi(ax, res[m].get("training", MODELS[m]["stored"]), MODELS[m]["label"])
        save(fig, cmp_, "mismatch_vs_chi1")


if __name__ == "__main__":
    main()
