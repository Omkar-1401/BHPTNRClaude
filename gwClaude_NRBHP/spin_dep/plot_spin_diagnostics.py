"""Three diagnostics for the four installed spin models (shipped coefficients, full window
[-5000.1, +100] M, each model's own alpha(t), beta(t); t0 and phi0 optimised as in its scoring):

  residual_vs_time_q4   phase error dphi(t) = arg(h_model conj h_NR) and amplitude error
                        |h_model|/|h_NR| - 1 vs NR time, q = 4, chi1 = -0.6, 0, +0.6
  error_by_band_q4      mathcalE split into time bands (the band terms add up to mathcalE
                        exactly: phi0 is global), q = 4, chi1 = -0.6 ... +0.6
  peak_errors_grid      peak-amplitude error and peak-time shift of the calibrated (2,2)
                        mode on the 96-point training grid

Outputs go in each model's own folder: Agentic_spin_plots/<model>/<name>.{png,pdf}.
Numbers in diagnostics_results/ (compute step only).

Gate: the training-grid mathcalE recomputed here must equal each model's stored per-case
values to 1% (ratio printed); the compute step stops otherwise.  Caches are only READ: q = 4
cases come from fig10_tend/ (exact q = 4, built for the Fig. 10 study), training-grid cases
from spin_dep/; a missing file or a stale drive stops the script instead of rebuilding.

  python plot_spin_diagnostics.py --compute     (compute node)
  python plot_spin_diagnostics.py --plot
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from multiprocessing import get_context

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "fig10_tend")); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import fig10_ourmodels as F                    # installed(), ab(), score(), STORED: the models' own code
S, FS = F.S, F.FS
import physics_common as PC

OUTD = HERE / "diagnostics_results"
PLOTS = HERE / "Agentic_spin_plots"
NAME = {"pn13": "pn_anchored_spin", "phys11": "pn_anc_spin_physical",
        "flux9": "fluxanchored_spin", "en8": "energy_anchored_spin"}
Q4_CHIS = [round(-0.6 + 0.1 * k, 1) for k in range(13)]
RES_CHIS = [-0.6, 0.0, 0.6]
BANDS = [(-5000.1, -1000.0), (-1000.0, -300.0), (-300.0, -100.0), (-100.0, -50.0), (-50.0, 0.0), (0.0, 100.0)]
BAND_LABELS = ["< -1000", "-1000..-300", "-300..-100", "-100..-50", "-50..0", "0..+100"]
CACHE_DIRS = {"q4": (ROOT / "fig10_tend" / "spin_fmt_cache", ROOT / "fig10_tend" / "loss_cache"),
              "grid": (HERE / "spin_wf_cache", HERE / "flux_loss_cache")}


def get_case(q, chi, where):
    wf, ls = CACHE_DIRS[where]
    w = wf / f"wf_q{q:.6f}_chi{chi:+.3f}.npz"; l = ls / f"loss_q{q:.6f}_chi{chi:+.3f}.npz"
    if not w.exists() or not l.exists():
        sys.exit(f"missing cache for q={q} chi={chi} in {where}: {w.exists()=} {l.exists()=}")
    z = np.load(l)
    if "drive" not in z.files or str(z["drive"]) != PC.ENERGY_DRIVE_2D:
        sys.exit(f"stale energy drive in {l}; not rebuilding")
    S.CACHE, FS.LOSSCACHE = wf, ls
    return FS.build_case(q, chi)


def calibrated(model, q, chi, case):
    """mathcalE, NR times, h_NR and the calibrated model on them (phi0 applied)."""
    p = F.installed(model, q, chi)
    _, t0 = F.score(model, p, q, chi, case, "fit", -50.0, wide=True)
    e, t0 = F.score(model, p, q, chi, case, "full", t0, wide=True)
    t, h, tn, hn = case["t_bhpt"], case["h_bhpt"], case["t_nr"], case["h_nr"]
    alpha, beta = F.ab(model, p, q, chi, case, "full")
    tau = t0 + F.tau_shape(beta, t)
    m = (tn >= max(tn[0], tau[0])) & (tn <= min(tn[-1], tau[-1]))
    tc, r = tn[m], hn[m]
    g = np.interp(tc, tau, alpha) * (np.interp(tc, tau, h.real) + 1j * np.interp(tc, tau, h.imag))
    g = g * np.exp(1j * np.angle(np.sum(r * np.conj(g))))
    n1 = np.sum(np.abs(r) ** 2)
    e_chk = float(np.sum(np.abs(r - g) ** 2) / (2 * n1))
    assert abs(e_chk / e - 1) < 1e-8, (model, q, chi, e, e_chk)       # same number as the scorer
    return e, tc, r, g, n1


def peak(t, h):
    a = np.abs(h); i = int(np.argmax(a)); i = min(max(i, 1), len(a) - 2)
    y0, y1, y2 = a[i - 1], a[i], a[i + 1]; d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return t[i] + d * (t[1] - t[0]), y1 - 0.25 * (y0 - y2) * d


def work(task):
    kind, model, q, chi = task
    case = get_case(q, chi, "q4" if kind == "q4" else "grid")
    e, tc, r, g, n1 = calibrated(model, q, chi, case)
    out = dict(kind=kind, model=model, q=q, chi1=chi, E=e)
    if kind == "q4":
        out["bands"] = [float(np.sum(np.abs((r - g)[(tc >= lo) & (tc < hi if hi < 100 else tc <= hi)]) ** 2)
                              / (2 * n1)) for lo, hi in BANDS]
        assert abs(sum(out["bands"]) / e - 1) < 1e-8, (model, chi, e, sum(out["bands"]))
        if chi in RES_CHIS:
            k = slice(None, None, 5)                                    # 0.5 M spacing for plotting
            dphi = np.unwrap(np.angle(g * np.conj(r)))
            out["series"] = dict(t=tc[k].tolist(), dphi=dphi[k].tolist(),
                                 damp=(np.abs(g) / np.abs(r) - 1)[k].tolist(),
                                 damp_pk=((np.abs(g) - np.abs(r)) / np.max(np.abs(r)))[k].tolist())
    else:
        tp_nr, ap_nr = peak(tc, r); tp_m, ap_m = peak(tc, g)
        out.update(dt_peak=float(tp_m - tp_nr), damp_peak=float(ap_m / ap_nr - 1))
    return out


def compute(nproc):
    tasks = [("q4", m, 4.0, c) for m in NAME for c in Q4_CHIS]
    tasks += [("grid", m, q, c) for m in NAME for q in S.TRAIN_Q for c in S.CHIS]
    for kind, m, q, c in tasks:                                           # cache checks before forking
        get_case(q, c, "q4" if kind == "q4" else "grid")
    with get_context("fork").Pool(nproc) as pool:
        res = pool.map(work, tasks, chunksize=1)
    bad = 0                                                               # reproduction gate
    for m in NAME:
        st = json.loads(Path(F.STORED[m]).read_text())["mismatches"]
        rs = []
        for r in res:
            if r["kind"] != "grid" or r["model"] != m:
                continue
            key = next(k for k in st if abs(float(k.split("_")[0]) - r["q"]) < 1e-3
                       and abs(float(k.split("_")[1]) - r["chi1"]) < 1e-6)
            rs.append(r["E"] / st[key])
        rs = np.array(rs); ok = np.all(np.abs(rs - 1) <= 0.01); bad += not ok
        print(f"GATE {m}: {len(rs)} grid cases, ratio min {rs.min():.5f} max {rs.max():.5f}  "
              f"{'ok' if ok else 'FAIL'}", flush=True)
    if bad:
        sys.exit("gate failed: not writing results")
    OUTD.mkdir(exist_ok=True)
    (OUTD / "diagnostics.json").write_text(json.dumps(dict(
        note="installed coefficients, full window; E = repo mathcalE; bands add up to E; "
             "dt_peak = model - NR (M); damp_peak = |h|peak ratio - 1", bands=BAND_LABELS, results=res)))
    print("wrote", OUTD / "diagnostics.json")


def style():
    import matplotlib.pyplot as plt
    plt.rcdefaults()                      # gw_remnant's plotter changes rcParams on import (CLAUDE.md)
    plt.rcParams.update({"axes.facecolor": "#EAEAF2", "axes.edgecolor": "white", "axes.grid": True,
                         "grid.color": "white", "grid.linewidth": 1.0, "axes.axisbelow": True,
                         "xtick.major.size": 0, "ytick.major.size": 0})


def save(fig, m, name):
    for e in ("png", "pdf"):
        fig.savefig(PLOTS / NAME[m] / f"{name}.{e}", dpi=150)


def plot():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.ticker
    import matplotlib.pyplot as plt
    R = json.loads((OUTD / "diagnostics.json").read_text())["results"]
    for m in NAME:
        # ---- residual vs time
        style()
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True)
        lim, off = 0.0, []
        for chi, col in zip(RES_CHIS, ("tab:blue", "black", "tab:red")):
            s = next(r for r in R if r["kind"] == "q4" and r["model"] == m and r["chi1"] == chi)["series"]
            t = np.array(s["t"]); dphi = np.array(s["dphi"])
            a1.plot(t, dphi, color=col, lw=1.3, label=rf"$\chi_1={chi:+.1f}$")
            a2.plot(t, 100 * np.array(s["damp_pk"]), color=col, lw=1.3)
            lim = max(lim, np.max(np.abs(dphi[t <= 30.0])))      # axis set by inspiral + merger
            off.append((chi, float(dphi[np.argmax(np.abs(dphi))]), col))
        ylim = 1.3 * lim + 0.1
        a1.set_ylim(-ylim, ylim)
        off = [o for o in off if abs(o[1]) > ylim]                # only curves that really leave it
        for k, (chi, v, col) in enumerate(off):          # say what runs off the axis, do not hide it
            a1.text(0.02, 0.92 - 0.1 * k, rf"$\chi_1={chi:+.1f}$ reaches {v:+.0f} rad by +100 M (off scale)",
                    transform=a1.transAxes, fontsize=9, color=col, va="top",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        ticks = [-5000, -1000, -300, -100, -50, -20, 0, 20, 50, 100]
        for a in (a1, a2):
            a.set_xscale("symlog", linthresh=50, linscale=1.0); a.axvline(0, color="grey", ls="--", lw=0.8)
            a.set_xticks(ticks); a.set_xticklabels([str(v) for v in ticks]); a.set_xlim(-5000, 100)
            a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        a1.set_ylabel(r"$\Delta\phi$  (rad)", fontsize=12)
        a2.set_ylabel(r"$(|h_{\rm model}|-|h_{\rm NR}|)\,/\,|h_{\rm NR}|_{\rm peak}$  (%)", fontsize=11)
        a2.set_xlabel(r"$t$  (M, NR peak at 0; logarithmic before $-50$, linear after)", fontsize=11)
        a1.legend(ncol=3, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
        a1.set_title(f"{NAME[m]}, q = 4: residual vs time (installed coefficients)", fontsize=12, pad=26)
        fig.tight_layout(); save(fig, m, "residual_vs_time_q4"); plt.close(fig)

        # ---- error by band
        style()
        rr = sorted([r for r in R if r["kind"] == "q4" and r["model"] == m], key=lambda r: r["chi1"])
        chis = np.array([r["chi1"] for r in rr]); E = np.array([r["E"] for r in rr])
        B = np.array([r["bands"] for r in rr]); frac = B / B.sum(axis=1, keepdims=True)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 6.6), sharex=True,
                                     gridspec_kw=dict(height_ratios=[1, 1.6]))
        a1.semilogy(chis, E, "ko-"); a1.set_ylabel(r"$\mathcal{E}_{22}$ (full window)", fontsize=11)
        cols = plt.cm.viridis(np.linspace(0.05, 0.95, len(BAND_LABELS)))
        bot = np.zeros(len(chis))
        for j, lab in enumerate(BAND_LABELS):
            a2.bar(chis, 100 * frac[:, j], width=0.08, bottom=bot, color=cols[j], label=f"{lab} M")
            bot += 100 * frac[:, j]
        a2.set_ylim(0, 100); a2.set_ylabel("share of the error (%)", fontsize=11)
        a2.set_xlabel(r"$\chi_1$", fontsize=13); a2.set_xticks(chis)
        a2.legend(ncol=3, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
                  title="time band", title_fontsize=8.5)
        a1.set_title(f"{NAME[m]}, q = 4: where the error comes from (installed coefficients)", fontsize=12)
        fig.tight_layout(); save(fig, m, "error_by_band_q4"); plt.close(fig)

        # ---- peak errors on the training grid
        plt.rcdefaults()
        G = [r for r in R if r["kind"] == "grid" and r["model"] == m]
        qs, cs = sorted({r["q"] for r in G}), sorted({r["chi1"] for r in G})
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
        for ax, key, lab, fac, fmt in ((axs[0], "damp_peak", "peak amplitude error (%)", 100, "{:+.1f}"),
                                       (axs[1], "dt_peak", "peak time shift, model - NR (M)", 1, "{:+.1f}")):
            Z = np.full((len(cs), len(qs)), np.nan)
            for r in G:
                Z[cs.index(r["chi1"]), qs.index(r["q"])] = fac * r[key]
            v = np.nanmax(np.abs(Z))
            im = ax.imshow(Z, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
            for i in range(len(cs)):
                for j in range(len(qs)):
                    ax.text(j, i, fmt.format(Z[i, j]), ha="center", va="center", fontsize=6.5,
                            color="white" if abs(Z[i, j]) > 0.6 * v else "black")
            ax.set_xticks(range(len(qs))); ax.set_xticklabels([f"{q:.2f}" for q in qs], rotation=60, fontsize=8)
            ax.set_yticks(range(len(cs))); ax.set_yticklabels([f"{c:+.1f}" for c in cs], fontsize=8)
            ax.set_xlabel("q"); ax.set_ylabel(r"$\chi_1$"); ax.set_title(lab, fontsize=11)
            fig.colorbar(im, ax=ax, shrink=0.9)
        fig.suptitle(f"{NAME[m]}: calibrated (2,2) peak vs NRHybSur3dq8, training grid (installed coefficients)",
                     fontsize=11)
        fig.tight_layout(); save(fig, m, "peak_errors_grid"); plt.close(fig)
        print("plotted", NAME[m])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute", action="store_true"); ap.add_argument("--plot", action="store_true")
    ap.add_argument("--nproc", type=int, default=100)
    a = ap.parse_args()
    if a.compute:
        compute(a.nproc)
    if a.plot:
        plot()


if __name__ == "__main__":
    main()
