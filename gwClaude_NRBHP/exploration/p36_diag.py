"""The installed models' diagnostic figures, for the pass-36 D1 leader (18c).  Same protocols as
spin_dep/plot_spin_diagnostics.py and fig10_tend/fig10_ourmodels.py:

  p36_fig10_tend_q4        Islam+ Fig. 10 method: for each (q = 4, chi1) and each NR window [-4500, t_end], re-fit the
                           model's own alpha(t), beta(t) form for that one case (all 18 coefficients free, Powell from
                           the shipped 18; best valid of refit and start).  E_installed (frozen 18) is stored as well.
  p36_error_by_band_q4     full-window E (frozen 18) split into NR time bands; phi0 is global, so the bands add to E
  p36_residual_vs_time_q4  phase and amplitude residual vs NR time, q = 4, chi1 = -0.6, 0, +0.6
  p36_peak_errors_grid     peak-amplitude error and peak-time shift on the 96 training points
q = 4 waveforms are READ from fig10_tend/{spin_fmt_cache, loss_cache} (the installed models used the same files); the
script refuses to build or rewrite them.  Gate: T96[::8] reproduces the stored per-case scores (ratio 1 within 2e-3).

    python p36_diag.py --compute [--budget MIN]     (compute node)  -> plots_p36/p36_diagnostics.json
    python p36_diag.py --plot
"""
from __future__ import annotations
import argparse, json, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
OUT = HERE / "plots_p36"; RES = OUT / "p36_diagnostics.json"
Q4 = 4.0
Q4_CHIS = [round(-0.6 + 0.1 * k, 1) for k in range(13)]
RES_CHIS = [-0.6, 0.0, 0.6]
T_START = -4500.0
T_ENDS = [75.0, 0.0, -50.0, -100.0, -300.0]
BANDS = [(-5000.1, -1000.0), (-1000.0, -300.0), (-300.0, -100.0), (-100.0, -50.0), (-50.0, 0.0), (0.0, 100.0)]
BAND_LABELS = ["< -1000", "-1000..-300", "-300..-100", "-100..-50", "-50..0", "0..+100"]
LABEL = "pass 36 D1 (18 coef)"
F10_WF, F10_LOSS = ROOT / "fig10_tend" / "spin_fmt_cache", ROOT / "fig10_tend" / "loss_cache"
BUDGET = 25.0
P6 = None


def setup():
    global P6, S, FS, TH, FREE, R
    argv0 = sys.argv[0]; sys.argv = [argv0, "D1"]
    import p36_rdslope as D
    P6 = D.P6; P6.get = D.get                       # BEFORE the fork
    import fit_pn_anchored_spin as S_; import fit_fluxanchored_spin as FS_
    S, FS = S_, FS_
    R = json.loads((HERE / "p36" / "p36_D1.json").read_text())
    TH = np.array(R["theta"], float); FREE = list(R["free"])
    build0, wf0, loss0 = FS.build_case, S.CACHE, FS.LOSSCACHE

    def build_case(q, chi1):                        # q = 4: the fig10_tend caches, read-only
        if abs(float(q) - Q4) > 1e-9:
            return build0(q, chi1)
        w = F10_WF / f"wf_q{q:.6f}_chi{chi1:+.3f}.npz"; l = F10_LOSS / f"loss_q{q:.6f}_chi{chi1:+.3f}.npz"
        if not (w.exists() and l.exists()):
            raise FileNotFoundError(f"q = 4 cache missing for chi1 = {chi1}; not rebuilding")
        S.CACHE, FS.LOSSCACHE = F10_WF, F10_LOSS
        try:
            return build0(q, chi1)
        finally:
            S.CACHE, FS.LOSSCACHE = wf0, loss0
    FS.build_case = build_case


def entry(k):
    c = P6.get(k)
    if "case_full" not in c:
        c["case_full"] = c["case"]
    return c


def window(k, t_end):
    c = entry(k); full = c["case_full"]
    if t_end is None:
        c["case"] = full; return
    cc = dict(full)
    for pre in ("", "fit_"):
        m = (full[pre + "t_nr"] >= T_START) & (full[pre + "t_nr"] <= t_end)
        cc[pre + "t_nr"] = full[pre + "t_nr"][m]; cc[pre + "h_nr"] = full[pre + "h_nr"][m]
    c["case"] = cc


def calibrated(th, k):
    """E, NR times, h_NR and the calibrated model on them (phi0 applied), aligned as P6._full."""
    _, t0 = P6.err(th, k, "fit"); e, t1 = P6.err(th, k, "full", seed=t0)
    c = P6.get(k); case = c["case"]; t, h, tn, hn = case["t_bhpt"], case["h_bhpt"], case["t_nr"], case["h_nr"]
    alpha, beta = P6.ab(th, k, "full")
    bc = P6.creative.cumulative_trapezoid(beta, t); tau = t1 + bc - np.interp(S.T_ANCHOR, t, bc)
    m = (tn >= max(tn[0], tau[0])) & (tn <= min(tn[-1], tau[-1])); tc, r = tn[m], hn[m]
    g = np.interp(tc, tau, alpha) * (np.interp(tc, tau, h.real) + 1j * np.interp(tc, tau, h.imag))
    g = g * np.exp(1j * np.angle(np.sum(r * np.conj(g)))); n1 = np.sum(np.abs(r) ** 2)
    e2 = float(np.sum(np.abs(r - g) ** 2) / (2 * n1))
    assert abs(e2 / e - 1) < 1e-6, (k, e, e2)
    return e, tc, r, g, n1


def peak(t, h):
    a = np.abs(h); i = int(np.argmax(a)); i = min(max(i, 1), len(a) - 2)
    y0, y1, y2 = a[i - 1], a[i], a[i + 1]; d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return t[i] + d * (t[1] - t[0]), y1 - 0.25 * (y0 - y2) * d


def refit(k, budget_min):
    from scipy.optimize import minimize
    st = dict(best=np.inf, x=TH.copy(), t=time.time())

    class Stop(Exception):
        pass

    def obj(z):
        th = TH.copy(); th[FREE] = z
        v = P6.err(th, k, "fit")[0]
        if v < st["best"]:
            st["best"], st["x"] = v, th
        if time.time() - st["t"] > budget_min * 60:
            raise Stop
        return v
    try:
        minimize(obj, TH[FREE], method="Powell", options={"xtol": 1e-6, "ftol": 1e-9, "maxiter": 200})
    except Stop:
        pass
    return st["x"]


def work(task):
    kind, k, te = task
    if kind == "f10":
        window(k, te)
        e_inst = P6._full((list(TH), k))
        th = refit(k, BUDGET)
        e_fit = P6._full((list(th), k))
        out = dict(kind=kind, chi1=k[1], t_end=te, E_installed=float(e_inst), E_refit=float(min(e_fit, e_inst)),
                   refit_improved=bool(e_fit < e_inst))
    elif kind == "band":
        window(k, None)
        e, tc, r, g, n1 = calibrated(TH, k)
        b = [float(np.sum(np.abs((r - g)[(tc >= lo) & ((tc < hi) if hi < 100 else (tc <= hi))]) ** 2) / (2 * n1))
             for lo, hi in BANDS]
        assert abs(sum(b) / e - 1) < 1e-8, (k, e, sum(b))
        out = dict(kind=kind, chi1=k[1], E=float(e), bands=b)
        if k[1] in RES_CHIS:
            s = slice(None, None, 5); dphi = np.unwrap(np.angle(g * np.conj(r)))
            out["series"] = dict(t=tc[s].tolist(), dphi=dphi[s].tolist(),
                                 damp_pk=((np.abs(g) - np.abs(r)) / np.max(np.abs(r)))[s].tolist())
    else:
        window(k, None)
        e, tc, r, g, _ = calibrated(TH, k)
        tp_nr, ap_nr = peak(tc, r); tp_m, ap_m = peak(tc, g)
        out = dict(kind=kind, q=k[0], chi1=k[1], E=float(e), dt_peak=float(tp_m - tp_nr), damp_peak=float(ap_m / ap_nr - 1))
    return out


def compute():
    setup()
    for c in Q4_CHIS:                                   # cache presence before forking
        if not (F10_WF / f"wf_q{Q4:.6f}_chi{c:+.3f}.npz").exists():
            sys.exit(f"missing q = 4 cache chi1 = {c}")
    tasks = [("f10", (Q4, c), te) for c in Q4_CHIS for te in T_ENDS]
    tasks += [("band", (Q4, c), None) for c in Q4_CHIS] + [("peak", k, None) for k in P6.T96]
    t0 = time.time()
    with mp.get_context("fork").Pool(96) as pool:
        chk = P6.T96[::8]
        e = np.array(pool.map(P6._full, [(list(TH), k) for k in chk]))
        r = e / np.array([R["T96_cases"][f"{q:.6f}_{c:+.4f}"] for q, c in chk])
        print(f"reproduce stored T96: {r.min():.4f}..{r.max():.4f}", flush=True)
        assert np.allclose(r, 1, atol=2e-3), r
        res = []
        for out in pool.imap_unordered(work, tasks, chunksize=1):
            res.append(out)
            if out["kind"] == "f10":
                print(f"  f10 chi1 {out['chi1']:+.1f} t_end {out['t_end']:+5.0f}: refit {out['E_refit']:.3e} "
                      f"installed {out['E_installed']:.3e}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pk = [o for o in res if o["kind"] == "peak"]
    rr = np.array([o["E"] / R["T96_cases"][f"{o['q']:.6f}_{o['chi1']:+.4f}"] for o in pk])
    print(f"peak-grid E vs stored T96: {rr.min():.4f}..{rr.max():.4f}", flush=True)
    assert np.allclose(rr, 1, atol=2e-3), rr
    RES.write_text(json.dumps(dict(note="pass 36 D1 diagnostics; f10 = per-case refit on NR [-4500, t_end]; band/peak "
                                        "= frozen 18c, full window", bands=BAND_LABELS, budget_min=BUDGET, results=res)))
    print(f"wrote {RES.name}", flush=True)


def plot():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.ticker
    import matplotlib.pyplot as plt
    Rr = json.loads(RES.read_text())["results"]

    def grey():
        plt.rcdefaults()
        plt.rcParams.update({"axes.facecolor": "#EAEAF2", "axes.edgecolor": "white", "axes.grid": True,
                             "grid.color": "white", "grid.linewidth": 1.0, "axes.axisbelow": True,
                             "xtick.major.size": 0, "ytick.major.size": 0})

    def save(fig, name):
        for e in ("png", "pdf"):
            fig.savefig(OUT / f"{name}.{e}", dpi=150)
        plt.close(fig); print(f"  wrote plots_p36/{name}.{{png,pdf}}", flush=True)

    # ---- Fig. 10: E22 vs chi1 per window end (per-case refit, as fig10_tend/plot_fig10_ourmodels.py)
    grey(); COL = {75.0: "blue", 0.0: "red", -50.0: "green", -100.0: "cyan", -300.0: "black"}
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for te in T_ENDS:
        rr = sorted([r for r in Rr if r["kind"] == "f10" and r["t_end"] == te], key=lambda r: r["chi1"])
        ax.semilogy([r["chi1"] for r in rr], [r["E_refit"] for r in rr], "o-", color=COL[te], label=rf"$t_{{\rm end}}={te:.0f}M$")
    ax.set_xlabel(r"$\chi_1$", fontsize=14); ax.set_ylabel(r"$\mathcal{E}_{22}$", fontsize=14)
    ax.set_title(f"{LABEL},  q = 4", fontsize=13, pad=34)
    ax.legend(ncol=5, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.5,
              columnspacing=0.9)
    fig.tight_layout(); save(fig, "p36_fig10_tend_q4")

    # ---- residual vs time
    grey()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True); lim, off = 0.0, []
    for chi, col in zip(RES_CHIS, ("tab:blue", "black", "tab:red")):
        s = next(r for r in Rr if r["kind"] == "band" and r["chi1"] == chi)["series"]
        t = np.array(s["t"]); dphi = np.array(s["dphi"])
        a1.plot(t, dphi, color=col, lw=1.3, label=rf"$\chi_1={chi:+.1f}$")
        a2.plot(t, 100 * np.array(s["damp_pk"]), color=col, lw=1.3)
        lim = max(lim, np.max(np.abs(dphi[t <= 30.0]))); off.append((chi, float(dphi[np.argmax(np.abs(dphi))]), col))
    ylim = 1.3 * lim + 0.1; a1.set_ylim(-ylim, ylim)
    for j, (chi, v, col) in enumerate([o for o in off if abs(o[1]) > ylim]):
        a1.text(0.02, 0.92 - 0.1 * j, rf"$\chi_1={chi:+.1f}$ reaches {v:+.0f} rad by +100 M (off scale)",
                transform=a1.transAxes, fontsize=9, color=col, va="top", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ticks = [-5000, -1000, -300, -100, -50, -20, 0, 20, 50, 100]
    for a in (a1, a2):
        a.set_xscale("symlog", linthresh=50, linscale=1.0); a.axvline(0, color="grey", ls="--", lw=0.8)
        a.set_xticks(ticks); a.set_xticklabels([str(v) for v in ticks]); a.set_xlim(-5000, 100)
        a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a1.set_ylabel(r"$\Delta\phi$  (rad)", fontsize=12)
    a2.set_ylabel(r"$(|h_{\rm model}|-|h_{\rm NR}|)\,/\,|h_{\rm NR}|_{\rm peak}$  (%)", fontsize=11)
    a2.set_xlabel(r"$t$  (M, NR peak at 0; logarithmic before $-50$, linear after)", fontsize=11)
    a1.legend(ncol=3, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    a1.set_title(f"{LABEL}, q = 4: residual vs time", fontsize=12, pad=26)
    fig.tight_layout(); save(fig, "p36_residual_vs_time_q4")

    # ---- error by band
    grey()
    rr = sorted([r for r in Rr if r["kind"] == "band"], key=lambda r: r["chi1"])
    chis = np.array([r["chi1"] for r in rr]); E = np.array([r["E"] for r in rr])
    B = np.array([r["bands"] for r in rr]); frac = B / B.sum(axis=1, keepdims=True)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.6]))
    a1.semilogy(chis, E, "ko-"); a1.set_ylabel(r"$\mathcal{E}_{22}$ (full window)", fontsize=11)
    cols = plt.cm.viridis(np.linspace(0.05, 0.95, len(BAND_LABELS))); bot = np.zeros(len(chis))
    for j, lab in enumerate(BAND_LABELS):
        a2.bar(chis, 100 * frac[:, j], width=0.08, bottom=bot, color=cols[j], label=f"{lab} M"); bot += 100 * frac[:, j]
    a2.set_ylim(0, 100); a2.set_ylabel("share of the error (%)", fontsize=11)
    a2.set_xlabel(r"$\chi_1$", fontsize=13); a2.set_xticks(chis)
    a2.legend(ncol=3, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, title="time band",
              title_fontsize=8.5)
    a1.set_title(f"{LABEL}, q = 4: where the error comes from", fontsize=12)
    fig.tight_layout(); save(fig, "p36_error_by_band_q4")

    # ---- peak errors on the training grid
    plt.rcdefaults()
    G = [r for r in Rr if r["kind"] == "peak"]; qs, cs = sorted({r["q"] for r in G}), sorted({r["chi1"] for r in G})
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
        ax.set_xlabel("q"); ax.set_ylabel(r"$\chi_1$"); ax.set_title(lab, fontsize=11); fig.colorbar(im, ax=ax, shrink=0.9)
    fig.suptitle(f"{LABEL}: calibrated (2,2) peak vs NRHybSur3dq8, training grid", fontsize=11)
    fig.tight_layout(); save(fig, "p36_peak_errors_grid")


def main():
    global BUDGET
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute", action="store_true"); ap.add_argument("--plot", action="store_true")
    ap.add_argument("--budget", type=float, default=BUDGET)
    a = ap.parse_args(); BUDGET = a.budget
    if a.compute:
        compute()
    if a.plot:
        plot()


if __name__ == "__main__":
    main()
